# DeepSeek-V3 아키텍처 해설

> Phase 10 · Lesson 14는 모든 open model이 조정하는 여섯 가지 architectural knob을 이름 붙였습니다. DeepSeek-V3(2024년 12월, 총 671B parameters, 37B active)는 그 여섯 가지를 모두 조정하고 네 가지를 더 추가합니다. Multi-Head Latent Attention, auxiliary-loss-free load balancing, Multi-Token Prediction, DualPipe training입니다. 이 lesson은 DeepSeek-V3 architecture를 위에서 아래까지 읽고, 공개된 config에서 모든 parameter count를 유도합니다. 끝나면 왜 671B/37B 비율이 올바른 베팅인지, 그리고 왜 MLA + MoE 조합이 frontier에서 둘 중 하나만 쓰는 것보다 나은지 설명할 수 있습니다.

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 minutes

## 학습 목표

- DeepSeek-V3 config를 위에서 아래까지 읽고 각 field를 여섯 GPT-2 knob과 네 DeepSeek-specific addition 관점에서 설명할 수 있습니다.
- total parameter count(671B), active parameter count(37B), 그리고 각 값에 기여하는 component를 유도할 수 있습니다.
- 128k context에서 MLA의 KV cache footprint를 계산하고, 같은 active-param 규모의 dense model이 GQA로 지불할 비용과 비교할 수 있습니다.
- 네 DeepSeek-specific innovation(MLA, MTP, auxiliary-loss-free routing, DualPipe)을 말하고 각각이 architecture/training stack의 어느 부분을 겨냥하는지 이름 붙일 수 있습니다.

## 문제

DeepSeek-V3는 Llama family와 architecture가 의미 있게 다른 첫 frontier open model입니다. Llama 3 405B는 "여섯 knob을 조정한 GPT-2"입니다. DeepSeek-V3는 여섯 knob을 모두 조정하고 네 개를 더한 GPT-2입니다. Llama 3 config를 읽는 것은 DeepSeek config를 읽기 위한 warmup이지만, 깊은 구조, 즉 attention block의 형태, routing logic, training-time objective가 충분히 다르므로 별도의 walkthrough가 필요합니다.

이를 배우는 보상은 큽니다. DeepSeek-V3의 open-weights release는 open model에서 "frontier capability"가 의미하는 바를 바꿨습니다. 이 architecture는 많은 2026년 training run이 복사하는 blueprint입니다. 이를 이해하는 것은 frontier LLM training이나 inference를 다루는 어떤 역할에도 기본 요건입니다.

## 개념

### 다시 보는 변하지 않는 핵심

DeepSeek-V3는 여전히 autoregressive입니다. 여전히 decoder block을 쌓습니다. 각 block에는 여전히 attention과 MLP, 두 RMSNorm이 있습니다. MLP에서는 여전히 SwiGLU를 씁니다. 여전히 RoPE를 씁니다. Pre-norm. Weight-tied embeddings. 모든 Llama나 Mistral과 같은 baseline입니다.

### Twist: GQA 대신 MLA

Phase 10 · 14에서 GQA는 K와 V를 Q head group 전반에 공유해 KV cache를 줄인다는 것을 배웠습니다. Multi-Head Latent Attention(MLA)은 더 나아갑니다. K와 V를 공유 low-rank latent representation(`kv_lora_rank`)으로 압축한 뒤, head별로 즉시 decompression합니다. KV cache는 latent만 저장합니다. 보통 token당 layer당 512 floats이지, 8 x 128 = 1024 floats가 아닙니다.

128k context에서 DeepSeek-V3 with MLA(token당 layer당 공유 latent `c^{KV}` 하나, K와 V는 이후 matmul에 흡수될 수 있는 up-projection을 통해 이 latent에서 유도됨):

```text
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

가상의 GQA baseline(Llama 3 70B shape, 8 KV heads, head dim 128)은 다음 비용을 냅니다:

```text
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

MLA는 128k context에서 Llama-3-70B-style GQA cache보다 4x 작습니다.

tradeoff: MLA는 attention computation마다(head마다) decompression step을 추가합니다. 추가 compute는 절약되는 bandwidth에 비해 작습니다. long-context inference에서는 순이익입니다.

### Routing: auxiliary-loss-free load balancing

MoE router는 각 token을 처리할 top-k expert를 결정합니다. naive router는 일을 일부 expert에 너무 집중시켜 다른 expert를 idle 상태로 둡니다. 표준 해법은 load imbalance를 벌주는 auxiliary loss term을 추가하는 것입니다. 이는 작동하지만 main-task performance를 약간 떨어뜨립니다.

DeepSeek-V3는 auxiliary-loss-free scheme을 도입합니다. expert별 bias term을 router logit에 더하고, training 중 간단한 규칙으로 조정합니다. expert `e`가 overloaded이면 `bias_e`를 낮추고, underloaded이면 높입니다. 추가 loss term이 없습니다. training은 깨끗하게 유지됩니다. expert load는 균형을 유지합니다.

main loss에 대한 효과: 측정 가능한 손상 없음. MoE architecture에 대한 효과: 더 깔끔하고, 조정할 auxiliary-loss hyperparameter가 없습니다.

### MTP: 더 dense한 training과 무료 draft

Phase 10 · 18에서 배웠듯이 DeepSeek-V3는 두 위치 앞 token을 예측하는 D=1 MTP module을 추가합니다. inference에서는 학습된 module이 acceptance 80%+의 speculative-decoding draft로 재사용됩니다. training에서는 각 hidden state가 D+1 = 2개 target으로 supervised되어 더 조밀한 signal을 제공합니다.

Parameters: 671B main 위에 14B. Overhead: 2.1%.

### Training: DualPipe

Phase 10 · 19에서 배웠듯이 DualPipe는 forward/backward chunk를 노드 간 all-to-all comms와 겹치는 bidirectional pipeline입니다. DeepSeek-V3의 2,048-H800 scale에서는 1F1B가 pipeline bubble로 잃었을 약 245k GPU-hour를 회수합니다.

### Config를 field별로 읽기

DeepSeek-V3 config(단순화)는 다음과 같습니다:

```yaml
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

이를 해석하면:

- `hidden_size=7168`: embedding dimension.
- `num_hidden_layers=61`: total block depth.
- `first_k_dense_layers=3`: 첫 3 block은 size 18432의 dense MLP를 사용합니다. 나머지 58개는 MoE를 사용합니다.
- `num_attention_heads=128`: 128 query heads.
- `kv_lora_rank=512`: K와 V가 이 latent dimension으로 압축되고 head별로 decompression됩니다.
- `num_experts=256, num_experts_per_tok=8`: 각 MoE block에는 256 expert가 있고 top-8로 route합니다.
- `shared_experts=1`: 256 routed expert 위에 1개의 always-on expert가 모든 token에 기여합니다. 모든 token이 안정적인 무언가를 받도록 보장하는 "dense floor"라고 생각하세요.
- `moe_intermediate_size=2048`: 각 expert의 MLP hidden size. 256개가 있으므로 dense MLP보다 작습니다.

### Parameter accounting

전체 계산은 `code/main.py`에 있습니다. 핵심 숫자:

- Embedding: `vocab * hidden = 129280 * 7168 = ~0.93B`.
- 첫 3개 dense block: MLA가 있는 attention(block당 ~144M) + dense MLP(block당 ~260M) + norms. 총 약 1.2B.
- 58개 MoE block: MLA가 있는 attention(~144M) + 각각 30M인 256 experts + 1 shared expert(30M) + norm. 모든 expert를 포함하면 block당 총 ~7.95B. 58개 MoE block 전체는 461B.
- MTP module: 14B.

Grand total: core architecture 약 476B + MTP 14B + 공개된 671B 숫자는 추가 structural parameter(bias tensor, expert-specific component, shared expert scaling 등)를 별도로 계상합니다. calculator가 재현하는 숫자는 공개값의 3-5% 이내입니다. 차이는 DeepSeek report가 Section 2 appendix에서 문서화한 fine-grained accounting에서 옵니다.

forward당 active parameters:

- Attention: layer당 144M * 61 = 8.8B(모든 layer가 실행됨).
- MLP active: 첫 3개 layer는 dense(3 * 260M = 780M), 58개 MoE layer는 각각 8 routed + 1 shared + routing overhead가 active입니다. layer당 active MLP: ~260M. 총: 3 * 260M + 58 * 260M = ~15.9B.
- Embedding + norms: 1.2B.
- Total active: 대략 26B core + 14B MTP(training되지만 inference에서 항상 실행되지는 않음) ≈ 37B.

### 671B / 37B 비율

18x sparsity ratio(active params는 total의 5.5%). DeepSeek-V3는 open weights로 출시된 frontier MoE model 중 가장 sparse합니다. Mixtral 8x7B는 13/47(28%) 비율로 훨씬 dense합니다. Llama 4 Maverick의 17B/400B(4.25%)는 비슷합니다. DeepSeek의 베팅: frontier scale에서는 더 많은 expert와 더 낮은 activation ratio가 active-FLOP당 더 나은 품질을 만든다는 것입니다.

### DeepSeek-V3의 위치

| Model | Total | Active | Ratio | Attention | 새로운 아이디어 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN extension |

### 후속: R1, V4

DeepSeek-R1(2025)은 V3 backbone 위에서 수행한 reasoning-training run입니다. R1은 같은 architecture를 씁니다. 바뀐 것은 pretraining architecture가 아니라 post-training recipe(verifiable task에 대한 large-scale RL)입니다.

DeepSeek-V4(출시된다면)는 MLA + MoE + MTP를 유지하고 Phase 10 · 17의 NSA 후속인 DSA(DeepSeek Sparse Attention)를 추가할 것으로 예상됩니다. 계보는 안정적입니다. architecture-level innovation은 축적되고, 각 version은 추가 knob을 조정합니다.

```figure
moe-routing
```

## 활용하기

`code/main.py`는 DeepSeek-V3 shape에 특화된 parameter calculator입니다. 실행해서 출력을 논문 숫자와 비교하고, 가상의 variant(256 experts vs 512, top-8 vs top-16, MLA rank 512 vs 1024)에 사용하세요.

볼 것:

- Total parameter count vs published 671B.
- Active parameter count vs published 37B.
- 128k context에서 KV cache, 즉 MLA vs GQA 비교.
- parameter budget이 실제로 어디에 쓰이는지 보여주는 per-layer breakdown.

## 산출물

이 lesson은 `outputs/skill-deepseek-v3-reader.md`를 산출합니다. DeepSeek-family model(V3, R1 또는 future variant)이 주어지면 config의 각 field를 이름 붙이고, component별 parameter count를 유도하며, 모델이 네 DeepSeek-specific innovation 중 무엇을 사용하는지 식별하는 component-by-component architecture reading을 산출합니다.

## 연습문제

1. `code/main.py`를 실행하세요. calculator의 total-parameter estimate를 공개된 671B와 비교하고 delta가 어디에서 오는지 식별하세요. 논문의 Section 2에 전체 itemization이 있습니다.

2. config를 수정해 MLA rank 512 대신 256을 사용하세요. 128k context에서 resulting KV cache size를 계산하세요. 몇 퍼센트 감소를 얻고, per-head expressiveness에 어떤 비용을 치르나요?

3. DeepSeek-V3의 (256 experts, top-8) routing을 가상의 (512 experts, top-8) variant와 비교하세요. Total parameters는 늘고 active parameters는 그대로입니다. 추가 expert capacity는 이론상 무엇을 제공하고, inference에서 무엇을 비용으로 지불하나요?

4. MLA에 대한 DeepSeek-V3 technical report(arXiv:2412.19437)의 Section 2.1을 읽으세요. K와 V decompression matrix가 inference-time efficiency를 위해 이후 matmul에 "absorbed"될 수 있는 이유를 세 문장으로 설명하세요.

5. DeepSeek-V3는 대부분의 operation에 FP8 training을 사용합니다. 671B weights 저장에서 FP8 vs BF16의 memory savings를 계산하세요. 이것이 14.8T-token training budget과 어떻게 맞물리나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| MLA | "Multi-Head Latent Attention" | K와 V를 공유 low-rank latent(kv_lora_rank, 보통 512)로 압축하고 head별로 on-the-fly decompression함; KV cache는 latent만 저장 |
| kv_lora_rank | "MLA compression dim" | K와 V를 위한 공유 latent의 크기; DeepSeek-V3는 512 사용 |
| First k dense layers | "Early layers stay dense" | 안정성을 위해 첫 몇 MoE-model layer는 MoE router를 건너뛰고 dense MLP를 실행 |
| num_experts_per_tok | "Top-k routing" | token당 실행되는 routed expert 수; DeepSeek-V3는 8 사용 |
| Shared experts | "Always-on experts" | routing과 무관하게 모든 token을 처리하는 expert; DeepSeek-V3는 1 사용 |
| Auxiliary-loss-free routing | "Bias-adjusted load balance" | loss term을 추가하지 않고 expert load를 균형 있게 유지하도록 training 중 expert별 bias term을 조정 |
| MTP module | "Extra prediction head" | h^(1)과 E(t+1)에서 t+2를 예측하는 Transformer block; 더 조밀한 training, 무료 speculative-decoding draft |
| DualPipe | "Bidirectional pipeline" | forward/backward compute를 노드 간 all-to-all과 겹치는 training schedule |
| Active parameter ratio | "Sparsity" | active_params / total_params; DeepSeek-V3는 5.5% |
| FP8 training | "8-bit training" | storage와 많은 compute op를 FP8로 수행; 작은 quality cost로 BF16 대비 memory를 대략 절반으로 줄임 |

## 더 읽을거리

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — architecture, training, results 전체 문서
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — config files와 deployment notes
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) — MLA를 도입한 predecessor
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — V3 architecture 위의 reasoning-training successor
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089) — DeepSeek-family attention의 future direction
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe) — training-schedule reference
