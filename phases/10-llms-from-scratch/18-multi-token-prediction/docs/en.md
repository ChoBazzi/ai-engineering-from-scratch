# Multi-Token Prediction (MTP)

> GPT-2부터 Llama 3까지 모든 autoregressive LLM은 position당 하나의 loss로 훈련합니다. Next token을 예측하는 것입니다. DeepSeek-V3는 position당 두 번째 loss를 추가했습니다. 그다음 token까지 예측합니다. 추가된 14B parameters(671B model 기준)는 gradient flow를 통해 main model로 distilled되었고, 훈련된 MTP heads는 inference에서 speculative-decoding drafter로 재사용되어 80%+ acceptance를 보였습니다. 1.8× generation throughput이 공짜로 따라왔습니다. 이 lesson은 DeepSeek technical report의 sequential MTP module을 만들고, loss와 shared-head parameter layout을 계산하며, MTP가 causal chain을 유지하는 반면 Gloeckle et al.의 original parallel MTP가 그것을 깨뜨린 이유를 설명합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 학습 목표

- MTP training objective를 말하고 prediction depth 전반의 joint loss 유도하기.
- Gloeckle et al.의 parallel MTP heads(2024)와 DeepSeek-V3의 sequential MTP modules의 차이를 설명하고 sequential design이 causal chain을 보존하는 이유 설명하기.
- Pre-training run에 MTP module을 추가할 때의 parameter 및 memory overhead 계산하기.
- Shared embedding, per-depth transformer block, projection, shared output head를 포함한 MTP module 하나를 처음부터 구현하기.

## 문제

Next-token prediction은 표준 LLM training objective입니다. 모든 hidden state는 정확히 하나, 즉 바로 다음 token을 예측하도록 supervise됩니다. 이는 놀라울 만큼 약한 signal입니다. Sequence의 정보 대부분은 한 token 너머로 이어집니다. Structure, coherence, factuality, arithmetic flow가 그렇습니다. Model은 trillion token에 걸쳐 많은 one-token signal을 누적하며 이를 배워야 합니다.

MTP는 묻습니다. 모든 hidden state가 여러 future token을 동시에 예측하도록 supervise된다면 어떨까요? Gloeckle et al.(Meta, 2024)은 이것이 도움이 됨을 보였습니다. 그들의 구현은 backbone 위에 여러 independent output head를 두고 각 head가 다른 offset을 예측하게 했습니다. Parallel하고 단순했지만, head들은 hierarchical refinement 없이 같은 hidden state를 보았습니다. Prediction이 causally chain되지 않았기 때문에 speculative decoding에 사용할 수도 없었습니다.

DeepSeek-V3(2024년 12월)는 MTP를 각 prediction depth에서 causal chain을 유지하는 sequential module로 재설계했습니다. Model은 `h_i^(0)`에서 `t+1`을 예측한 뒤, `h_i^(0)`과 `E(t+1)` embedding을 결합한 새 hidden state `h_i^(1)`에서 `t+2`를 예측합니다. 각 depth는 자체 small transformer block입니다. Shared embedding과 shared output head는 parameter overhead를 적당하게 유지합니다. DeepSeek-V3 scale에서는 671B main-model weights 위에 MTP module 전체로 14B extra parameters가 붙습니다. 이 2% overhead는 더 dense한 training signal과 inference에서 바로 쓸 수 있는 speculative-decoding draft를 함께 샀습니다.

이 lesson은 single MTP module과 D-depth loss를 처음부터 만듭니다. 수학은 단정합니다. 구현은 150줄입니다.

## 개념

### Sequential MTP recipe

DeepSeek-V3는 main model 위에 `D`개의 MTP module을 추가합니다. 각 module `k`(`k = 1..D`)는 depth `k`의 token, 즉 position `i`까지의 prefix가 주어졌을 때 `t_{i+k}`를 예측합니다.

Module `k`는 다음으로 구성됩니다.

- 자체 attention과 MLP를 가진 transformer block `T_k`.
- Previous-depth hidden state와 next-depth ground-truth token embedding을 결합하는 projection matrix `M_k`.
- Shared embedding `E`(main model과 동일).
- Shared output head `Out`(main model과 동일).

Training에서 position `i`까지의 prefix에 대해 per-depth hidden state는 다음과 같습니다.

```text
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Per-depth prediction:

```text
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Per-depth loss는 ground-truth `t_{i+k}`에 대한 cross-entropy입니다.

```text
L_k = CE(logits_{i+k}, t_{i+k})
```

Depth 전반의 joint loss:

```text
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda`는 작은 weighting factor입니다. DeepSeek-V3는 훈련 첫 10%에는 0.3, 이후에는 0.1을 사용합니다. Total training loss는 `L_main + L_MTP`입니다.

### 왜 parallel이 아니라 sequential인가

Gloeckle의 original parallel MTP는 D개의 output head를 `h_i^(0)`에 직접 적용했습니다. 각 head는 같은 backbone hidden state에서 `t_{i+k}`를 예측합니다. 훈련은 잘 되지만 prediction들이 서로 conditioned되지 않습니다. `head_1`의 output을 `head_2`에 도움으로 쓸 수 없습니다. Head들이 parallel하게 발화하기 때문입니다.

DeepSeek-V3의 sequential design은 `h_i^(k-1)`와 실제 next-token embedding `E(t_{i+k})`에서 `h_i^(k)`를 만듭니다. 이것이 causal chain을 보존합니다. `t_{i+k+1}`를 예측하려면 depth `k+1`의 module이 `t_{i+k}`에 있던 것을 봅니다. 이는 autoregressive decoder가 자신의 output을 소비하는 방식과 구조적으로 동일합니다. 그래서 MTP module을 speculative-decoding drafter로 직접 사용할 수 있습니다.

Inference에서는 `h_i^(k-1)`와 drafted `t_{i+k}`를 module `k+1`에 넣어 `t_{i+k+1}` prediction을 얻습니다. 반복하면 됩니다. 이는 trained MTP module을 draft network로 사용하는 EAGLE-style draft와 정확히 같습니다. DeepSeek-V3는 첫 MTP module에서 80%+ acceptance와 약 1.8× speedup을 보고합니다.

### Parameter accounting

Hidden `h`와 vocabulary `V`를 가진 model에 대해:

- Main model: billions of parameters, plus size `V * h`의 output head 하나.
- Shared output head: main model의 head를 재사용합니다. Extra params 없음.
- Shared embedding: main model의 embedding을 재사용합니다. Extra params 없음.
- Per-MTP module:
  - Projection `M_k`: `(2h) * h = 2h^2`.
  - Transformer block `T_k`: attention(MHA에서 `4h^2`) plus MLP(보통 ratio 8/3의 SwiGLU로 `8h^2`). Block당 약 `12h^2`.

Module당 total extra: `~14h^2`. DeepSeek-V3의 `h = 7168`, D = 1 module이면 paper 계산으로 `~14 * 7168^2 = ~720M` parameters입니다. DeepSeek-V3가 보고한 값은 14B입니다. 차이는 대부분 MTP module 안의 expert layer도 MoE이기 때문입니다.

### Speculative decoding payoff

Pre-training 중 MTP module은 훈련을 약 10% 늦춥니다(더 많은 forward compute, extra loss). 보상은 두 가지입니다.

1. 더 dense한 training signal. 각 hidden state는 D+1개의 supervision target을 봅니다. DeepSeek-V3의 ablation에서 MMLU, GSM8K, MATH, HumanEval에 일관된 몇 percentage point 개선이 측정되었습니다.

2. Inference에서 무료 speculative decoding draft. MTP module은 이미 다음 몇 token을 예측하도록 훈련되어 있습니다. Draft network로 재사용하면 80%+ acceptance rate를 냅니다. 그 수준에서는 N=3 또는 N=5 spec decoding이 1.8× throughput을 줍니다. 10% training-time cost는 inference를 처음 실행할 때 회수됩니다.

### EAGLE와의 관계

EAGLE은 pre-training 이후 작은 draft model을 별도로 훈련합니다. MTP는 draft를 pre-training에 bake합니다. 두 접근은 비슷한 accept rate로 수렴하지만 pipeline이 다릅니다.

| 차원 | EAGLE-3 | MTP(DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## 직접 만들기

`code/main.py`는 shared embedding, projection, transformer block, shared output head를 포함한 single MTP module을 end to end로 만듭니다. 이어서 짧은 synthetic sequence에서 per-depth cross-entropy loss를 계산하고 component별 parameter count를 출력합니다. 32 token의 toy vocabulary가 숫자를 읽기 쉽게 유지합니다.

### 1단계: shared embedding table

Single `vocab_size x hidden` table이 main model과 모든 depth의 모든 MTP module에 사용됩니다. 두 번째 copy가 아닙니다. 말 그대로 같은 tensor입니다.

### 2단계: depth별 combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

실제 DeepSeek-V3는 두 RMSNormed vector를 `[2h]`로 concat하고 `h x 2h` matrix로 projection합니다. Toy는 stdlib brevity를 위해 vector addition을 사용합니다.

### 3단계: depth k의 transformer block

Self-attention plus MLP입니다. Toy에서는 one-layer linear attention block과 SwiGLU MLP로 numpy 없이 구조를 보이게 유지합니다.

### 4단계: shared output head

Main model의 output projection을 재사용합니다. Vocabulary에 대한 logits입니다.

### 5단계: depth별 loss

Offset `k`의 ground-truth token에 대한 softmax(logits)의 cross-entropy입니다. `lambda / D` scaling factor로 depth 전반을 aggregate합니다.

### 6단계: parameter accounting

Total parameter count, shared(embedding, head) count, per-module extra count를 출력합니다. Main-model size 대비 MTP extra ratio를 보이세요.

## 활용하기

MTP는 DeepSeek-V3(2024년 12월)와 DeepSeek-R1 series에 통합되어 있습니다. Inference에서는:

- DeepSeek 자체 serving stack이 MTP module을 speculative decoder로 바로 사용합니다.
- vLLM과 SGLang은 2026년 4월 기준 DeepSeek-V3 MTP integration path를 갖고 있습니다.
- AMD의 ROCm SGLang tutorial은 V3 checkpoint에서 측정된 1.8× speedup과 함께 구체적 MTP speculative-decoding config를 보여 줍니다.

새 pre-training run에서 MTP를 사용할 때:

- Full pre-training pipeline을 제어하고 denser training signal을 확보하고 싶은 경우.
- Model을 scale에서 serving할 것을 알고 speculative decoding을 무료로 얻고 싶은 경우.
- Hidden size가 적어도 4096인 경우. 1B-scale에서는 overhead가 gain보다 더 아픕니다.

쓰지 않을 때:

- 기존 pre-trained dense model을 fine-tuning하는 경우. MTP module이 훈련되어 있지 않습니다.
- 비교를 위한 clean baseline이 필요한 research model. MTP는 architecture를 바꿉니다.

## 산출물

이 lesson은 `outputs/skill-mtp-planner.md`를 생성합니다. Pre-training run specification(model size, data, compute)이 주어지면 MTP integration plan을 반환합니다. 여기에는 depth 수 D, `lambda` schedule, memory overhead, inference-time speculative-decoding wiring이 포함됩니다.

## 연습문제

1. `code/main.py`를 실행하세요. Synthetic signal이 강해질수록 per-depth loss가 단조롭게 감소하는지 보이세요. Synthetic을 fixed pattern으로 바꾸고 depth-1과 depth-2 loss가 모두 수렴하는지 검증하세요.

2. D=1 MTP module을 가진 dense 70B model(hidden 8192, 80 layers)의 parameter overhead를 계산하세요. DeepSeek-V3가 보고한 14B overhead와 비교하세요. DeepSeek의 숫자가 더 큰 이유를 설명하세요. MTP transformer block이 같은 MoE structure를 상속해 per-module parameter count가 커지기 때문입니다.

3. Toy에서 D=2를 구현하세요. h^(1)을 받아 `t_{i+2}`를 예측하는 두 번째 MTP module을 추가하세요. Joint loss와 parameter accounting이 DeepSeek paper의 equations 19-21과 일치하는지 검증하세요.

4. Toy를 parallel MTP(Gloeckle-style)로 바꾸세요. Main hidden state 위에 D개의 output head를 추가하고 각 head가 다른 offset을 예측하게 하세요. 같은 synthetic signal에서 sequential version과 depth별 loss를 비교하세요. Sequential version은 intermediate prediction에 conditioned되므로 k > 1에서 더 낮은 depth-k loss를 내야 합니다.

5. Trained MTP module을 EAGLE-style draft로 사용하세요. Inference에서 module k를 호출해 `t_{i+k}`를 제안하세요. Held-out sequence에서 main model prediction 대비 draft token의 acceptance rate를 측정하세요. Toy에서 50%+에 도달하면 MTP-as-draft property를 empirical하게 재현한 것입니다.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| MTP module | "Extra loss block" | Main model보다 `k` position 앞의 token을 예측하는 small transformer block plus projection입니다 |
| Prediction depth | "어떤 offset인가" | Module `k`가 position `i`까지의 prefix에서 `t_{i+k}`를 예측한다는 integer `k`입니다 |
| Parallel MTP | "Gloeckle-style" | 같은 backbone hidden state 위의 D independent heads입니다. Conditional chain이 없습니다 |
| Sequential MTP | "DeepSeek-V3 style" | 각 module이 previous depth hidden state와 next token embedding에 conditioned됩니다. Causal chain을 보존합니다 |
| Shared output head | "Main head 재사용" | MTP module은 별도 output projection이 아니라 main model의 LM head를 호출합니다 |
| Shared embedding | "Main table 재사용" | 같은 vocabulary embedding table을 모든 곳에서 사용합니다. Duplicate parameter가 없습니다 |
| Projection matrix M_k | "hidden + next-token 결합" | Previous hidden state와 target-token embedding을 next depth input으로 접는 `h x 2h` linear layer입니다 |
| Joint loss L_MTP | "평균 extra losses" | Per-depth cross-entropy loss의 arithmetic mean에 `lambda`를 곱한 것입니다 |
| Acceptance rate at depth 1 | "MTP draft가 맞는 빈도" | D=1 MTP module의 top-1 prediction이 main model의 top-1 prediction과 같은 비율입니다. DeepSeek-V3에서 80%+입니다 |
| Lambda weighting | "Extra-loss 중요도" | Per-depth scaling factor입니다. DeepSeek-V3에서는 훈련 초기에 0.3, 이후 0.1입니다 |

## 더 읽을거리

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — joint-loss equation과 inference 1.8× speedup을 포함한 full sequential MTP description(Section 2.2)
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — DeepSeek design이 개선한 parallel MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B total(671B main + 14B MTP), deployment notes
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — MTP가 들어맞는 speculative-decoding framework
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — MTP가 경쟁하는 counterpart인 EAGLE의 2025 draft architecture
