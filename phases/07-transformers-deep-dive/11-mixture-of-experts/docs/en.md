# Mixture of Experts (MoE)

> dense 70B 트랜스포머는 모든 토큰마다 모든 파라미터를 활성화합니다. 671B MoE는 토큰마다 37B만 활성화하고 모든 벤치마크에서 이를 이깁니다. 희소성은 이번 10년에서 가장 중요한 스케일링 아이디어입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 문제

dense 트랜스포머의 추론 FLOPs는 파라미터 수와 같습니다(forward pass 때문에 2배를 곱합니다). dense 모델을 키우면 모든 토큰이 전체 비용을 냅니다. 2024년쯤 frontier는 compute wall에 부딪혔습니다. 의미 있게 더 똑똑해지려면 토큰당 FLOPs가 지수적으로 더 필요했습니다.

Mixture of Experts는 이 연결을 끊습니다. 각 FFN을 `E`개의 독립 expert + 토큰마다 `k`개의 expert를 고르는 router로 바꿉니다. 전체 파라미터 = `E × FFN_size`입니다. 토큰당 활성 파라미터 = `k × FFN_size`입니다. 2026년의 전형적 설정은 `E=256`, `k=8`입니다. 저장 공간은 `E`에 따라 늘고, compute는 `k`에 따라 늘어납니다.

2026년 frontier는 거의 전부 MoE입니다. DeepSeek-V3(총 671B / 활성 37B), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss가 여기에 속합니다. Artificial Analysis의 독립 leaderboard에서 상위 10개 오픈소스 모델은 모두 MoE입니다.

## 개념

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 교체

Dense transformer block은 다음과 같습니다.

```text
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block은 다음과 같습니다.

```text
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

각 expert는 독립적인 FFN입니다(일반적으로 SwiGLU). router는 단일 선형층입니다. 각 토큰은 자기만의 `k`개 expert를 고르고, 그 출력의 gated mixture를 받습니다.

### load-balancing 문제

router가 토큰의 90%를 expert 3으로 보내면 나머지 expert는 굶습니다. 세 가지 해결책이 시도되었습니다.

1. **Auxiliary load-balancing loss**(Switch Transformer, Mixtral). expert 사용량 분산에 비례하는 penalty를 추가합니다. 동작하지만 hyperparameter와 두 번째 gradient signal이 생깁니다.
2. **Expert capacity + token dropping**(초기 Switch). 각 expert가 최대 `C × N/E`개 토큰만 처리하고, 넘치는 토큰은 레이어를 건너뜁니다. 품질을 해칩니다.
3. **Auxiliary-loss-free balancing**(DeepSeek-V3). router의 top-k 선택을 이동시키는 expert별 학습 bias를 추가합니다. bias는 training loss 바깥에서 업데이트됩니다. 주목적에 penalty가 없습니다. 2024년의 큰 돌파구입니다.

DeepSeek-V3의 접근은 이렇습니다. 각 training step 뒤에 모든 expert에 대해 사용량이 목표보다 높은지 낮은지 확인합니다. bias를 `±γ`만큼 살짝 조정합니다. 선택에는 `scores + bias`를 사용합니다. gating에 쓰이는 expert 확률은 원래 `scores` 그대로입니다. routing과 expression을 분리합니다.

### Shared experts

DeepSeek-V2/V3는 expert를 *shared*와 *routed*로도 나눕니다. 모든 토큰은 모든 shared expert를 통과합니다. routed expert는 top-k로 선택됩니다. shared expert는 공통 지식을 포착하고, routed expert는 전문화됩니다. V3는 shared expert 1개와 256개의 routed expert 중 top-8을 사용합니다.

### Fine-grained experts

고전적 MoE(GShard, Switch)에서는 각 expert가 전체 FFN만큼 넓습니다. `E`는 작고(8–64), `k`도 작습니다(1–2).

현대 fine-grained MoE(DeepSeek-V3, Qwen-MoE)에서는 각 expert가 더 좁습니다(FFN 크기의 1/8). `E`는 크고(256+), `k`도 더 큽니다(8+). 전체 파라미터는 같지만 조합 수가 훨씬 빠르게 늘어납니다. `C(256, 8) = 400 trillion`개의 가능한 "expert" 조합이 토큰마다 존재합니다. 품질은 오르고 지연 시간은 평평하게 유지됩니다.

### 비용 특성

토큰당, 레이어당:

| 설정 | 토큰당 활성 파라미터 | 전체 파라미터 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3는 **토큰당 활성 FLOPs가 더 적은데도** 거의 모든 벤치마크에서 Llama 3 70B(dense)를 이깁니다. 파라미터가 많을수록 지식이 많습니다. 활성 FLOPs가 많을수록 토큰당 compute가 큽니다. MoE는 둘을 분리합니다.

### 문제는 memory

어떤 expert가 실행되는지와 무관하게 모든 expert는 GPU에 올라가 있어야 합니다. 671B 모델은 fp16 weights에 약 1.3 TB VRAM이 필요합니다. Frontier MoE 배포에는 expert parallelism이 필요합니다. expert를 여러 GPU에 shard하고, 토큰을 네트워크 너머로 route합니다. 지연 시간은 matmul이 아니라 all-to-all 통신이 지배합니다.

## 직접 만들기

`code/main.py`를 보세요. 순수 stdlib로 만든 compact MoE layer이며 다음을 포함합니다.

- `n_experts=8` SwiGLU-ish experts(설명을 위해 각각 하나의 선형층)
- top-k=2 routing
- softmax-normalized gating weights
- expert별 bias를 통한 auxiliary-loss-free balancing

### 1단계: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

bias는 선택에는 영향을 주지만 gate weight에는 영향을 주지 않습니다. 이것이 DeepSeek-V3의 트릭입니다. bias는 모델의 예측을 조종하지 않고 load imbalance를 바로잡습니다.

### 2단계: router에 토큰 100개 통과시키기

어떤 expert가 얼마나 자주 실행되는지 추적합니다. bias가 없으면 사용량이 치우칩니다. bias update loop(과도하게 사용된 expert에는 `-γ`, 덜 사용된 expert에는 `+γ`)를 쓰면 몇 번의 반복 후 사용량이 균등 분포로 수렴합니다.

### 3단계: 파라미터 수 비교

MoE 설정의 "dense equivalent"를 출력합니다. DeepSeek-V3 형태는 256 routed + 1 shared, 8 active, d_model=7168입니다. 전체 파라미터 수는 엄청납니다. 활성 파라미터 수는 dense Llama 3 70B의 1/7입니다.

## 사용하기

HuggingFace 로딩:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026년 프로덕션 추론에서는 vLLM이 MoE routing을 네이티브로 지원합니다. SGLang은 가장 빠른 expert-parallel path를 갖고 있습니다. 둘 다 top-k 선택과 expert parallelism을 자동으로 처리합니다.

**MoE를 고를 때:**
- 더 낮은 토큰당 추론 비용으로 frontier 품질을 원할 때.
- VRAM / expert-parallel 인프라가 있을 때.
- workload가 context-heavy(긴 문서)가 아니라 token-heavy(채팅, 코드)일 때.

**MoE를 고르지 말아야 할 때:**
- 엣지 배포입니다. 활성 FLOP이 얼마든 전체 저장 비용을 냅니다.
- 지연 시간에 민감한 단일 사용자 serving입니다. expert routing이 overhead를 추가합니다.
- 작은 모델(<7B)입니다. MoE의 품질 이점은 compute threshold(활성 파라미터 약 6B) 이상에서만 나타납니다.

## 실전 적용

`outputs/skill-moe-configurator.md`를 보세요. 이 스킬은 파라미터 예산, training tokens, deployment target이 주어졌을 때 새 MoE의 E, k, shared-expert layout을 고릅니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. auxiliary-loss-free bias update가 50번 반복되는 동안 expert 사용량을 어떻게 고르게 만드는지 보세요.
2. **보통.** 학습 router를 hash-based router(결정적, 학습 없음)로 바꾸세요. 품질과 balance를 비교하세요. 왜 학습 router가 더 나은가요?
3. **어려움.** GRPO 스타일의 "rollout-matched routing"(DeepSeek-V3.2 트릭)을 구현하세요. 추론 중 어떤 expert가 실행되는지 기록하고, gradient 계산 중 같은 routing을 강제합니다. 장난감 policy-gradient 설정에서 효과를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 보통 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Expert | "많은 FFN 중 하나" | 독립적인 feed-forward network입니다. FFN 계산의 희소 slice 전용 파라미터입니다. |
| Router | "gate" | 각 토큰을 각 expert에 대해 점수화하는 작은 선형층입니다. top-k 선택을 합니다. |
| Top-k routing | "토큰당 k개의 활성 expert" | 각 토큰의 FFN 계산은 정확히 k개의 expert를 gate로 가중해 통과합니다. |
| Auxiliary loss | "Load-balance penalty" | expert 사용량 치우침에 penalty를 주는 추가 loss term입니다. |
| Auxiliary-loss-free | "DeepSeek-V3의 트릭" | router 선택에만 작용하는 expert별 bias로 balance를 맞춥니다. 추가 gradient가 없습니다. |
| Shared expert | "항상 켜짐" | 모든 토큰이 통과하는 추가 expert입니다. 공통 지식을 포착합니다. |
| Expert parallelism | "expert 기준 shard" | 서로 다른 expert를 서로 다른 GPU에 분산합니다. 토큰은 네트워크를 가로질러 route됩니다. |
| Sparsity | "활성 params < 전체 params" | `k × expert_size / (E × expert_size)` 비율입니다. DeepSeek-V3는 37/671 ≈ 5.5%입니다. |

## 더 읽을거리

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 아이디어의 출발점입니다.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — 고전적 MoE인 Switch입니다.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B입니다.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP입니다.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — bias 기반 balancing 논문입니다.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 이 lesson의 router가 사용하는 fine-grained + shared-expert 분할입니다.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 원래 shared-expert 논문입니다.
