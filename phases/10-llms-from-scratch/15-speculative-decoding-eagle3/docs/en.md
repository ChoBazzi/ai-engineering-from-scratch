# Speculative Decoding과 EAGLE-3

> Phase 7 · Lesson 16은 수학을 증명했습니다. Leviathan rejection rule은 verifier의 distribution을 정확히 보존합니다. 이 lesson은 2026 production speculative decoding을 training-stack 관점에서 봅니다. EAGLE-3는 draft model을 값싼 approximation에서 verifier 자신의 hidden state로 훈련된 목적형 tiny network로 바꿨고, train distribution과 inference distribution을 맞추는 training-time test loop를 더했습니다. 결과: end-to-end 3×에서 6.5× speedup, chat에서 token당 accepted rate 0.9 이상, distributional tradeoff 없음. 2026년 모든 production inference stack은 이를 기본으로 ship합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 minutes

## 학습 목표

- Leviathan theorem을 한 문장으로 말하고 speculative loop가 verifier와 동일한 distribution의 sample을 생성함을 증명하기.
- Vanilla spec-decoding(Leviathan 2023)에서 EAGLE, EAGLE-2, EAGLE-3까지의 2년 progression을 훑고 각 단계가 제거한 정확한 limitation 이름 붙이기.
- Acceptance rate `α`와 draft-to-verifier cost ratio `c`로 expected speedup을 계산하고 각 regime에 최적인 draft length `N` 고르기.
- Full speculative loop를 처음부터 구현하기: draft, verify, residual에서 reject-sample, rejection 시 KV cache rollback, full acceptance 시 bonus token emit.

## 문제

70B model의 autoregressive decoding은 H100에서 초당 35 token 정도일 수 있습니다. GPU는 거의 포화되지 않습니다. 한계는 memory bandwidth입니다. Token마다 HBM에서 70B weights를 읽고, 한 단계 산술을 수행하고, float 하나를 생성합니다. Compute unit은 대부분 놀고 있습니다.

Speculative decoding은 이를 실제로 풀 수 있는 throughput 문제로 바꿉니다. 값싼 draft가 `N`개의 작은 forward pass로 `N` token을 제안합니다. Verifier는 prefix와 모든 `N` drafts를 한 번에 실행합니다. 위치 `i`에서 verifier의 distribution이 draft와 통계적 의미에서 동의하면 accept하고, 그렇지 않으면 reject한 뒤 residual distribution에서 correction을 sample합니다. 하나의 big-model forward가 하나가 아니라 최대 `N+1` accepted tokens를 생성합니다.

중요한 theorem은 Leviathan, Kalman, Matias(ICML 2023)입니다. Output distribution은 verifier에서 직접 sampling했을 때와 동일합니다. 근사가 아닙니다. 동일합니다. 이것이 speculative decoding이 production에서 허용되는 전부의 이유입니다. Quality tradeoff가 없는 순수 latency optimization입니다.

Phase 7 · Lesson 16이 준 것은 수학입니다. 이 lesson이 주는 것은 training stack입니다. 좋은 draft는 싼 draft보다 speedup을 2× 더 끌어올릴 수 있습니다. EAGLE, EAGLE-2, EAGLE-3(Li et al., 2024-2025)는 "draft = 같은 model의 더 작은 version"을 정밀한 engineering discipline으로 바꿨습니다. 2026 production inference server는 EAGLE-3를 기본값으로 둡니다.

## 개념

### Invariant: Leviathan rejection sampling

`p(t)`를 어떤 prefix가 주어졌을 때 draft의 next token distribution, `q(t)`를 verifier의 distribution이라고 합시다. Draft token `d ~ p`를 sample합니다. `min(1, q(d) / p(d))` 확률로 accept합니다. Reject하면 residual distribution `(q - p)_+ / ||(q - p)_+||_1`에서 sample합니다. 결과 sample은 `q`에 따라 분포합니다. 이는 `p`가 얼마나 나쁜지와 무관하게 참입니다. 나쁠수록 더 자주 reject할 뿐 output은 exact입니다.

이 호출 `N`개를 `prefix + d_1 + ... + d_N`에 대한 verifier forward pass 하나로 이어 붙입니다. Verifier는 `q_1, q_2, ..., q_{N+1}`을 동시에 반환합니다. 왼쪽에서 오른쪽으로 진행합니다. 위치 `j`에서 첫 rejection이 나오면 `residual(q_j, p_j)`에서 sample하고 멈춥니다. 모두 accept되면 `q_{N+1}`에서 bonus token 하나를 sample합니다.

### Speedup을 결정하는 요소

`α`를 drafted token당 expected acceptance rate라고 합시다. `c = cost(draft) / cost(verifier)`를 cost ratio라고 합시다. Verifier forward당 expected accepted tokens는 다음과 같습니다.

```text
E[accepted] = (1 - α^(N+1)) / (1 - α)
```

Accepted token당 expected total wall time은 `(N * c + 1) / E[accepted]`입니다. 이를 `N`에 대해 최소화하면 sweet spot을 얻습니다. `α = 0.8, c = 0.05`이면 optimal `N`은 대략 5-7이고 speedup은 3.2×입니다. `α = 0.95, c = 0.02`이면 optimal `N`은 8-10 정도이며 speedup은 5×에 가깝습니다.

가장 큰 lever는 `α`입니다. Fixed `N = 5`에서 `α = 0.6`(vanilla draft)에서 `α = 0.9`(EAGLE-3)로 가면 verifier forward당 expected accepted tokens가 2.2에서 4.1로 올라갑니다. 같은 verifier에서 거의 2× throughput입니다.

### 2년간의 발전

**Vanilla speculative (Leviathan, 2023).** Draft model은 같은 family에서 독립적으로 훈련된 더 작은 LLM입니다. 연결은 쉽지만 `α ≈ 0.6`, speedup은 좋아야 2× 정도입니다.

**EAGLE-1 (Li et al., 2024).** Draft는 verifier의 last-layer hidden state를 input으로 받아 next token을 직접 예측하는 tiny transformer, 보통 한두 layer입니다. Draft가 verifier의 feature representation을 보므로 distribution이 verifier와 훨씬 가깝습니다. `α`는 0.7-0.8로 올라갑니다.

**EAGLE-2 (Li et al., 2024).** Dynamic draft tree를 더합니다. `N` token의 단일 sequence를 제안하는 대신 작은 candidate tree를 제안하고, verifier로 한 번에 score(tree attention)한 뒤 highest-probability path를 따라갑니다. Draft length는 step별로 adaptive해집니다. Accepted-path token당 `α`는 0.85 이상으로 올라갑니다.

**EAGLE-3 (Li et al., 2025, NeurIPS).** 두 가지를 더 바꿉니다. 첫째, feature-prediction loss를 완전히 제거합니다. EAGLE-1/2는 draft가 verifier hidden state를 맞추도록 훈련했는데, 이는 데이터가 더 많아져도 이득을 제한합니다. EAGLE-3는 token prediction을 직접 훈련합니다. 둘째, training-time test(TTT): draft training 중에 inference와 같은 방식으로 draft 자신의 이전 prediction을 여러 step에 걸쳐 다시 input으로 넣습니다. 이것이 train/test distribution을 맞추고 error accumulation을 막습니다. 측정 speedup: chat에서 최대 6.5×, H100의 SGLang batch 64에서 38% throughput improvement.

### KV cache rollback

Verification은 verifier의 KV cache를 한 pass에서 `N` entry만큼 확장합니다. Rejection이 위치 `j`에서 일어나면 `j-1` 이후 cache content는 틀린 상태입니다. 흔한 구현은 두 가지입니다. Scratch buffer에 쓰고 accept 시 commit하거나(vLLM, TensorRT-LLM), physical KV cache와 logical length를 유지하고 reject 시 truncate합니다. 어느 쪽이든 rollback cost는 layer와 head별 byte 수이며 forward-pass cost에 비해 무시할 수 있습니다.

EAGLE-2 tree search에서 verifier는 tree topology를 존중하는 non-causal mask로 attention을 실행합니다. Engineering은 까다롭지만 계산은 custom mask를 가진 표준 flash-attention call입니다.

### 2026년의 draft architecture

| Strategy | Draft type | `α` | Speedup | Training cost |
|----------|-----------|-----|---------|---------------|
| Vanilla | Separate small LLM | 0.55-0.70 | 1.8-2.3× | None (reuse existing small model) |
| Medusa | Extra LM heads on verifier | 0.65-0.75 | 2-3× | ~1B SFT tokens |
| EAGLE-1 | 1-layer transformer on hidden states | 0.70-0.80 | 2.5-3× | ~60B tokens |
| EAGLE-2 | EAGLE-1 + dynamic draft tree | 0.80-0.88 | 3-4× | ~60B tokens |
| EAGLE-3 | Multi-layer feature fusion + TTT | 0.88-0.92 | 3.5-6.5× | ~60-200B tokens |
| Lookahead | No draft (Jacobi iteration) | N/A | 1.3-1.6× | None |

2026 production에서는 vLLM과 SGLang이 가능하면 EAGLE-3를 기본값으로, 아니면 EAGLE-2를 사용합니다. TensorRT-LLM은 Meta와 NVIDIA public model에 대해 가장 빠른 Medusa path를 갖습니다. llama.cpp는 CPU deployment용 vanilla draft를 제공합니다.

## 직접 만들기

`code/main.py`를 보세요. Draft-of-N, verifier parallel pass, per-position rejection, residual sampling, bonus token, KV rollback, output distribution이 `q`에서 직접 sampling한 것과 일치함을 보이는 empirical verification까지 포함한 full Leviathan speculative loop입니다.

### 1단계: rejection rule

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 2단계: residual distribution

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 3단계: 전체 speculative step

`spec_step` 함수는 `N` token을 `p`에서 draft한 뒤 하나의 parallel `q` evaluation으로 모두 verify합니다. 각 drafted token에 rejection rule을 적용하고, 첫 rejection에서는 residual에서 correction을 sample합니다. 모두 accept되면 `q_{N+1}`에서 bonus token을 emit합니다.

### 4단계: KV rollback bookkeeping

Simulator는 worker별 logical `kv_length`를 추적합니다. `k` drafts가 accept되면 `kv_length += k`입니다. 위치 `j`에서 rejection이 일어나면 cache는 이미 `j`를 지나 써져 있지만 logical length는 `prefix_length + j + 1`, 즉 correction token 바로 다음으로 설정됩니다. 이후 read는 logical length로 truncate됩니다.

### 5단계: Leviathan check

50,000 speculative step을 실행하세요. Accepted tokens의 empirical distribution을 세세요. `q`에서 직접 뽑은 50,000 sample과 비교하세요. Chi-square statistic은 critical value보다 충분히 낮아야 합니다. Theorem이 실제에서도 통과합니다.

### 6단계: speedup vs. α

`p`를 여러 amplitude로 `q`에서 멀어지게 perturb하여 draft quality를 sweep하세요. `α`를 측정한 뒤 `α`와 `N`의 함수로 verifier call당 expected tokens를 출력하세요. 코드는 EAGLE-3급 draft quality(`α ≈ 0.9`)가 verifier call당 4-5 token을 가능하게 함을 보여 주는 table을 출력합니다.

## 활용하기

Production-level `vllm serve` with EAGLE-3:

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

H100에서 batch 64로 EAGLE-3를 쓰는 SGLang은 EAGLE-3 paper 기준 batch-64 vanilla decoding보다 대략 1.38× 더 높은 throughput을 냅니다.

Speculative decoding을 꺼낼 때:

- p50 latency가 peak throughput보다 중요한 모든 interactive chat workload.
- Code generation과 structured output(JSON, SQL). Target distribution이 매우 예측 가능하므로 `α`가 0.9를 넘습니다.
- Long-form generation(수천 token). Amortized speedup이 계속 유효합니다.

꺼내지 않을 때:

- 매우 작은 model(< 3B). Draft가 verifier보다 충분히 싸지 않습니다.
- Tiny batch-1 CPU deployment. Draft model의 memory overhead가 그만한 가치가 없을 수 있습니다.
- `α`가 무너지는 very-high-temperature creative sampling.

## 산출물

이 lesson은 `outputs/skill-eagle3-tuner.md`를 생성합니다. Inference workload(model, batch size, target latency, task profile)가 주어지면 speculative-decoding strategy와 tuning parameters(draft family, `N`, tree depth, temperature-aware switching)를 추천합니다.

## 연습문제

1. `code/main.py`를 실행하세요. Leviathan distribution check의 chi-square statistic이 50,000 sample에서 95% critical value 아래에 머무는지 확인하세요.

2. `α`를 0.9, `c`를 0.04로 고정하고 `N`을 1부터 10까지 sweep하세요. Verifier call당 expected tokens와 token당 actual wall time을 plot하세요. Wall time을 최소화하는 `N`을 찾고 curve의 모양을 설명하세요.

3. 코드를 수정해 EAGLE-2 tree search를 시뮬레이션하세요. 각 step에서 draft가 `[2, 2, 2]` 모양의 tree(여덟 candidate path)를 제안합니다. Verifier는 한 번 실행되고 highest-probability accepted path가 이깁니다. Leaf당 `α`와 verifier call당 total tokens를 계산하세요. Equivalent compute의 linear-chain spec-decoding과 비교하세요.

4. 두 concurrent sequence에 대한 batched KV rollback simulator를 구현하세요. Sequence A는 모든 draft가 accept되고, sequence B는 position 2에서 reject됩니다. 올바른 `kv_length`가 sequence별로 갱신되고 낭비되는 work가 없음을 보이세요.

5. EAGLE-3 paper의 Section 4(Training-Time Test)를 읽으세요. TTT 없는 naive draft training이 왜 exposure bias를 겪는지, 그리고 training 중 draft 자신의 prediction을 먹이는 것이 왜 이를 고치는지 두 문장으로 설명하세요. 이를 seq2seq의 scheduled-sampling literature와 연결하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| Leviathan rule | "min(1, q over p)" | `min(1, q(d)/p(d))` 확률의 Bernoulli accept/reject입니다. Rejection 시 residual에서 sample하면 verifier distribution을 정확히 보존합니다 |
| Residual distribution | "(q minus p) plus, normalized" | `(q - p)_+`를 0에서 clamp한 뒤 renormalize한 것입니다. Rejection 시 sample해야 하는 올바른 distribution입니다 |
| Acceptance rate α | "draft가 맞는 빈도" | Rejection rule 아래의 token당 expected Bernoulli-success probability입니다. 모든 speedup math를 좌우합니다 |
| EAGLE-1 | "hidden-state draft" | Verifier의 last-layer hidden state에 conditioned된 tiny transformer draft(Li et al., 2024) |
| EAGLE-2 | "dynamic draft tree" | EAGLE-1에 candidate continuation tree를 더하고, verifier pass 하나에서 tree attention으로 score합니다 |
| EAGLE-3 | "training-time test" | Feature-prediction loss를 제거하고, training 중 draft가 자신의 output을 먹도록 하여 direct token prediction을 훈련합니다 |
| Training-time test (TTT) | "exposure bias fix" | Train/test input distribution이 맞도록 training 중 draft를 autoregressive하게 실행합니다. Scheduled sampling의 직접적 analog입니다 |
| KV rollback | "rejected drafts 되돌리기" | Rejection 뒤 verifier의 KV cache를 accepted-prefix length로 reset하는 bookkeeping입니다 |
| Bonus token | "공짜 token" | `N` drafts가 모두 accept되면 추가 verifier cost 없이 `q_{N+1}`에서 sample하는 token입니다 |
| Tree attention | "많은 candidate를 한 번에 verify" | Draft tree의 topology를 존중하는 non-causal mask가 있는 attention입니다. Tree의 모든 node에 대한 `q_i`를 forward pass 하나에서 계산합니다 |

## 더 읽을거리

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192) — foundational paper와 equivalence theorem
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318) — 깔끔한 proof를 가진 동시 독립 소개
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — EAGLE-1, hidden-state-conditioned draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — dynamic tree search
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840) — 2026 production default
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774) — alternative draft-free approach
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 모든 strategy가 연결된 canonical production reference
