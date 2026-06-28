# Speculative Decoding — draft하고 검증하고 반복하기

> Autoregressive decoding은 직렬이다. 각 토큰은 이전 토큰을 기다린다. Speculative decoding은 이 사슬을 끊는다. 싼 모델이 N개 토큰을 draft하고, 비싼 모델이 N개를 한 번의 forward pass로 verify한다. Draft가 맞으면 큰 forward 한 번으로 N번의 생성을 산 셈이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 minutes

## 문제

70B LLM이 토큰 하나를 샘플링하는 데 H100에서 약 30ms가 걸린다. 3B draft model은 약 3ms가 걸린다. 3B가 5토큰 앞을 draft하게 한 뒤 70B를 *한 번* 실행해 5개를 모두 verify하면, 최대 5개 accepted token에 대해 총 `5×3 + 30 = 45 ms`가 든다. 직선적인 생성의 `5×30 = 150 ms`와 비교된다. 이것이 speculative decoding의 전체 제안이다. 약간의 추가 GPU 메모리(draft model)를 내고 decode latency를 2-4배 낮춘다.

이 트릭은 분포를 보존해야 한다. Leviathan 등(2023)과 Chen 등이 동시에 소개한 speculative sampling은 출력 시퀀스가 큰 모델이 혼자 생성했을 때와 **동일한 분포**를 따른다는 것을 보장한다. 품질 tradeoff가 없다. 그저 더 빠르다.

2026년 추론에서는 네 종류의 draft-verifier pair가 지배적이다.

1. **Vanilla speculative (Leviathan 2023).** 별도 draft model(예: Llama 3 1B) + verifier(예: Llama 3 70B).
2. **Medusa (Cai 2024).** Verifier 위의 여러 decoding head가 위치 `t+1..t+k`를 병렬로 예측한다. 별도 draft model이 없다.
3. **EAGLE 계열 (Li 2024, 2025).** Verifier의 hidden state를 재사용하는 경량 draft. Vanilla보다 acceptance rate가 더 가깝다. 보통 3-4배.
4. **Lookahead decoding (Fu 2024).** Jacobi iteration. Draft model이 전혀 필요 없다. Self-speculation. 틈새지만 의존성이 없다.

2026년의 모든 production inference stack은 speculative decoding을 기본으로 제공한다. vLLM, TensorRT-LLM, SGLang, llama.cpp는 모두 적어도 vanilla + EAGLE-2를 지원한다.

## 개념

### 핵심 알고리즘

Verifier `M_q`와 더 싼 draft `M_p`가 주어졌다고 하자.

1. 이미 decoded된 prefix를 `x_1..x_k`라고 둔다.
2. **Draft**: `M_p`를 autoregressive하게 사용해 draft probabilities `p_1..p_N`과 함께 `d_{k+1}, d_{k+2}, ..., d_{k+N}`를 제안한다.
3. **병렬 verify**: `M_q`를 `x_1..x_k, d_{k+1}, ..., d_{k+N}`에 한 번 실행해 위치 `k+1..k+N+1`에 대한 verifier probabilities `q_1..q_{N+1}`를 얻는다.
4. **각 draft token을 왼쪽에서 오른쪽으로 accept/reject**: 각 `i`에 대해 확률 `min(1, q_i(d_i) / p_i(d_i))`로 accept한다.
5. 위치 `j`에서 첫 rejection이 발생하면 정규화된 "residual" distribution `(q_j - p_j)_+`에서 `t_j`를 샘플링한다. `j` 뒤의 모든 draft는 버린다.
6. `N`개를 모두 accept하면 `q_{N+1}`에서 추가 토큰 `t_{N+1}` 하나를 샘플링한다(공짜 보너스 토큰).

Residual distribution 트릭이 수학적 핵심이다. 이것이 출력을 정확히 `M_q`가 처음부터 샘플링한 것과 같은 분포로 유지한다.

### 속도를 결정하는 것

`α` = draft token당 기대 acceptance rate라고 하자. `c` = draft-to-verifier 비용 비율이라고 하자. Step마다:

- Naive generation은 토큰마다 큰 모델 호출 1회를 만든다.
- Speculative는 `α`가 높을 때 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 토큰마다 큰 모델 호출 1회를 만든다.

`α = 0.75`, `N = 5`에서의 일반적인 경험칙: 큰 모델 호출이 3배 줄어든다. Draft 비용은 싼 호출 5회다. 총 wall-clock은 약 2.5배 줄어든다.

**α가 의존하는 것:**

- Draft가 verifier를 얼마나 잘 근사하는가. 같은 계열 / 같은 훈련 데이터는 α를 크게 높인다.
- Decoding strategy. Greedy draft 대 greedy verifier는 α가 높다. Temperature sampling은 맞추기 더 어렵고 acceptance가 떨어진다.
- 작업 유형. Code와 structured output은 더 많이 accept된다(예측 가능). 자유 형식 창작 글쓰기는 덜 accept된다.

### Medusa — draft model 없는 draft

Medusa는 draft model을 verifier 위의 추가 output head로 교체한다. 위치 `t`에서:

```text
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

각 head는 자체 logits를 출력한다. 추론 시 각 head에서 샘플링해 candidate sequence를 얻고, 모든 candidate continuation을 한 번에 고려하는 tree-attention scheme으로 한 번의 forward pass에서 verify한다.

장점: 두 번째 모델이 없다. 단점: 학습 가능한 파라미터가 늘고, supervised fine-tuning stage(약 1B 토큰)가 필요하며, 좋은 draft가 있는 vanilla speculative보다 acceptance rate가 약간 낮다.

### EAGLE — hidden state 재사용으로 더 나은 draft

EAGLE-1/2/3(Li et al., 2024-2025)은 draft model을 verifier의 마지막 layer hidden state를 받아들이는 아주 작은 트랜스포머(보통 1 layer)로 만든다. Draft가 verifier의 feature representation을 보기 때문에, 그 예측은 verifier의 출력 분포와 강하게 상관된다. Acceptance rate는 ~0.6(vanilla)에서 0.85+로 오른다.

EAGLE-3(2025)은 candidate continuation 위에 tree search를 추가했다. vLLM과 SGLang은 Llama 3/4와 Qwen 3의 기본 spec pathway로 EAGLE-2/3를 제공한다.

### KV cache 춤

Verification은 verifier에 draft token `N`개를 한 번의 forward pass로 넣는다. 이때 verifier의 KV cache는 `N`개 entry만큼 확장된다. 일부 draft가 reject되면 accepted prefix 길이까지 cache를 rollback해야 한다.

Production 구현(vLLM의 `--speculative-model`, TensorRT-LLM의 LookaheadDecoder)은 scratch KV buffer로 이를 처리한다. 먼저 쓰고, acceptance 시 commit한다. 개념적으로 어렵지는 않지만 손이 많이 간다.

## 직접 만들기

`code/main.py`를 보라. 다음을 사용해 핵심 speculative-sampling 알고리즘(rejection step + residual distribution)을 구현한다.

- "Big model"은 손으로 작성한 분포 위의 deterministic-softmax다(acceptance 수학을 해석적으로 검증할 수 있게).
- "Draft model"은 big model의 perturbation이다.
- Acceptance / rejection loop는 direct sampling과 같은 marginal distribution을 만든다.

### 1단계: rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u`는 uniform random number다. `q_prob`은 drafted token에 대한 verifier 확률이다. `p_prob`은 draft model의 확률이다. Leviathan 정리는 이 Bernoulli 결정 뒤 rejection 시 residual에서 샘플링하면 verifier의 분포가 정확히 보존된다고 말한다.

### 2단계: residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

`q`에서 `p`를 element-wise로 빼고, 음수는 0으로 clamp한 뒤 다시 정규화한다. Rejection이 발생하면 여기서 샘플링한다.

### 3단계: speculative step 하나

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

다섯 개가 accept되면 하나가 보너스로 붙어, verifier pass 한 번에 토큰 여섯 개가 만들어진다.

### 4단계: acceptance rate 측정

Draft 품질 수준을 바꿔 가며 10,000번 speculative step을 실행한다. Acceptance rate 대 draft와 verifier distribution 사이의 KL divergence를 그린다. 깔끔한 단조 관계가 보여야 한다.

### 5단계: 분포 동등성 검증

경험적으로 speculative loop가 만든 토큰 histogram은 verifier에서 직접 샘플링해 만든 histogram과 맞아야 한다. 이것이 실제로 보는 Leviathan 정리다. Chi-square test가 sampling error 안에서 이를 확인한다.

## 활용하기

Production 예시:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

2026년 중반 기준 TensorRT-LLM은 가장 빠른 Medusa path를 갖고 있다. `faster-whisper`는 작은 draft를 이용해 Whisper-large에 speculative decoding을 감싼다.

**Draft 고르기:**

| 전략 | 고르는 시점 | Speedup |
|----------|--------------|---------|
| Vanilla draft (1B/3B Llama family) | 빠른 prototype, 훈련 없음 | 1.8-2.3× |
| Medusa heads | Verifier를 fine-tune할 수 있음 | 2-3× |
| EAGLE-2 / 3 | Production, 최대 속도 | 3-4× |
| Lookahead | Draft 없음, 훈련 없음, 추가 params 없음 | 1.3-1.6× |

**Spec-decode하지 말아야 할 때:**

- 1-5토큰짜리 단일 시퀀스 생성. Overhead가 지배한다.
- 매우 창의적이거나 high-temperature sampling(α가 떨어진다).
- 메모리 제약 배포(draft model이 VRAM을 추가로 쓴다).

## 출시하기

`outputs/skill-spec-decode-picker.md`를 보라. 이 스킬은 새 inference workload에 대해 speculative decoding 전략(vanilla / Medusa / EAGLE / lookahead)과 tuning parameter(N, draft temperature)를 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. 50,000토큰에서 speculative token distribution이 verifier의 direct-sample distribution과 chi-square p > 0.05 범위 안에서 일치하는지 확인하라.
2. **중간.** `α = 0.5, 0.7, 0.85`에 대해 `N`의 함수로 speedup(big-model forward당 토큰)을 그려라. 각 α에 대한 최적 `N`을 찾으라. (힌트: verify call당 기대 토큰 = `(1 - α^{N+1}) / (1 - α)`.)
3. **어려움.** 작은 Medusa를 구현하라. Lesson 14의 capstone GPT를 가져와 위치 t+2, t+3, t+4를 예측하는 추가 LM head 3개를 더한다. Joint multi-head loss로 tinyshakespeare에서 학습한다. 같은 모델을 truncate해 만든 vanilla draft 대비 acceptance rate를 비교하라.
4. **어려움.** Rollback을 구현하라. 10-token prefix KV cache로 시작해 draft token 5개를 넣고, 위치 3에서 rejection을 시뮬레이션하라. 다음 iteration에서 cache read가 "prefix + first 2 accepted drafts"와 정확히 일치하는지 확인하라.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Draft model | "싼 모델" | Candidate token을 제안하는 더 작은 모델. 보통 verifier보다 10-50배 싸다. |
| Verifier | "큰 모델" | 우리가 보존하려는 분포를 가진 target model. Speculative step마다 한 번 실행된다. |
| Acceptance rate (α) | "Draft가 맞는 빈도" | Verifier가 draft를 accept할 token당 확률. 보통 0.7-0.9. |
| Residual distribution | "Rejection fallback" | 정규화된 `(q - p)_+`. Rejection 시 여기서 샘플링해야 verifier 분포가 보존된다. |
| Bonus token | "공짜 토큰" | N개 draft가 모두 accept되면 verifier의 next-step distribution에서 하나 더 샘플링한다. |
| Medusa | "Draft 없는 speculative" | Verifier 위의 여러 LM head가 위치 t+1..t+k를 병렬로 예측한다. |
| EAGLE | "Hidden-state draft" | Verifier의 마지막 layer hidden state에 conditioned된 작은 transformer draft. |
| Lookahead decoding | "Jacobi iteration" | Fixed-point iteration을 사용하는 self-speculation. Draft model이 없다. |
| Tree attention | "여러 candidate를 한 번에 verify" | 여러 draft continuation을 동시에 고려하는 branching verification. |
| KV rollback | "Rejected draft 되돌리기" | Scratch KV buffer. Acceptance 시 commit하고 reject 시 버린다. |

## 더 읽을거리

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 핵심 알고리즘과 동등성 정리.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 동시 소개. 깔끔한 Bernoulli-rejection 증명.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 논문. Tree-attention verification.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1. Hidden-state-conditioned draft.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2. Dynamic tree depth.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, no-draft 접근.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 네 전략을 모두 연결한 표준 production reference.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 reference code.
