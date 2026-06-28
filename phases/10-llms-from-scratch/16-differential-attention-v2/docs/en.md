# Differential Attention (V2)

> Softmax attention은 matching되지 않는 모든 token에 작은 probability를 퍼뜨립니다. 100k token에서는 그 noise가 쌓여 signal을 덮습니다. Differential Transformer(Ye et al., ICLR 2025)는 attention을 두 softmax의 차이로 계산해 shared noise floor를 빼는 방식으로 이를 고칩니다. DIFF V2(Microsoft, 2026년 1월)는 production-stack rewrite입니다. Decode latency를 baseline Transformer와 맞추고, custom kernel 없이 FlashAttention과 호환됩니다. 이 lesson은 V1부터 V2까지 end-to-end로 다루며, stdlib Python에서 실행할 수 있는 difference operation toy implementation을 제공합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 minutes

## 학습 목표

- Softmax attention에 noise floor가 왜 생기고 context length와 함께 왜 커지는지 정확히 말하기.
- Differential attention formula를 유도하고 subtraction이 shared noise component를 지우면서 signal을 보존하는 이유 설명하기.
- V1-to-V2 diff를 훑기: 무엇이 빨라졌고, 무엇이 단순해졌고, 무엇이 안정해졌으며, 각 변화가 production pre-training에 왜 필요했는지.
- Pure Python으로 differential attention을 처음부터 구현하고 synthetic signal-plus-noise query에서 noise-cancellation property를 empirical하게 검증하기.

## 문제

Standard softmax attention에는 scale에서 operational headache로 변하는 수학적 성질이 있습니다. Query `q`에 대해 attention weights는 `softmax(qK^T / sqrt(d))`입니다. Softmax는 정확한 zero를 만들 수 없습니다. Matching되지 않는 모든 token이 양의 mass를 조금씩 받습니다. 그 residual mass가 noise이고, context length와 함께 scale됩니다. 128k token에서 matching되지 않는 token 하나가 probability의 0.001%만 받아도, 127,999개가 합쳐지면 전체의 약 12%를 기여합니다. Model은 context와 함께 커지는 noise floor를 우회하는 법을 배워야 합니다.

경험적으로 이는 attention-head interference로 나타납니다. Long-context RAG의 hallucinated citations, 100k-token retrieval task의 lost-in-the-middle failure, 32k 이후 needle-in-haystack benchmark에서의 미묘한 accuracy degradation입니다. Differential Transformer paper(arXiv:2410.05258, ICLR 2025)는 gap을 측정했습니다. DIFF Transformer는 같은 크기 baseline보다 낮은 perplexity, 높은 long-context accuracy, 더 적은 hallucination을 보였습니다.

DIFF V1에는 frontier pre-training pipeline에 들어가지 못하게 한 세 문제가 있었습니다. Decode step마다 value cache를 두 번 load해야 했고, FlashAttention compatibility를 깨는 custom CUDA kernel이 필요했으며, per-head RMSNorm이 70B-plus scale의 long-run training을 불안정하게 만들었습니다. DIFF V2(Microsoft unilm blog, 2026년 1월 20일)는 세 가지를 모두 고쳤습니다. 이 lesson은 두 version을 훑고, difference operator를 만들며, toy query에서 noise cancellation을 benchmark합니다.

## 개념

### Softmax의 noise floor

Query `q`와 keys `K = [k_1, ..., k_N]`에 대해 attention weights는 다음과 같습니다.

```text
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

어떤 `w_i`도 zero가 아닙니다. `k_i`가 `q`와 완전히 무관해도 score `q . k_i`는 0이 아닙니다. `||q||^2 / d` variance를 가지고 zero 주변에서 흔들립니다. Softmax normalization 뒤에는 무관한 token도 여전히 weighted sum에 `O(1/N)`을 기여합니다. 무관한 token 전체의 기여는 `O((N-1)/N) = O(1)`입니다. 작은 양이 아닙니다.

Model이 원하는 것은 hard top-k에 가깝습니다. Matching token에는 높은 weight를, 나머지에는 거의 zero weight를 주는 것입니다. Softmax는 이를 직접 하기에는 너무 smooth합니다.

### Differential idea

각 head의 Q와 K projection을 둘로 나눕니다. Q = (Q_1, Q_2), K = (K_1, K_2). 두 attention map을 계산합니다.

```text
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

Output:

```text
DiffAttn = (A_1 - lambda * A_2) V
```

Subtraction은 두 map이 공유하는 noise distribution을 지웁니다. 두 map 모두 127k개의 무관한 token에 대략 uniform weight를 둔다면(random initialization에서 그럴 가능성이 큼), 그것들은 상쇄됩니다. Signal, 즉 실제 관련 token 몇 개에 집중된 peaked weight는 두 map에 같은 magnitude로 나타날 때만 상쇄되는데, model이 훈련되면 그렇게 되지 않습니다.

`lambda`는 head별 learnable scalar이며 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`으로 parameterize됩니다. 음수가 될 수 있습니다. `lambda_init` 기본값은 0.8 같은 작은 양수입니다.

### 이것이 head별 noise canceling과 맞는 이유

같은 목소리를 녹음하는 두 noisy microphone을 생각해 보세요. 둘 다 speaker와 correlated background noise를 잡습니다. 하나를 다른 하나에서 빼면 shared noise가 사라집니다. Voice는 두 signal의 phase 또는 amplitude가 충분히 달라 완전히 상쇄되지 않기 때문에 살아남습니다. Head별 `lambda`는 정확히 이 균형을 학습합니다.

### V1 vs V2: 차이

V1은 parameter count를 baseline Transformer와 같게 유지했습니다. Head당 query 두 개를 얻기 위해 head dimension을 절반으로 줄였습니다. 이는 head expressiveness를 깎았고, 더 아프게는 head당 value cache를 절반으로 만들었습니다. Decode는 step마다 value cache를 두 번 load해야 했습니다(softmax branch마다 한 번). 결과적으로 parameter count는 같아도 decode가 baseline보다 느렸습니다.

V2는 query head 수를 두 배로 늘리고 KV head는 그대로 둡니다(up-projection에서 parameter를 빌려옴). Head dimension은 baseline과 같습니다. Subtraction 뒤에는 extra dimension을 다시 baseline Transformer의 O_W projection에 맞게 down-project합니다. 세 가지가 동시에 일어납니다.

1. Decode speed가 baseline과 같습니다(KV cache를 한 번 load).
2. FlashAttention이 변경 없이 동작합니다(custom kernel 없음).
3. Decode의 arithmetic intensity가 올라갑니다(HBM에서 load한 byte당 compute가 더 많음).

V2는 V1이 subtraction을 안정화하기 위해 사용한 per-head RMSNorm도 제거합니다. 70B-class pre-training scale에서 그 RMSNorm은 late training을 불안정하게 만들었습니다. V2는 extra module 없이 training을 안정화하는 더 단순한 initialization scheme으로 대체합니다.

### 언제 사용할까

| Workload | 이점 |
|----------|---------|
| Long-context RAG (64k+) | 더 깨끗한 attention map, 더 적은 hallucinated citation |
| Needle-in-haystack benchmarks | 32k 이후 substantial accuracy lift |
| Multi-document QA | 더 적은 cross-document interference |
| Code completion at 8k | Marginal, architecture change 가치가 낮음 |
| Short chat (< 4k) | Baseline과 사실상 구분 불가 |

가치는 context length와 함께 커집니다. 4k token에서는 noise floor가 충분히 작아 standard attention도 괜찮습니다. 128k에서는 해가 됩니다.

### 다른 2026년 knob과의 조합

| Feature | DIFF V2와 호환되는가? |
|---------|------------------------|
| GQA | Yes (V2 increases Q heads, not KV heads) |
| MLA (DeepSeek) | Yes in principle, no published paper combining them |
| MoE | Yes (attention is independent of MLP block) |
| RoPE | Yes (unchanged) |
| YaRN / long-context scaling | Yes (exactly where DIFF helps most) |
| FlashAttention | Yes in V2 (was no in V1) |
| Speculative decoding | Yes (attention change is invisible to the spec-decode loop) |

```figure
differential-attention
```

## 직접 만들기

`code/main.py`는 pure Python으로 differential attention을 구현합니다. 알려진 signal-plus-noise 구조를 가진 toy query를 사용해 noise-cancellation ratio를 직접 측정할 수 있습니다.

### 1단계: standard softmax attention

Stdlib matrix ops: list of lists, manual matmul, max를 빼는 numerical-stability softmax.

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 2단계: Q와 K를 두 절반으로 나누기

V1 style: head dimension을 절반으로 줄입니다. V2 style: head dimension은 유지하고 head 수를 두 배로 늘립니다. Toy implementation은 pedagogical clarity를 위해 V1을 사용합니다. 수학은 동일하고 bookkeeping만 다릅니다.

### 3단계: 두 softmax branch와 subtraction

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

주의: output weights는 음수가 될 수 있습니다. 괜찮습니다. Value cache는 여전히 signed contribution을 처리합니다. 이후 V projection이 sign을 흡수합니다.

### 4단계: noise cancellation 측정

길이 1024의 synthetic sequence를 만드세요. Signal token을 알려진 위치에 두고 나머지는 noise로 채웁니다. (a) standard softmax attention의 signal position weight와 (b) differential attention weight를 계산하세요. 각각의 signal-to-noise ratio를 측정하세요. DIFF attention은 두 branch가 얼마나 다르게 훈련되었는지에 따라 3x-10x 높은 signal-to-noise ratio를 안정적으로 만듭니다.

### 5단계: V1 vs V2 parameter accounting

Config(hidden=4096, heads=32, d_head=128)가 주어지면 다음을 출력합니다.

- Baseline Transformer: Q, K, V 각각 `hidden * hidden`, MLP는 4 * hidden.
- DIFF V1: Q, K 각각 `hidden * hidden`, V는 `hidden * hidden`(unchanged), 내부적으로 head dim 절반. Per-head `lambda` parameters 추가(O(heads * d_head)).
- DIFF V2: Q size `2 * hidden * hidden`, K size `hidden * hidden`, V size `hidden * hidden`. Extra dim은 O_W 전에 다시 down-project됩니다. 같은 `lambda` parameters 추가.

Toy는 V2의 extra parameter cost(대략 attention block당 `hidden * hidden` extra)를 측정해 출력합니다.

## 활용하기

2026년 4월 기준 DIFF V2가 모든 production inference server에 ship된 것은 아니지만, vLLM과 SGLang에서 integration이 진행 중입니다. 한편 pattern은 다음에 나타납니다.

- Microsoft 내부 long-context production model.
- 256k-plus context를 목표로 하는 여러 open model training run의 research replication.
- Alternate layer에서 DIFF attention과 sliding-window attention을 결합한 hybrid architecture.

2026년에 이를 쓸 때:

- 64k-plus effective context를 목표로 새 model을 처음부터 훈련하는 경우. Differential attention을 처음부터 넣으세요. 나중에 재훈련하는 비용은 큽니다.
- Lost-in-the-middle failure가 eval을 지배하는 long-context model fine-tuning. Q projection에 대한 LoRA가 DIFF structure를 근사할 수 있습니다.

쓰지 않을 때:

- Stable long-context performance를 가진 pre-trained dense model을 serving하는 경우. Retraining cost는 기존 weights에서 거의 회수되지 않습니다.
- Context가 항상 16k 이하인 경우. Noise floor는 무시할 만합니다.

## 산출물

이 lesson은 `outputs/skill-diff-attention-integrator.md`를 생성합니다. Model architecture, target context length, hallucination profile, training budget이 주어지면 새 pre-training run 또는 LoRA fine-tune에 differential attention을 추가하는 integration plan을 만듭니다.

## 연습문제

1. `code/main.py`를 실행하세요. Synthetic query에서 differential attention의 signal-to-noise ratio가 standard softmax attention보다 높은지 검증하세요. Noise amplitude를 바꿔 standard attention이 unusable해지는 crossover point를 보이세요.

2. 7B-class model(hidden=4096, heads=32, d_head=128, 32 layers)에 대해 baseline에서 DIFF V1으로, baseline에서 DIFF V2로 갈 때 parameter-count delta를 계산하세요. 어떤 component가 parameter를 얻고 무엇이 그대로인지 보이세요.

3. DIFF V1 paper(arXiv:2410.05258)의 Section 3과 DIFF V2 Hugging Face blog의 Section 2를 읽으세요. V1 per-head RMSNorm이 왜 필요했고 V2가 왜 training divergence 없이 이를 제거할 수 있었는지 두 문장으로 설명하세요.

4. Ablation을 구현하세요. `lambda = 0`(pure first softmax)과 `lambda = 1`(full subtraction)으로 differential attention을 계산하세요. Synthetic query에서 sweep 전반의 signal-to-noise 변화를 측정하고 signal-to-noise를 최대화하는 `lambda`를 식별하세요.

5. Toy를 GQA + DIFF V2로 확장하세요. 8 KV heads와 32 Q heads를 선택하세요. KV cache size가 같은 (8, 32) configuration을 가진 baseline GQA model과 일치함을 보이세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| Differential attention | "Two softmaxes minus each other" | Q, K를 둘로 나누고 두 softmax map을 계산한 뒤 두 번째를 lambda로 scale해 첫 번째에서 빼고 V를 곱합니다 |
| Noise floor | "Softmax의 non-zero tail" | Softmax가 모든 unrelated token에 주는 O(1/N) weight입니다. Long context 전체에서는 O(1)로 합산됩니다 |
| lambda | "Subtraction scale" | `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`으로 parameterize된 per-head learnable scalar입니다. 음수가 될 수 있습니다 |
| DIFF V1 | "ICLR 2025 version" | Original Differential Transformer. Parameter count를 보존하려고 head dim을 절반으로 줄이며 custom kernel이 필요하고 decode가 느립니다 |
| DIFF V2 | "January 2026 fix" | KV head는 유지하고 Q head를 두 배로 늘립니다. Baseline decode speed와 맞고 FlashAttention과 작동합니다 |
| Per-head RMSNorm | "V1 stabilizer" | Difference 뒤에 V1이 적용한 extra norm입니다. V2는 late-training instability를 막기 위해 제거했습니다 |
| Signal-to-noise ratio | "attention이 얼마나 낭비되는가" | True signal position의 weight와 unrelated position 평균 weight의 비율입니다 |
| Lost in the middle | "Long-context failure mode" | 긴 context 중간의 document에 대한 retrieval accuracy가 떨어지는 empirical phenomenon입니다. DIFF attention이 이를 줄입니다 |
| Arithmetic intensity | "Load된 byte당 FLOPs" | V2가 decode에서 KV load당 query를 두 배로 늘려 증가시킨 비율입니다. Memory-bound decode에 중요합니다 |

## 더 읽을거리

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) — noise-cancellation theory와 long-context ablation을 담은 original paper
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) — production-stack rewrite, baseline decode matching, FlashAttention-compatible
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) — subtraction이 pretrained attention structure를 회복하는 이유에 대한 theoretical analysis
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) — parameter-sharing variant
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) — DIFF가 subtract하는 baseline Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) — DIFF attention이 겨냥하는 long-context benchmark
