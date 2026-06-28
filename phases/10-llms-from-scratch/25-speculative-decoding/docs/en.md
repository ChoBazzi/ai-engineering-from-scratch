# Speculative Decoding과 EAGLE

> frontier LLM이 token 하나를 생성하려면 수십억 parameter에 대한 full forward pass가 필요합니다. 그 forward pass는 대부분의 시간에 과도하게 큰 장비입니다. 훨씬 작은 모델이 다음 3-5 token을 맞힐 수 있고, 큰 모델은 그 추측을 *verify*하기만 하면 되는 경우가 많습니다. 추측이 맞으면 한 번의 비용으로 5 token을 얻습니다. Speculative decoding(Leviathan et al. 2023)은 이를 정확하게 만들었고, EAGLE-3(2025)은 acceptance rate를 verify당 ~4.5 token까지 밀어 올렸습니다. matched output distribution에서 4-5x speedup입니다.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## 문제

H100에서 70B-class model의 decode throughput은 보통 40-80 tokens/second입니다. 각 token은 HBM에서 모든 model weight를 읽는 full forward pass를 요구합니다. output을 바꾸지 않고 model을 작게 만들 수 없습니다. memory를 넘지 않고 batch size를 더 늘릴 수도 없습니다. 갇힌 셈입니다. 단, model이 forward pass 하나당 token을 하나보다 많이 출력하게 만들 수 있다면 예외입니다.

Autoregressive generation은 본질적으로 serial처럼 보입니다. `x_{t+1} = sample(p(· | x_{1:t}))`. 하지만 concurrency 기회가 있습니다. "다음 4 token은 아마 [a, b, c, d]일 것"이라고 말하는 cheap predictor가 있다면, **big model의 single forward pass**에서 5개 position을 모두 verify하고 가장 긴 matching prefix를 accept할 수 있습니다.

Leviathan, Kalai, Matias(2023, "Fast Inference from Transformers via Speculative Decoding")는 target model의 sampling distribution을 보존하는 영리한 accept/reject rule로 이를 exact하게 만들었습니다. 같은 output distribution, 2-4× 더 빠름.

## 개념

### 두 model setup

- **Target model** `M_p`: 실제로 sample하고 싶은 크고 느리며 품질 좋은 model. Distribution: `p(x)`.
- **Draft model** `M_q`: 작고 빠르며 품질이 낮은 model. Distribution: `q(x)`. 5-30× 작습니다.

step마다:

1. Draft model이 `K` token을 autoregressively 제안합니다: `x_1, x_2, ..., x_K ~ q`.
2. Target model이 모든 `K+1` position에 대해 ONE forward pass를 병렬로 실행하여 각 proposed token의 `p(x_k)`를 만듭니다.
3. 아래 modified rejection-sampling rule로 token을 왼쪽에서 오른쪽으로 accept/reject합니다. 가장 긴 matching prefix를 accept합니다.
4. token 하나라도 reject되면 corrected distribution에서 replacement를 sample하고 멈춥니다. 그렇지 않으면 `p(· | x_1...x_K)`에서 bonus token 하나를 sample합니다.

draft가 target과 완벽히 맞으면 target-forward 하나당 K+1 token을 얻습니다. draft가 position 1에서 틀리면 token 하나만 얻습니다.

### Exactness rule

Speculative decoding은 **sampling from p와 distribution상 동등함이 증명되어 있습니다**. rejection rule:

```text
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

여기서 `(p - q)+`는 pointwise difference의 positive part입니다. draft와 target이 동의할 때(`p ≈ q`) acceptance는 거의 1입니다. 동의하지 않을 때는 전체 sample이 여전히 정확히 `p`가 되도록 residual distribution을 구성합니다.

**Greedy case.** temperature=0 sampling에서는 `argmax(p) == x_t`인지 확인하기만 하면 됩니다. 맞으면 accept하고, 아니면 `argmax(p)`를 output하고 멈춥니다.

### Expected speedup

draft model의 token-level acceptance rate가 `α`이면, target-forward pass당 생성되는 expected token 수는 다음과 같습니다:

```text
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

`α = 0.8, K = 4`이면 `(1 - 0.8^5)/(1 - 0.8) = 3.36` token per forward입니다. target forward 하나의 비용은 대략 `cost_q * K + cost_p`(K draft steps + one target verify)입니다. `cost_p >> cost_q * K`이면 throughput speedup ratio는 `3.36× / 1 = 3.36×`입니다.

진짜 parameter는 `α` 하나뿐이고, 이는 전적으로 draft-target alignment에 달려 있습니다. 좋은 draft가 전부입니다.

### Draft 학습: Distillation

random small model은 poor draft입니다. 표준 recipe는 target에서 distill하는 것입니다:

1. 작은 architecture를 고릅니다(70B target에는 ~1B, 7B target에는 ~500M).
2. 큰 text corpus에 target model을 실행하고 next-token distribution을 저장합니다.
3. ground-truth token이 아니라 target distribution에 대한 KL divergence로 draft를 학습합니다.

결과: `α`는 보통 coding에서 0.6-0.8, natural-language chat에서 0.7-0.85입니다. production speedup은 2-3×입니다.

### EAGLE: Tree Drafting + Feature Reuse

Li, Wei, Zhang, Zhang(2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty")은 표준 speculative decoding의 두 inefficiency를 관찰했습니다:

1. draft는 K serial step을 수행하고, 각 step은 full-stack입니다. 하지만 draft는 가장 최근 verify에서 나온 target의 feature(hidden state)를 재사용할 수 있습니다. target은 draft가 처음부터 다시 유도하고 있는 rich representation을 이미 계산했습니다.
2. draft는 linear chain을 output합니다. draft가 candidate의 *tree*를 output할 수 있다면(각 node에 여러 guess), target의 single forward pass는 tree attention mask로 여러 candidate path를 병렬 verify하고 가장 긴 accepted branch를 고를 수 있습니다.

EAGLE-1 changes:
- Draft input = raw token이 아니라 position t의 target final hidden state.
- Draft architecture = 별도 small model이 아니라 1 transformer decoder layer.
- Output = depth당 K = 4-8 candidates, depth 4-6의 tree.

EAGLE-2(2024)는 dynamic tree topology를 추가합니다. draft가 uncertain한 곳에서는 tree가 더 넓어지고 confident한 곳에서는 좁게 유지됩니다. verify cost를 늘리지 않고 `α_effective`를 높입니다.

EAGLE-3(Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test")은 fixed top-layer feature dependency를 제거하고 새로운 "test-time simulation" loss로 draft를 학습합니다. draft는 teacher-forced training distribution이 아니라 target의 test-time distribution과 맞는 output으로 학습됩니다. Acceptance rate는 0.75(EAGLE-2)에서 0.82(EAGLE-3)로 오르고, mean tokens/verify는 3.0에서 4.5로 오릅니다.

### Tree Attention verification

draft가 tree를 output하면 target model은 **tree attention mask**를 사용해 single forward pass로 verify합니다. 이는 순수 line이 아니라 tree topology를 encode하는 causal mask입니다. 각 token은 tree에서 자기 ancestor에만 attend합니다. verify pass는 여전히 one forward, one matmul이고, topological mask는 KV entry 몇 개만 더 비용으로 냅니다.

```text
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

`a, b`가 경쟁 first-token candidate이고 `c, d, e, f`가 second-token candidate이면, 여섯 position 모두가 한 forward pass에서 verify됩니다. output은 accepted path 중 가장 긴 prefix입니다.

### 유리한 경우와 그렇지 않은 경우

**Wins:**
- predictable text(code, common English, structured output)를 가진 chat / completion. `α`가 높습니다.
- decode 중 GPU compute가 남는 setting(memory-bound phase). Tree drafting이 사용 가능한 FLOPs를 씁니다.

**Loses / no win:**
- 매우 stochastic한 output(high temperature creative writing). `α`가 `1/|vocab|` 쪽으로 떨어집니다.
- concurrency가 매우 높은 batch serving. batching이 이미 FLOPs를 채워 tree verification을 위한 여유가 거의 없습니다.
- draft가 target보다 훨씬 작지 않은 very small target model.

production shop은 보통 chat에서 2-3× wall-clock speedup, code generation에서 3-5×, creative writing에서는 거의 0에 가까운 이득을 보고합니다.

```figure
speculative-decoding
```

## 직접 만들기

`code/main.py`:

- exact rejection rule을 구현하고 target distribution을 보존함을 검증하는 reference `speculative_decode(target, draft, prompt, K, temperature)`(empirical KL < 0.01 vs plain target sampling).
- top-p branching으로 depth-K tree를 만드는 EAGLE-style tree drafter.
- verifier에 맞는 causal pattern을 만드는 tree attention mask builder.
- tiny LM에서 둘 다 실행하는 acceptance-rate harness(GPT-2-medium target에서 GPT-2-small 하나를 distill).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 활용하기

- **vLLM**과 **SGLang**은 first-class speculative decoding을 제공합니다. Flags: `--speculative_model`, `--num_speculative_tokens`. EAGLE-2/3 support는 `--spec_decoding_algorithm eagle` flag를 통해 사용합니다.
- **NVIDIA TensorRT-LLM**은 Medusa와 EAGLE tree를 native로 지원합니다.
- **Reference draft models**: `Qwen/Qwen3-0.6B-spec`(Qwen3-32B용 draft), `meta-llama/Llama-3.2-1B-Instruct-spec`(70B용 draft).
- **Medusa heads**(Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): draft model 대신 target 자체에 K개의 parallel prediction head를 추가합니다. 배포가 더 단순하지만 acceptance는 EAGLE보다 약간 낮습니다.

## 산출물

이 lesson은 `outputs/skill-speculative-tuning.md`를 산출합니다. target model의 workload를 profile하고 draft model, K(draft length), tree width, temperature, plain decode로 fallback할 시점을 선택하는 skill입니다.

## 연습문제

1. exact rejection rule을 구현하고 empirically verify하세요. `speculative_decode`와 plain target sampling으로 10K sample을 실행하고 두 output distribution 사이의 TV distance를 계산하세요. < 0.01이어야 합니다.

2. speedup formula를 계산하세요. 고정된 `α`와 `K`가 주어졌을 때 target-forward당 expected token을 plot하세요. α ∈ {0.5, 0.7, 0.9}에 대해 optimal K를 찾으세요.

3. tiny draft를 학습하세요. 124M GPT-2 target을 가져와 100M token에서 KL loss로 30M GPT-2 draft를 distill하세요. held-out text에서 `α`를 측정하세요. 기대값: 0.6-0.7.

4. EAGLE-style tree drafting을 구현하세요. chain 대신 draft가 각 depth에서 top-3 branch를 output하게 하세요. tree attention mask를 만드세요. target이 가장 긴 correct branch를 accept하는지 검증하세요.

5. failure mode를 측정하세요. temperature=1.5(high stochasticity)에서 speculative decode를 실행하세요. α가 collapse하고 draft overhead 때문에 algorithm이 plain decode보다 느려짐을 보이세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|-----------------|------------------------|
| Target model | "The big model" | sample하고 싶은 느리고 품질 좋은 model(p distribution) |
| Draft model | "The speculator" | 작고 빠른 predictor(q distribution); 5-30x 작음 |
| K / draft length | "Look-ahead" | verify pass당 speculated token 수 |
| α / acceptance rate | "Hit rate" | draft proposal이 accept될 token당 확률 |
| Exact rejection rule | "The accept test" | target distribution을 보존하는 r < p/q 비교 |
| Residual distribution | "Corrected p-q" | rejection 시 sample할 distribution인 (p - q)+ / ||(p - q)+||_1 |
| Tree drafting | "Branching speculation" | draft가 candidate tree를 output하고 tree-structured attention mask로 한 pass에 verify |
| Tree attention mask | "Topological mask" | 각 node가 ancestor에만 attend하도록 tree topology를 encode하는 causal mask |
| Medusa heads | "Parallel heads" | target 자체의 K extra prediction heads; 별도 draft model 없음 |
| EAGLE feature reuse | "Hidden-state draft" | draft input이 raw token이 아니라 target의 last hidden state라서 draft가 작아짐 |
| Test-time simulation loss | "EAGLE-3 training" | teacher forcing이 아니라 target의 test-time distribution과 맞는 output으로 draft를 학습 |

## 더 읽을거리

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — exact rejection rule과 theoretical speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — DeepMind의 concurrent speculative-sampling paper
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — draft model의 대안인 parallel-heads
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — feature reuse와 tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — train-time test-time matching
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — speculator-free alternative인 Jacobi/lookahead decoding
