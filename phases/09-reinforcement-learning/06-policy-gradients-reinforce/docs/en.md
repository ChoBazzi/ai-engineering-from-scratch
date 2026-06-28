# Policy Gradient — REINFORCE를 처음부터 만들기

> Value 추정을 멈춰라. policy를 직접 parameterize하고, expected return의 gradient를 계산한 뒤, 오르막 방향으로 한 걸음 이동하라. Williams (1992)는 이를 하나의 정리로 썼다. 이것이 PPO, GRPO, 모든 LLM RL loop가 존재하는 이유다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## 문제

Q-learning과 DQN은 *value* function을 parameterize한다. 행동은 `argmax Q`로 고른다. 이는 discrete action과 discrete state에서는 괜찮다. 하지만 action이 continuous일 때(10차원 torque 위에서 어떤 `argmax`를 할 것인가?) 또는 stochastic policy를 원할 때(`argmax`는 구조적으로 deterministic이다) 깨진다.

Policy gradient는 대신 *policy* 자체를 parameterize한다. `π_θ(a | s)`는 action에 대한 분포를 출력하는 신경망이다. 이 분포에서 샘플링해 행동한다. `θ`에 대한 expected return의 gradient를 계산한다. 오르막으로 이동한다. `argmax`가 없다. Bellman recursion도 없다. `J(θ) = E_{π_θ}[G]`에 대한 gradient ascent만 있다.

REINFORCE theorem(Williams 1992)은 이 gradient를 계산할 수 있다고 말한다. `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. episode를 실행한다. return을 계산한다. 모든 step에서 `∇ log π_θ(a | s)`에 곱한다. 평균낸다. Gradient ascent. 끝이다.

2026년의 모든 LLM-RL 알고리즘, 즉 PPO, DPO, GRPO는 REINFORCE를 다듬은 것이다. 이를 손에 익히는 것은 이 phase의 나머지와 Phase 10 · 07(RLHF implementation), Phase 10 · 08(DPO)의 prerequisite다.

## 개념

![Policy gradient: softmax policy, log-π gradient, return-weighted update 과정](../assets/policy-gradient.svg)

**Policy gradient theorem.** `θ`로 parameterize된 임의의 policy `π_θ`에 대해:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

여기서 `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}`는 step `t`부터의 discounted return이다. Expectation은 `π_θ`에서 샘플링한 전체 trajectory `τ`에 대한 것이다.

**증명은 짧다.** Expectation 아래에서 `J(θ) = Σ_τ P(τ; θ) G(τ)`를 미분한다. `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`(log-derivative trick)를 사용한다. `log P(τ; θ) = Σ log π_θ(a_t | s_t) + θ에 의존하지 않는 environment terms`로 분해한다. Environment terms는 사라진다. 두 줄짜리 대수로 theorem이 나온다.

**Variance reduction 요령.** Vanilla REINFORCE는 variance가 살인적으로 크다. return은 noisy하고, `∇ log π`도 noisy하며, 둘의 곱은 매우 noisy하다. 표준적인 해결책은 두 가지다.

1. **Baseline subtraction.** `G_t`를 `G_t - b(s_t)`로 바꾼다. 여기서 baseline `b(s_t)`는 `a_t`에 의존하지 않는 임의의 함수다. `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`이므로 unbiased다. 대표 선택: critic이 학습한 `b(s_t) = V̂(s_t)` → actor-critic(Lesson 07).
2. **Reward-to-go.** `Σ_t G_t · ∇ log π_θ(a_t | s_t)`를 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`로 바꾼다. 특정 action에는 미래 return만 중요하다. 과거 reward는 평균이 0인 noise만 더한다.

둘을 합치면 다음을 얻는다.

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

이것이 baseline이 있는 REINFORCE이며, A2C(Lesson 07)와 PPO(Lesson 08)의 직접 조상이다.

**Softmax policy parameterization.** Discrete action에서 표준 선택은 다음과 같다.

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

여기서 `f_θ`는 action별 score를 출력하는 임의의 신경망이다. gradient는 깔끔한 형태를 갖는다.

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

즉, 선택한 action의 score에서 policy 아래의 expected value를 뺀 것이다.

**Continuous action을 위한 Gaussian policy.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)`는 닫힌형식을 가진다. Phase 9 · 07의 SAC에 필요한 전부다.

```figure
policy-gradient-landscape
```

## 직접 만들기

### 1단계: softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Tabular env에는 linear policy(action마다 weight vector 하나)를 사용하라. Atari에서는 CNN으로 바꾸고 softmax head는 유지한다.

### 2단계: sampling과 log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 3단계: log-prob을 함께 기록하는 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 4단계: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)`(`a`의 onehot에서 probability를 뺀 것)가 softmax policy gradient의 핵심이다. 근육 기억으로 새겨라.

### 5단계: baseline

최근 episode의 `G` running mean만으로도 4×4 GridWorld를 돌릴 만큼 variance를 줄일 수 있다. 수렴에는 약 500 episode가 걸린다. baseline을 학습된 `V̂(s)`로 업그레이드하면 actor-critic이 된다.

## 함정

- **Exploding gradients.** Return은 매우 커질 수 있다. `∇ log π`를 곱하기 전에 batch 전체에서 항상 `G`를 `~N(0, 1)`로 normalize하라.
- **Entropy collapse.** Policy가 너무 일찍 거의 deterministic한 action으로 수렴하면 exploration을 멈추고 막힌다. 해결책: objective에 entropy bonus `β · H(π(·|s))`를 더하라.
- **High variance.** Vanilla REINFORCE에는 수천 episode가 필요하다. critic baseline(Lesson 07) 또는 TRPO/PPO의 trust region(Lesson 08)이 표준 해결책이다.
- **Sample inefficiency.** On-policy는 각 transition을 한 번 update한 뒤 버린다는 뜻이다. Importance sampling을 통한 off-policy correction은 데이터를 되살리지만 variance 비용이 있다(PPO의 ratio는 clipped IS weight다).
- **Non-stationary gradients.** 100 episode 전의 같은 gradient는 오래된 `π`를 사용한다. On-policy 방법이 rollout 몇 개마다 update하는 이유다.
- **Credit assignment.** Reward-to-go가 없으면 과거 reward가 noise를 더한다. 항상 reward-to-go를 사용하라.

## 활용하기

2026년에 REINFORCE를 직접 실행하는 경우는 드물지만, 그 gradient 공식은 어디에나 있다.

| 사용 사례 | 파생 방법 |
|----------|-----------|
| 연속 제어 | Gaussian policy를 쓰는 PPO / SAC |
| LLM RLHF | token-level policy에서 실행되는 KL penalty 포함 PPO |
| LLM reasoning(DeepSeek) | GRPO — group-relative baseline을 쓰고 critic이 없는 REINFORCE |
| Multi-agent | Centralized-critic REINFORCE(MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| preference-only 설정 | DPO — preference-likelihood loss로 다시 쓴 REINFORCE, sampling 없음 |

2026년 학습 script에서 `loss = -advantage * log_prob`을 보면, 그것은 baseline이 있는 REINFORCE다. DPO, GRPO, RLOO 같은 전체 논문들이 이 한 줄 위에 얹은 variance-reduction 요령이다.

## 산출물

`outputs/skill-policy-gradient-trainer.md`로 저장하라.

```markdown
---
name: policy-gradient-trainer
description: 주어진 작업에 대한 REINFORCE / actor-critic / PPO 학습 config를 만들고 variance 문제를 진단한다.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

환경(discrete / continuous actions, horizon, reward stats)이 주어지면 다음을 출력하라.

1. Policy head. Softmax(discrete) 또는 Gaussian(continuous), parameter count 포함.
2. Baseline. None(vanilla), running mean, 학습된 `V̂(s)`, 또는 A2C critic.
3. Variance 제어. Reward-to-go는 기본으로 켜고, return normalization, gradient clip value를 정한다.
4. Entropy bonus. Coefficient β와 decay schedule.
5. Batch size. Update당 episode 수와 on-policy data freshness contract.

horizon이 500 step을 넘는 REINFORCE-no-baseline은 거부하라. Softmax head를 쓰는 continuous-action control은 거부하라. `β = 0`이고 관측된 policy entropy가 0.1 미만인 run은 entropy-collapsed로 표시하라.
```

## 연습문제

1. **쉬움.** Linear softmax policy로 4×4 GridWorld에서 REINFORCE를 구현하라. Baseline 없이 1,000 episode 동안 학습하라. Learning curve를 그리고 variance(return의 std)를 측정하라.
2. **보통.** Running-mean baseline을 추가하라. 다시 학습하라. Sample efficiency와 variance를 vanilla run과 비교하라. Baseline이 수렴까지 필요한 step을 얼마나 줄였는가?
3. **어려움.** Entropy bonus `β · H(π)`를 추가하라. `β ∈ {0, 0.01, 0.1, 1.0}`를 sweep하라. Final return과 policy entropy를 그려라. 이 task에서 sweet spot은 어디인가?

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|------------------|-----------|
| Policy gradient | "policy를 직접 학습한다" | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`; log-derivative trick에서 유도된다. |
| REINFORCE | "원래 PG 알고리즘" | Williams (1992). Monte Carlo return에 log-policy gradient를 곱한다. |
| Log-derivative trick | "score-function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`. expectation의 gradient를 다룰 수 있게 한다. |
| Baseline | "variance 감소" | `G`에서 빼는 임의의 `b(s)`. `E[b · ∇ log π] = 0`이므로 unbiased다. |
| Reward-to-go | "미래 return만 센다" | 전체 `G_0` 대신 `G_t^{from t}`를 쓴다. 올바르고 variance가 낮다. |
| Entropy bonus | "탐색 장려" | `+β · H(π(·\|s))` 항이 policy collapse를 막는다. |
| On-policy | "방금 본 것으로 학습한다" | Gradient expectation은 current policy에 대한 것이다. old data를 직접 재사용할 수 없다. |
| Advantage | "평균보다 얼마나 나은가" | `A(s, a) = G(s, a) - V(s)`. REINFORCE-with-baseline이 곱하는 부호 있는 양. |

## 더 읽을거리

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — 원래 REINFORCE 논문.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 함수 근사를 포함한 현대적 policy-gradient theorem.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 교과서 설명.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — PyTorch 코드가 포함된 명확한 교육용 설명.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance reduction과 REINFORCE를 trust-region 계열(TRPO, PPO)에 연결하는 natural-gradient 관점.
