# Actor-Critic — A2C와 A3C

> REINFORCE는 noisy하다. `V̂(s)`를 학습하는 critic을 추가하고 return에서 이를 빼면, expectation은 같지만 variance가 훨씬 낮은 advantage를 얻는다. 이것이 actor-critic이다. A2C는 이를 동기식으로 실행하고, A3C는 thread 전체에서 실행한다. 둘 다 모든 현대 deep-RL 방법의 mental model이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 minutes

## 문제

Vanilla REINFORCE는 작동하지만 variance가 끔찍하다. Monte Carlo return `G_t`는 episode 사이에서 10배 이상 흔들릴 수 있다. 그 noise에 `∇ log π`를 곱해 평균내면, 훨씬 적은 DQN update로 움직일 수 있는 policy 거리만큼 이동하는 데 수천 episode가 걸리는 gradient estimator가 나온다.

Variance는 raw return을 사용하기 때문에 생긴다. Baseline `b(s_t)`, 즉 state의 임의 함수(학습된 value 포함)를 빼면 expectation은 변하지 않고 variance가 줄어든다. 다룰 수 있는 가장 좋은 baseline은 `V̂(s_t)`다. 이제 `∇ log π`에 곱하는 양은 *advantage*가 된다.

`A(s, a) = G - V̂(s)`

Action이 평균보다 높은 return을 만들었으면 좋고, 낮으면 나쁘다. 학습된 critic이 있는 REINFORCE가 *actor-critic*이다. Critic은 actor에게 낮은 variance의 teacher를 제공한다. 이것이 2015년 이후 모든 deep-policy 방법(A2C, A3C, PPO, SAC, IMPALA)이다.

## 개념

![Actor-critic: policy net과 value net, advantage로 쓰는 TD residual](../assets/actor-critic.svg)

**두 네트워크, 하나의 공유 loss:**

- **Actor** `π_θ(a | s)`: policy다. 행동을 샘플링하는 데 쓰이며 policy gradient로 학습된다.
- **Critic** `V_φ(s)`: state에서의 expected return을 추정한다. `(V_φ(s) - target)²`를 최소화하도록 학습된다.

**Advantage.** 표준 형태는 두 가지다.

- *MC advantage:* `A_t = G_t - V_φ(s_t)`. Unbiased지만 variance가 높다.
- *TD advantage:* `A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`. Biased(`V_φ`를 사용)지만 variance가 훨씬 낮다. *TD residual* `δ_t`라고도 한다.

**n-step advantage.** 둘 사이를 보간한다.

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

`n = 1`은 순수 TD다. `n = ∞`는 MC다. 대부분의 구현은 Atari에 `n = 5`, MuJoCo의 PPO에 `n = 2048`을 쓴다.

**Generalized Advantage Estimation(GAE).** Schulman et al. (2016)은 모든 n-step advantage에 대한 지수 가중 평균을 제안했다.

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

여기서 `λ ∈ [0, 1]`이다. `λ = 0`은 TD(low variance, high bias)다. `λ = 1`은 MC(high variance, unbiased)다. `λ = 0.95`가 2026년 기본값이다. bias/variance 다이얼이 원하는 위치에 올 때까지 조정하라.

**A2C: synchronous advantage actor-critic.** `N`개의 parallel environment에서 `T` step을 수집한다. 각 step의 advantage를 계산한다. 결합된 batch에서 actor와 critic을 update한다. 반복한다. A3C의 더 단순하고 scale하기 쉬운 sibling이다.

**A3C: asynchronous advantage actor-critic.** Mnih et al. (2016). `N`개의 worker thread를 띄우고 각각 env를 실행한다. 각 worker는 자기 rollout에서 local gradient를 계산한 뒤 shared parameter server에 비동기로 적용한다. Replay buffer가 필요 없다. worker들이 서로 다른 trajectory를 실행해 decorrelation을 만든다. A3C는 CPU에서 scale 학습이 가능하다는 것을 보였다. 2026년에는 GPU가 큰 batch를 원하기 때문에 GPU 기반 A2C(batched parallel env)가 지배적이다.

**결합 loss.**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

세 항은 policy-gradient loss, value regression, entropy bonus다. `c_v ~ 0.5`, `c_e ~ 0.01`은 표준 시작점이다.

## 직접 만들기

### 1단계: critic

Linear critic `V_φ(s) = w · features(s)`를 MSE로 update한다.

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

Tabular env에서는 critic이 몇백 episode 안에 수렴한다. Atari에서는 linear critic을 shared CNN trunk + value head로 바꾼다.

### 2단계: n-step advantage

길이 `T`의 rollout과 bootstrapped final `V(s_T)`가 주어졌을 때:

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns`는 critic target이다. `advantages`는 `∇ log π`에 곱해지는 값이다.

### 3단계: 결합 update

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy이며 update마다 rollout 하나를 쓰고, actor와 critic에는 별도 learning rate를 둔다.

### 4단계: 병렬화(A3C vs A2C)

- **A3C:** `N`개의 thread를 띄운다. 각 thread는 자기 env와 자기 forward pass를 실행한다. 주기적으로 shared master에 gradient update를 push한다. master에 lock은 없다. race는 괜찮고 noise를 더할 뿐이다.
- **A2C:** 단일 process에서 `N`개의 env instance를 실행하고, observation을 `[N, obs_dim]` batch로 stack한 뒤 batched forward pass와 batched backward pass를 수행한다. GPU utilization이 높고 deterministic하며 reasoning하기 쉽다. 2026년의 기본값이다.

우리 toy code는 명확성을 위해 single-threaded다. Batched A2C로 고치는 것은 numpy 세 줄이면 된다.

## 함정

- **Actor gradient 전 critic bias.** Critic이 random이면 baseline은 정보가 없고 순수 noise로 학습하게 된다. Policy gradient를 켜기 전에 critic을 몇백 step warm up하거나 actor learning rate를 느리게 둔다.
- **Advantage normalization.** Batch마다 advantage를 zero-mean/unit-std로 normalize하라. 비용은 거의 없고 학습 안정성이 크게 좋아진다.
- **Shared trunk.** Image input에서는 actor와 critic이 shared feature extractor를 쓰고 head만 분리하라. Shared feature는 두 loss에서 함께 이득을 얻는다.
- **On-policy contract.** A2C는 데이터를 정확히 한 번의 update에만 재사용한다. 더 재사용하면 gradient가 biased된다(importance-sampling correction이 PPO가 추가하는 것이다).
- **Entropy collapse.** `c_e > 0`이 없으면 policy가 몇백 update 안에 거의 deterministic해지고 exploration을 멈춘다.
- **Reward scale.** Advantage 크기는 reward scale에 의존한다. Task 사이에서 gradient 크기를 일관되게 만들려면 reward를 normalize하라(예: running-std로 나누기).

## 활용하기

A2C/A3C는 2026년에 최종 선택인 경우가 드물지만, 이후 방법들이 다듬는 기본 아키텍처다.

| 방법 | A2C와의 관계 |
|------|--------------|
| PPO | multi-epoch update를 위한 clipped importance ratio를 추가한 A2C |
| IMPALA | V-trace off-policy correction을 추가한 A3C |
| SAC (Phase 9 · 07) | soft-value critic을 쓰는 off-policy A2C(다음 lesson) |
| GRPO (Phase 9 · 12) | critic이 없는 A2C — group-relative advantage |
| DPO | sampling이 없는 preference-ranking loss로 축약된 A2C |
| AlphaStar / OpenAI Five | league training + imitation pre-training을 곁들인 A2C |

2026년 논문에서 "advantage"를 보면 actor-critic을 떠올려라.

## 산출물

`outputs/skill-actor-critic-trainer.md`로 저장하라.

```markdown
---
name: actor-critic-trainer
description: 주어진 환경을 위한 A2C / A3C / GAE configuration을 만들고, advantage estimation과 loss weight를 명시한다.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

환경과 compute budget이 주어지면 다음을 출력하라.

1. 병렬성. A2C(GPU batched) vs A3C(CPU async)와 worker 수.
2. Rollout 길이 T. Update마다 env별 step 수.
3. Advantage 추정기. n-step 또는 GAE(λ). λ를 명시한다.
4. Loss 가중치. `c_v`(value), `c_e`(entropy), gradient clip.
5. Learning rate. Actor와 critic(사용한다면 분리).

horizon이 1000을 넘는 환경에서 single-worker A2C는 거부하라(너무 on-policy이고 너무 느리다). Advantage normalization 없이 ship하지 말라. `c_e = 0`이고 관측된 entropy가 0.1 미만인 run은 entropy-collapsed로 표시하라.
```

## 연습문제

1. **쉬움.** 4×4 GridWorld에서 MC advantage(`G_t - V(s_t)`)로 actor-critic을 학습하라. Lesson 06의 REINFORCE-with-running-mean-baseline과 sample efficiency를 비교하라.
2. **보통.** TD-residual advantage(`r + γ V(s') - V(s)`)로 바꿔라. Advantage batch의 variance를 측정하라. 얼마나 줄어드는가?
3. **어려움.** GAE(λ)를 구현하라. `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}`를 sweep하라. Final return vs sample efficiency를 그려라. 이 task에서 bias/variance sweet spot은 어디인가?

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|------------------|-----------|
| Actor | "policy net" | `π_θ(a\|s)`, policy gradient로 update된다. |
| Critic | "value net" | `V_φ(s)`, return / TD target에 대한 MSE regression으로 update된다. |
| Advantage | "평균보다 얼마나 나은가" | `A(s, a) = Q(s, a) - V(s)` 또는 그 estimator들. `∇ log π`의 multiplier다. |
| TD residual | "δ" | `δ_t = r + γ V(s') - V(s)`. one-step advantage estimate. |
| GAE | "보간 knob" | `λ`로 parameterize된 n-step advantage의 지수 가중합. |
| A2C | "동기식 actor-critic" | env 전체에서 batched 처리한다. Rollout마다 gradient step 하나. |
| A3C | "비동기 actor-critic" | Worker thread가 shared param server에 gradient를 push한다. 원 논문 방식이며 2026년에는 덜 흔하다. |
| Bootstrap | "horizon에서 V 사용" | Rollout을 자르고 `γ^n V(s_{t+n})`를 더해 합을 닫는다. |

## 더 읽을거리

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C, 원래 async actor-critic 논문.
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE.
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 기초. Critic이 신경망일 때 함수 근사에 대한 Ch. 9와 함께 읽어라.
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — V-trace off-policy correction을 쓰는 scalable distributed actor-critic.
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 읽어볼 만한 production A2C/PPO 구현.
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — two-timescale actor-critic decomposition의 기초 수렴 결과.
