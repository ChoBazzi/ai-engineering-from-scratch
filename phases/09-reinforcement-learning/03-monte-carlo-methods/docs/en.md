# Monte Carlo Methods — 완전한 Episode로부터 학습하기

> Dynamic programming에는 model이 필요합니다. Monte Carlo에는 episode만 있으면 됩니다. policy를 실행하고, return을 관찰하고, 평균을 냅니다. RL에서 가장 단순한 아이디어이며, 이후의 모든 것을 열어주는 아이디어입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## 문제

Dynamic programming은 우아하지만, 모든 state와 action에 대해 `P(s' | s, a)`를 query할 수 있다고 가정합니다. 현실 세계의 거의 어떤 것도 그렇게 작동하지 않습니다. robot은 joint torque 이후 camera pixel 분포를 해석적으로 계산할 수 없습니다. pricing algorithm은 가능한 모든 customer reaction에 대해 적분할 수 없습니다. LLM은 token 이후 가능한 모든 continuation을 열거할 수 없습니다.

environment에서 *sample*을 뽑을 수 있기만 하면 되는 method가 필요합니다. policy를 실행합니다. trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`를 얻습니다. 이것으로 value를 추정합니다. 이것이 Monte Carlo입니다.

DP에서 MC로의 전환은 철학적으로 중요합니다. *known model + exact backup*에서 *sampled rollout + averaged return*으로 이동하기 때문입니다. variance는 커지지만 applicability는 폭발적으로 넓어집니다. 이 lesson 이후의 모든 RL 알고리즘, 즉 TD, Q-learning, REINFORCE, PPO, GRPO는 본질적으로 Monte Carlo estimator이며, 때로는 그 위에 bootstrapping이 얹혀 있습니다.

## 개념

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**핵심 아이디어를 한 줄로 쓰면:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`입니다. 여기서 `G^{(i)}(s)`는 policy `π` 아래에서 `s` 방문 이후 관찰한 return입니다.

**First-visit vs every-visit MC.** episode가 state `s`를 여러 번 방문한다면, first-visit MC는 첫 방문에서의 return만 셉니다. every-visit MC는 모든 방문을 셉니다. 둘 다 극한에서는 unbiased입니다. first-visit은 분석하기 더 쉽습니다(iid sample). every-visit은 episode당 더 많은 data를 사용하며 보통 실제로 더 빠르게 수렴합니다.

**Incremental mean.** 모든 return을 저장하는 대신 running average를 업데이트합니다.

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

다시 정리하면 `V_new = V_old + α · (target - V_old)`이고 `α = 1/n`입니다. `1/n`을 constant step-size `α ∈ (0, 1)`로 바꾸면 `π`의 변화를 추적하는 non-stationary MC estimator를 얻습니다. 이 이동이 MC에서 TD로, 그리고 모든 modern RL algorithm으로 넘어가는 전체 도약입니다.

**Exploration이 이제 문제가 됩니다.** DP는 enumeration으로 모든 state를 만졌습니다. MC는 policy가 방문하는 state만 봅니다. `π`가 deterministic이면 state space의 전체 영역이 sample되지 않고, 그 value estimate는 영원히 0에 머뭅니다. 역사적 순서로 세 가지 해결책이 있습니다.

1. **Exploring starts.** 각 episode를 random (s, a) pair에서 시작합니다. coverage를 보장하지만 실제로는 비현실적입니다(robot을 임의 state로 "reset"할 수는 없습니다).
2. **ε-greedy.** 현재 Q에 대해 greedy하게 행동하되 probability `ε`로 random action을 고릅니다. 모든 state-action pair가 asymptotically sample됩니다.
3. **Off-policy MC.** behavior policy `μ` 아래에서 data를 모으고, importance sampling으로 target policy `π`에 대해 학습합니다. variance는 높지만 DQN 같은 replay-buffer method로 이어지는 bridge입니다.

**Monte Carlo Control.** policy iteration처럼 evaluate → improve → evaluate를 수행하지만, evaluation은 sampling-based입니다.

1. `π`를 실행해 episode를 얻습니다.
2. 관찰한 return으로 `Q(s, a)`를 업데이트합니다.
3. `π`를 `Q`에 대한 ε-greedy로 만듭니다.
4. 반복합니다.

온건한 조건(모든 pair가 무한히 자주 방문되고, `α`가 Robbins-Monro를 만족함) 아래에서 probability 1로 `Q*`와 `π*`에 수렴합니다.

```figure
epsilon-greedy
```

## 직접 만들기

### 1단계: rollout → (s, a, r)의 list

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

model은 없고 `env.reset()`과 `env.step(s, a)`만 있습니다. gym environment와 같은 interface지만 더 간단하게 벗겨낸 형태입니다.

### 2단계: return 계산(reverse sweep)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

한 번의 pass, `O(T)`입니다. backward recurrence `G_t = r_{t+1} + γ G_{t+1}`는 다시 합산하는 일을 피합니다.

### 3단계: first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

세 줄이 핵심 작업을 합니다. 첫 방문에서 state를 seen으로 표시하고, count를 증가시키고, running mean을 업데이트합니다.

### 4단계: ε-greedy MC control(on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 5단계: DP gold standard와 비교하기

`V^π`의 MC estimate는 episode → ∞일 때 Lesson 02의 DP result와 일치해야 합니다. 실제로 4×4 GridWorld에서 50,000 episode를 실행하면 DP answer의 `~0.1` 이내로 들어옵니다.

## 함정

- **Infinite episode.** MC는 episode가 *terminate*되어야 합니다. policy가 영원히 loop할 수 있다면 `max_steps`를 두고 cap을 implicit failure로 취급하세요. random policy의 GridWorld는 자주 timeout됩니다. 정상입니다. 다만 올바르게 세야 합니다.
- **Variance.** MC는 full return을 사용합니다. 긴 episode에서는 variance가 큽니다. 끝의 불운한 reward 하나가 `V(s_0)`를 같은 크기만큼 흔듭니다. TD method(Lesson 04)는 bootstrapping으로 이것을 줄입니다.
- **State coverage.** fresh Q에서 tie가 있는 greedy MC는 한 action만 계속 시도합니다. 반드시 explore해야 합니다(ε-greedy, exploring starts, UCB).
- **Non-stationary policy.** `π`가 변하면(MC control처럼) old return은 다른 policy에서 나온 것입니다. Constant-α MC는 이것을 처리하지만 sample-average MC는 그렇지 않습니다.
- **Off-policy importance sampling.** weight `π(a|s)/μ(a|s)`는 trajectory를 따라 곱해집니다. horizon이 길면 variance가 폭발합니다. per-decision weighted IS로 제한하거나 TD로 전환하세요.

## 활용하기

2026년 Monte Carlo method의 역할은 다음과 같습니다.

| 사용 사례 | MC를 쓰는 이유 |
|----------|----------------|
| 짧은 horizon의 게임(blackjack, poker) | episode가 자연스럽게 끝나고 return이 깔끔함. |
| logged policy의 offline evaluation | 저장된 trajectory 위에서 discounted return을 평균냄. |
| Monte Carlo Tree Search (AlphaZero) | tree leaf에서의 MC rollout이 selection을 안내함. |
| LLM RL evaluation | 주어진 policy에 대해 sampled completion의 average reward를 계산함. |
| PPO의 baseline estimation | advantage target `A_t = G_t - V(s_t)`가 MC `G_t`를 사용함. |
| RL 교육 | 실제로 작동하는 가장 단순한 알고리즘. bootstrapping을 제거해 core를 볼 수 있음. |

Modern deep-RL algorithm(PPO, SAC)은 `n`-step return 또는 GAE를 통해 pure MC(full return)와 pure TD(one-step bootstrap) 사이를 보간합니다. 양 끝점 모두 같은 estimator의 instance입니다.

## 결과물 만들기

`outputs/skill-mc-evaluator.md`로 저장하세요.

```markdown
---
name: mc-evaluator
description: Monte Carlo rollout으로 policy를 평가하고, 가능하면 DP 비교를 포함한 수렴 보고서를 만듭니다.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

environment(episodic, reset+step API 포함)와 policy가 주어지면 다음을 출력하세요.

1. 방법. First-visit vs every-visit MC. 이유.
2. Episode 예산. 목표 개수, variance diagnostic, expected standard error.
3. 탐색 계획. ε schedule(필요하면) 또는 exploring starts.
4. Gold-standard 비교. tabular이면 DP-optimal V*, 아니면 Q-learning / PPO baseline에서 얻은 bound.
5. 종료 확인. Max-step cap, timeout, non-terminating trajectory 처리.

finite horizon cap이 없는 non-episodic 과제에서는 MC 실행을 거부하세요. tabular 과제에서 state당 100 episode 미만으로 얻은 V^π 추정치는 보고하지 마세요. action variance가 0인 policy는 exploration risk로 표시하세요.
```

## 연습문제

1. **Easy.** 4×4 GridWorld에서 uniform-random policy의 first-visit MC evaluation을 구현하세요. 10,000 episode를 실행하세요. episode count에 따른 `V(0,0)`를 DP answer와 함께 plot하세요.
2. **Medium.** `ε ∈ {0.01, 0.1, 0.3}`로 ε-greedy MC control을 구현하세요. 20,000 episode 이후 mean return을 비교하세요. curve는 어떤 모양인가요? bias-variance tradeoff는 어디에 있나요?
3. **Hard.** importance sampling을 사용하는 *off-policy* MC를 구현하세요. uniform-random policy `μ` 아래에서 data를 모으고 deterministic optimal policy `π`의 `V^π`를 추정합니다. plain IS, per-decision IS, weighted IS를 비교하세요. 어느 것이 variance가 가장 낮나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|--------------------------|-----------|
| Monte Carlo | "무작위 sampling" | distribution에서 iid sample을 평균내 expectation을 추정. |
| Return `G_t` | "미래 reward" | step `t`부터 episode 끝까지 discounted reward의 합: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "각 state를 한 번만 세기" | 한 episode에서 첫 방문만 value estimate에 기여. |
| Every-visit MC | "모든 방문 사용" | 모든 방문이 기여. 약간 biased이지만 sample-efficient함. |
| ε-greedy | "탐색 noise" | probability `1-ε`로 greedy action, probability `ε`로 random action 선택. |
| Importance sampling | "잘못된 distribution에서 sample한 것을 보정" | `μ` data에서 `V^π`를 추정하기 위해 return에 `π(a\|s)/μ(a\|s)` product를 곱함. |
| On-policy | "내 data로 학습" | target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "다른 사람의 data로 학습" | target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## 더 읽을거리

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 표준 설명입니다.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit vs every-visit analysis입니다.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC와 variance control입니다.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — modern low-variance IS estimator입니다.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — MC/TD self-play가 superhuman play로 수렴할 수 있음을 보여준 첫 대규모 실증이며, 이 phase 후반 모든 lesson의 개념적 선행 사례입니다.
