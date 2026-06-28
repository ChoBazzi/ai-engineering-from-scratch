# Temporal Difference — Q-Learning과 SARSA

> Monte Carlo는 episode가 끝날 때까지 기다립니다. TD는 다음 value estimate를 bootstrap해 매 step마다 업데이트합니다. Q-learning은 off-policy이고 optimistic합니다. SARSA는 on-policy이고 cautious합니다. 둘 다 code 한 줄입니다. 둘 다 이 phase의 모든 deep-RL method를 떠받칩니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 minutes

## 문제

Monte Carlo는 작동하지만 두 가지 비용이 큰 요구사항이 있습니다. 끝나는 episode가 필요하고, final return이 들어올 때까지 업데이트하지 않습니다. episode가 1,000 step이면 MC는 무엇이든 업데이트하기 위해 1,000 step을 기다립니다. high-variance, low-bias이며 실제로는 느립니다.

Dynamic programming은 반대 성격을 가집니다. zero-variance bootstrapped backup이지만 known model이 필요합니다.

Temporal difference(TD) learning은 그 중간을 택합니다. 단일 transition `(s, a, r, s')`에서 one-step target `r + γ V(s')`를 만들고 `V(s)`를 그쪽으로 조금 이동시킵니다. model이 필요 없습니다. complete episode도 필요 없습니다. RHS에 approximate `V`를 쓰기 때문에 bias가 생기지만, MC보다 variance가 훨씬 낮고 첫 step부터 online update가 가능합니다.

이것이 modern RL 전체(DQN, A2C, PPO, SAC)가 회전하는 pivot입니다. Phase 9의 나머지는 이 lesson에서 작성할 one-step TD update 위에 function approximation과 trick을 얹은 것입니다.

## 개념

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V에 대한 TD(0) update:**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

괄호 안의 값은 TD error `δ = r + γ V(s') - V(s)`입니다. MC의 `G_t - V(s_t)`에 대응하는 online analogue입니다. 수렴하려면 `α`가 Robbins-Monro(`Σ α = ∞`, `Σ α² < ∞`)를 만족하고 모든 state가 무한히 자주 방문되어야 합니다.

**Q-learning.** control을 위한 off-policy TD method입니다.

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

`max`는 agent가 실제로 어떤 action을 취하든 `s'` 이후에는 *greedy* policy가 따를 것이라고 가정합니다. 이 decoupling 덕분에 agent가 ε-greedy로 explore하는 동안에도 Q-learning은 `Q*`를 학습합니다. Mnih et al. (2015)은 이것을 Atari의 deep Q-learning으로 바꾸었습니다(Lesson 05).

**SARSA.** on-policy TD method입니다.

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

이름은 tuple `(s, a, r, s', a')`에서 왔습니다. SARSA는 greedy `argmax`가 아니라 agent가 다음에 *실제로* 취하는 action `a'`를 사용합니다. 실행 중인 ε-greedy `π`가 무엇이든 그에 대한 `Q^π`로 수렴하며, 극한에서 `ε → 0`이면 `Q*`가 됩니다.

**Cliff-walking 차이.** classic cliff-walking task(fall-off-cliff = reward -100)에서 Q-learning은 cliff edge를 따라가는 optimal path를 학습하지만 exploration 중에는 가끔 penalty를 받습니다. SARSA는 exploration noise를 Q-value에 반영하므로 cliff에서 한 칸 떨어진 더 안전한 path를 학습합니다. 학습이 진행되고 `ε → 0`이면 둘 다 optimal에 도달합니다. 실제로는 중요합니다. deployment에서 exploration이 실제로 일어난다면 SARSA의 behavior가 더 보수적입니다.

**Expected SARSA.** `Q(s', a')`를 `π` 아래의 expected value로 바꿉니다.

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

SARSA보다 variance가 낮고(`a'` sample이 없음), 같은 on-policy target입니다. modern textbook에서는 종종 default입니다.

**n-step TD와 TD(λ).** bootstrap하기 전에 `n` step을 기다려 TD(0)와 MC 사이를 보간합니다. `n=1`은 TD이고, `n=∞`는 MC입니다. TD(λ)는 geometric weight `(1-λ)λ^{n-1}`로 모든 `n`을 평균냅니다. 대부분의 deep-RL은 3에서 20 사이의 `n`을 사용합니다.

```figure
qlearning-gridworld
```

## 직접 만들기

### 1단계: ε-greedy policy 위의 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

여덟 줄입니다. Q-learning과의 *유일한* 차이는 target line입니다.

### 2단계: Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max`가 target과 behavior를 분리합니다. 이 한 symbol이 on-policy와 off-policy의 차이입니다.

### 3단계: learning curve

100 episode마다 mean return을 추적하세요. Q-learning은 단순한 deterministic GridWorld에서 더 빠르게 수렴합니다. SARSA는 cliff-walking에서 더 보수적입니다. `code/main.py`의 4×4 GridWorld에서는 `α=0.1, ε=0.1`일 때 둘 다 약 2,000 episode 뒤 near-optimal입니다.

### 4단계: DP truth와 비교하기

value iteration(Lesson 02)을 실행해 `Q*`를 구하세요. `max_{s,a} |Q_learned(s,a) - Q*(s,a)|`를 확인하세요. 건강한 tabular TD agent는 4×4 GridWorld에서 10,000 episode 뒤 `~0.5` 이내에 들어옵니다.

## 함정

- **Initial Q value가 중요합니다.** optimistic init(negative-reward task에서 `Q = 0`)은 exploration을 장려합니다. pessimistic init은 greedy policy를 영원히 가둘 수 있습니다.
- **α schedule.** Constant `α`는 non-stationary problem에 적합합니다. decaying `α_n = 1/n`은 이론적으로 수렴하지만 실제로는 너무 느립니다. `α`를 `[0.05, 0.3]`에 고정하고 learning curve를 monitor하세요.
- **ε schedule.** 높게 시작해(`ε=1.0`) `ε=0.05`까지 decay하세요. "GLIE"(greedy in the limit with infinite exploration)가 convergence condition입니다.
- **Q-learning의 max bias.** `Q`가 noisy할 때 `max` operator는 upward-biased입니다. overestimation으로 이어집니다. Hasselt의 Double Q-learning(Lesson 05의 DDQN에서 사용)이 두 Q table로 이것을 고칩니다.
- **Non-terminating episode.** TD는 terminal 없이도 학습할 수 있지만 step을 cap하거나 cap에서 bootstrap을 올바르게 처리해야 합니다. 표준 방식은 cap을 non-terminal로 취급하고 계속 bootstrap하는 것입니다.
- **State hashing.** state가 tuple/tensor라면 hashable key를 사용하세요(list가 아니라 tuple, raw float가 아니라 rounded float의 tuple).

## 활용하기

2026년 TD landscape는 다음과 같습니다.

| 과제 | 방법 | 이유 |
|------|--------|--------|
| 작은 tabular environment | Q-learning | optimal policy를 직접 학습함. |
| On-policy 안전 중요 과제 | SARSA / Expected SARSA | exploration 중에 보수적임. |
| 고차원 state | DQN (Phase 9 · 05) | replay와 target net을 가진 neural-net Q-function. |
| Continuous action | SAC / TD3 (Phase 9 · 07) | Q-network에 대한 TD update. policy net은 action을 냄. |
| LLM RL (reward-model-based) | PPO / GRPO (Phase 9 · 08, 12) | GAE를 통한 TD-style advantage가 있는 actor-critic. |
| Offline RL | CQL / IQL (Phase 9 · 08) | conservative regularization이 있는 Q-learning. |

2026년 논문에서 읽는 "RL"의 90%는 Q-learning 또는 SARSA의 어떤 elaboration입니다. 더 깊이 읽기 전에 tabular update를 손에 익히세요.

## 결과물 만들기

`outputs/skill-td-agent.md`로 저장하세요.

```markdown
---
name: td-agent
description: tabular 또는 small-feature RL 과제에 대해 Q-learning, SARSA, Expected SARSA 중에서 선택합니다.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

tabular 또는 small-feature environment가 주어지면 다음을 출력하세요.

1. 알고리즘. Q-learning / SARSA / Expected SARSA / n-step variant. on-policy vs off-policy와 variance에 연결한 한 문장 이유.
2. Hyperparameter. α, γ, ε, decay schedule.
3. 초기화. Q_0 값(optimistic vs zero)과 근거.
4. 수렴 진단. Target learning curve, DP가 가능하면 `|Q - Q*|` check.
5. 배포 caveat. inference에서 exploration이 어떻게 동작할지. SARSA의 보수성이 필요한지.

state space가 10⁶을 넘으면 tabular TD 적용을 거부하세요. max-bias caveat 없는 Q-learning agent는 제공하지 마세요. ε를 끝까지 1.0으로 유지해 학습한 agent(no exploitation phase)는 표시하세요.
```

## 연습문제

1. **Easy.** 4×4 GridWorld에서 Q-learning과 SARSA를 구현하세요. 2,000 episode 동안 learning curve(100 episode당 mean return)를 plot하세요. 누가 더 빨리 수렴하나요?
2. **Medium.** cliff-walking environment(4×12, 마지막 row는 reward -100을 주고 start로 reset되는 cliff)를 만드세요. Q-learning과 SARSA의 final policy를 비교하세요. 각 path를 screenshot으로 남기세요. 어느 쪽이 cliff에 더 가깝나요?
3. **Hard.** Double Q-learning을 구현하세요. noisy-reward GridWorld(per-step reward에 Gaussian noise σ=5 추가)에서 Q-learning이 `V*(0,0)`를 의미 있게 overestimate하지만 Double Q-learning은 그렇지 않음을 보이세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|--------------------------|-----------|
| TD error | "업데이트 신호" | `δ = r + γ V(s') - V(s)`, bootstrapped residual. |
| TD(0) | "one-step TD" | 다음 state의 estimate만 사용해 transition마다 update. |
| Q-learning | "off-policy RL 입문" | next-state action 위의 `max`를 쓰는 TD update. behavior policy와 무관하게 `Q*`를 학습. |
| SARSA | "on-policy Q-learning" | 실제 next action을 사용하는 TD update. 현재 ε-greedy π에 대한 `Q^π`를 학습. |
| Expected SARSA | "낮은 variance의 SARSA" | sampled `a'`를 π 아래 expectation으로 대체. |
| GLIE | "올바른 탐색 schedule" | Greedy in the Limit with Infinite Exploration. Q-learning convergence에 필요. |
| Bootstrapping | "target에 현재 estimate 사용" | TD와 MC를 구분하는 것. bias의 원인이지만 variance를 크게 줄임. |
| Maximization bias | "Q-learning의 과대추정" | noisy estimate 위의 `max`는 upward-biased. Double Q-learning으로 고침. |

## 더 읽을거리

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — original paper와 convergence proof입니다.
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0), SARSA, Q-learning, Expected SARSA를 다룹니다.
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — maximization bias를 고치는 방법입니다.
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA의 동기입니다.
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — SARSA라는 이름을 만든 paper입니다(당시에는 "modified connectionist Q-learning"이라고 불림).
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)를 TD(n)으로 일반화합니다. Q-learning에서 eligibility trace로, 나중에는 PPO의 GAE로 이어지는 길입니다.
