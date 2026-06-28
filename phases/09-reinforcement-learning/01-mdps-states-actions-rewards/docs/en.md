# MDP, 상태, 행동, 보상

> Markov Decision Process는 다섯 가지로 이루어집니다. 상태, 행동, 전이, 보상, 할인율입니다. RL의 모든 것, 즉 Q-learning, PPO, DPO, GRPO는 이 형태 위에서 최적화됩니다. 한 번 익히면 강화학습의 나머지를 훨씬 쉽게 읽을 수 있습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## 문제

체스 봇을 작성한다고 해봅시다. 또는 재고 계획기, 트레이딩 agent, reasoning model을 학습하는 PPO loop일 수도 있습니다. 네 도메인은 서로 다르지만 놀라운 공통점이 있습니다. 모두 같은 수학적 객체로 접힙니다.

지도학습은 `(x, y)` 쌍을 주고 함수를 맞추라고 합니다. 강화학습은 label을 주지 않습니다. 상태의 흐름, 내가 취한 행동, scalar reward만 줍니다. 그 수가 게임을 이겼나요? 재입고 결정이 비용을 줄였나요? 거래가 이익을 냈나요? LLM이 방금 만든 token이 judge의 더 높은 reward로 이어졌나요?

이 흐름을 정식화하기 전에는 학습할 수 없습니다. "무엇을 보았는가", "무엇을 했는가", "다음에 무엇이 일어났는가", "그것이 얼마나 좋았는가"가 각각 추론 가능한 객체가 되어야 합니다. 이 정식화가 Markov Decision Process입니다. 이 phase의 모든 RL 알고리즘은 마지막의 RLHF와 GRPO loop까지 포함해 이 형태 위에서 최적화됩니다.

## 개념

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**다섯 객체.**

- **States** `S`. agent가 결정하는 데 필요한 모든 것. GridWorld에서는 cell, 체스에서는 board, LLM에서는 context window와 모든 memory입니다.
- **Actions** `A`. 선택지입니다. 위/아래/왼쪽/오른쪽으로 이동하기, 수를 두기, token을 내보내기입니다.
- **Transitions** `P(s' | s, a)`. 상태 `s`와 행동 `a`가 주어졌을 때 다음 상태에 대한 분포입니다. 체스에서는 deterministic, 재고에서는 stochastic, LLM decoding에서는 거의 deterministic입니다.
- **Rewards** `R(s, a, s')`. scalar signal입니다. 승리 = +1, 패배 = -1. 매출에서 비용을 뺀 값. GRPO의 log-likelihood ratio 항입니다.
- **Discount** `γ ∈ [0, 1)`. 미래 reward를 현재에 비해 얼마나 반영할지입니다. `γ = 0.99`는 약 100 step의 horizon을 사고, `γ = 0.9`는 약 10 step을 삽니다.

**Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. 미래는 현재 상태에만 의존합니다. 그렇지 않다면 state representation이 불완전한 것입니다. 이는 method의 실패가 아니라 state의 실패입니다.

**Policy와 return.** policy `π(a | s)`는 state를 action distribution에 매핑합니다. return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …`은 미래 reward의 discounted sum입니다. value `V^π(s) = E[G_t | s_t = s]`는 policy `π` 아래에서 `s`부터 시작할 때의 expected return입니다. Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`는 특정 action으로 시작할 때의 expected return입니다. 모든 RL 알고리즘은 이 둘 중 하나를 추정한 다음 그에 맞게 `π`를 개선합니다.

**Bellman equation.** 이 phase의 모든 것이 사용하는 fixed-point equation입니다.

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

이 식은 expected return을 "이번 step의 reward"와 "도착한 곳의 discounted value"로 나눕니다. 재귀적입니다. Phase 9의 모든 알고리즘은 이 equation을 수렴할 때까지 반복하거나(dynamic programming), 여기서 sample을 뽑거나(Monte Carlo), 한 step만 bootstrap합니다(temporal difference).

```figure
discount-horizon
```

## 직접 만들기

### 1단계: 작은 deterministic MDP

4×4 GridWorld입니다. agent는 왼쪽 위에서 시작하고, terminal은 오른쪽 아래이며, step마다 reward -1을 받고, action은 `{up, down, left, right}`입니다. `code/main.py`를 참고하세요.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

다섯 줄이면 전체 environment가 됩니다. deterministic transition, constant step penalty, absorbing terminal state입니다.

### 2단계: policy rollout

policy는 state에서 action distribution으로 가는 함수입니다. 가장 단순한 것은 uniform random입니다.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

random policy를 1000번 실행해보세요. 이 4×4 board에서 average return은 대략 -60에서 -80 사이입니다. optimal return은 -6입니다(아래와 오른쪽으로 직선 이동). 이 격차를 줄이는 것이 Phase 9의 전부입니다.

### 3단계: Bellman equation으로 `V^π`를 정확히 계산하기

작은 MDP에서는 Bellman equation이 linear system입니다. state를 열거하고 expectation을 적용한 뒤 값이 더 이상 변하지 않을 때까지 반복합니다.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

이것이 iterative policy evaluation입니다. Sutton & Barto의 첫 번째 알고리즘이며, 뒤따르는 모든 RL method의 이론적 토대입니다.

### 4단계: `γ`는 물리적 의미가 있는 hyperparameter

effective horizon은 대략 `1 / (1 - γ)`입니다. `γ = 0.9` → 10 step. `γ = 0.99` → 100 step. `γ = 0.999` → 1000 step.

너무 낮으면 agent가 근시안적으로 행동합니다. 너무 높으면 먼 미래 reward에 대한 책임을 많은 초기 step이 공유하므로 credit assignment가 noisy해집니다. LLM RLHF는 episode가 짧고 bounded이기 때문에 보통 `γ = 1`을 씁니다. control task는 `0.95–0.99`를 씁니다. long-horizon strategy game은 `0.999`를 씁니다.

## 함정

- **Non-Markovian state.** 결정하려면 최근 세 observation이 필요하다면 "state"는 현재 observation만이 아닙니다. 해결: frame을 쌓거나(Atari의 DQN은 4개를 쌓음) recurrent state를 사용합니다(observation 위의 LSTM/GRU).
- **Sparse reward.** 승리만 reward로 주면 큰 state space에서 학습이 거의 불가능합니다. reward를 shape하거나(intermediate signal), imitation으로 bootstrap하세요(Phase 9 · 09).
- **Reward hacking.** proxy reward를 최적화하면 병적인 행동이 자주 생깁니다. OpenAI의 boat-racing agent는 race를 끝내지 않고 powerup을 계속 모으려고 원을 그리며 돌았습니다. reward는 항상 proxy가 아니라 target outcome에서 정의하세요.
- **Discount mis-spec.** infinite-horizon task에서 `γ = 1`이면 모든 value가 무한대가 됩니다. 항상 finite horizon 또는 `γ < 1`로 제한하세요.
- **Reward scale.** reward {+100, -100}과 {+1, -1}은 같은 optimal policy를 주지만 gradient magnitude는 크게 다릅니다. PPO/DQN에 넣기 전에 `[-1, 1]` 근처로 normalize하세요.

## 활용하기

2026년의 stack은 코드를 만지기 전에 모든 RL pipeline을 MDP로 축소합니다.

| 상황 | 상태 | 행동 | 보상 | γ |
|------|-------|--------|--------|---|
| 제어(locomotion, manipulation) | 관절 각도 + 속도 | 연속 torque | 과제별 shaped reward | 0.99 |
| 게임(chess, Go, poker) | board + history | 합법 수 | 승리=+1 / 패배=-1 | 1.0 (finite) |
| 재고 / 가격 책정 | 재고 + 수요 | 주문량 | 매출 - 비용 | 0.95 |
| LLM용 RLHF | context token | next token | 끝의 reward-model score | 1.0 (episode ~200 tokens) |
| reasoning용 GRPO | prompt + partial response | next token | 끝의 verifier 0/1 | 1.0 |

training loop를 작성하기 전에 다섯 tuple을 먼저 쓰세요. "RL이 작동하지 않는다"는 bug report 대부분은 종이에 적은 MDP formulation이 망가져 있었던 데서 시작합니다.

## 결과물 만들기

`outputs/skill-mdp-modeler.md`로 저장하세요.

```markdown
---
name: mdp-modeler
description: 과제 설명이 주어지면 Markov Decision Process 명세를 만들고 학습 전에 정식화 위험을 표시합니다.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

과제(control / game / recommendation / LLM fine-tuning)가 주어지면 다음을 출력하세요.

1. State. 정확한 feature vector 또는 tensor 명세. Markov property를 정당화합니다.
2. Action. 이산 집합 또는 연속 범위. 차원 수.
3. Transition. Deterministic, stochastic-with-known-model, 또는 sample-only.
4. Reward. 함수와 출처. Sparse인지 shaped인지. Terminal인지 per-step인지.
5. Discount. 값과 horizon 근거.

frame-stacking 또는 recurrent state를 명시하지 않은 non-Markovian 상태의 MDP는 제공을 거부하세요. target outcome 기준으로 정의되지 않은 reward는 거부하세요. infinite-horizon 과제에서 `γ ≥ 1.0`이면 표시하세요. reward 범위가 일반적인 step reward의 100배를 넘으면 gradient explosion의 유력한 원인으로 표시하세요.
```

## 연습문제

1. **Easy.** `code/main.py`에 4×4 GridWorld와 random-policy rollout을 구현하세요. 10,000 episode를 실행하세요. return의 mean과 std를 보고하세요. optimal return(-6)과 비교하세요.
2. **Medium.** uniform-random policy에 대해 `γ ∈ {0.5, 0.9, 0.99}`로 `policy_evaluation`을 실행하세요. 각 결과의 `V`를 4×4 grid로 출력하세요. 왜 terminal에 가까운 state value가 더 큰 `γ`에서 더 빠르게 커지는지 설명하세요.
3. **Hard.** GridWorld를 stochastic하게 바꾸세요. 각 action이 probability `p = 0.1`로 인접한 방향으로 미끄러지게 만드세요. uniform policy를 다시 평가하세요. `V[start]`는 좋아지나요, 나빠지나요? 왜 그런가요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|--------------------------|-----------|
| MDP | "강화학습 설정" | Markov property를 만족하는 tuple `(S, A, P, R, γ)`. |
| State | "agent가 보는 것" | 선택한 policy class 아래에서 미래 dynamics에 대한 sufficient statistic. |
| Policy | "agent의 행동" | conditional distribution `π(a \| s)` 또는 deterministic map `s → a`. |
| Return | "총 보상" | 현재 step부터의 discounted sum `Σ γ^t r_t`. |
| Value | "state가 얼마나 좋은지" | `s`에서 시작해 `π` 아래에서 얻는 expected return. |
| Q-value | "action이 얼마나 좋은지" | `s`에서 시작하고 첫 action `a`를 취했을 때 `π` 아래의 expected return. |
| Bellman equation | "dynamic programming 재귀식" | value / Q를 one-step reward와 discounted successor value로 나누는 fixed-point decomposition. |
| Discount `γ` | "미래와 현재의 비중" | 먼 미래 reward에 대한 geometric weight. effective horizon은 `~1/(1-γ)`. |

## 더 읽을거리

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 표준 교과서입니다. Ch. 3은 MDP와 Bellman equation을 다루고, Ch. 1은 이후 모든 lesson의 기반이 되는 reward hypothesis를 설명합니다.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman equation의 기원입니다.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — deep-RL 관점의 간결한 MDP 입문서입니다.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — MDP와 exact solution method에 대한 operations-research reference입니다.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — MDP를 dynamic-programming specialization으로 도출하는 가장 깔끔한 설명입니다.
