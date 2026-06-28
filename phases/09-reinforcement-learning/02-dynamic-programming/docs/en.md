# Dynamic Programming — Policy Iteration과 Value Iteration

> Dynamic programming은 cheating이 가능한 RL입니다. transition과 reward function을 이미 알고 있으므로 `V` 또는 `π`가 더 이상 움직이지 않을 때까지 Bellman equation을 반복하면 됩니다. sampling-based method가 도달하려는 benchmark입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 minutes

## 문제

known model을 가진 MDP가 있습니다. 어떤 state-action pair에 대해서도 `P(s' | s, a)`와 `R(s, a, s')`를 query할 수 있습니다. 재고 관리자는 demand distribution을 압니다. board game은 deterministic transition을 가집니다. gridworld는 Python 네 줄입니다. 여러분에게는 *model*이 있습니다.

Model-free RL(Q-learning, PPO, REINFORCE)은 model이 없는 경우를 위해 만들어졌습니다. environment에서 sample만 얻을 수 있는 경우입니다. 하지만 model이 있다면 더 빠르고 더 좋은 method가 있습니다. dynamic programming입니다. Bellman은 1957년에 이것들을 설계했습니다. 이들은 여전히 correctness를 정의합니다. 사람들이 "이 MDP의 optimal policy"라고 말하면 DP가 반환할 policy를 뜻합니다.

2026년에 이것들이 필요한 이유는 세 가지입니다. 첫째, RL research의 모든 tabular environment(GridWorld, FrozenLake, CliffWalking)는 gold-standard policy를 만들기 위해 DP로 풉니다. 둘째, exact value는 sampling method를 *debug*하게 해줍니다. Q-learning의 `V*(s_0)` 추정치가 DP 정답과 30% 다르면 Q-learning에 bug가 있습니다. 셋째, modern offline RL과 planning method(MCTS, AlphaZero의 search, Phase 9 · 10의 model-based RL)는 모두 learned model 또는 given model 위에서 Bellman backup을 반복합니다.

## 개념

![Policy iteration과 value iteration 비교](../assets/dp.svg)

**두 알고리즘 모두 Bellman 위의 fixed-point iteration입니다.**

**Policy iteration.** policy가 더 이상 변하지 않을 때까지 두 step을 번갈아 수행합니다.

1. *Evaluation:* policy `π`가 주어지면 수렴할 때까지 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`를 반복 적용해 `V^π`를 계산합니다.
2. *Improvement:* `V^π`가 주어지면 `V^π`에 대해 greedy하도록 `π`를 만듭니다. `π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`.

수렴은 보장됩니다. (a) 각 improvement step은 `π`를 그대로 두거나 어떤 state의 `V^π`를 엄격히 증가시키고, (b) deterministic policy의 공간은 finite이기 때문입니다. 큰 state space에서도 보통 5~20번 정도의 outer iteration으로 수렴합니다.

**Value iteration.** evaluation과 improvement를 한 sweep으로 합칩니다. Bellman *optimality* equation을 적용합니다.

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

`max_s |V_{new}(s) - V(s)| < ε`가 될 때까지 반복합니다. 마지막에 greedy action을 취해 policy를 추출합니다. iteration당 엄격히 더 빠릅니다. inner evaluation loop가 없기 때문입니다. 하지만 보통 수렴까지 더 많은 iteration이 필요합니다.

**Generalized policy iteration (GPI).** 통합 관점입니다. value function과 policy는 양방향 improvement loop에 묶여 있습니다. 둘을 mutual consistency 쪽으로 밀어가는 모든 method(async value iteration, modified policy iteration, Q-learning, actor-critic, PPO)는 GPI의 instance입니다.

**왜 `γ < 1`이 중요한가.** Bellman operator는 sup-norm에서 `γ`-contraction입니다. `||T V - T V'||_∞ ≤ γ ||V - V'||_∞`. contraction은 unique fixed point와 geometric convergence를 의미합니다. `γ < 1`을 버리면 guarantee를 잃습니다. finite horizon 또는 absorbing terminal state가 필요합니다.

```figure
value-iteration-gamma
```

## 직접 만들기

### 1단계: GridWorld MDP model 만들기

Lesson 01의 같은 4×4 GridWorld를 사용합니다. stochastic variant를 추가합니다. probability `0.1`로 agent가 random perpendicular direction으로 미끄러집니다.

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)`는 `(s', r, p)`의 list를 반환합니다. 이것이 전체 model입니다.

### 2단계: policy evaluation

policy `π(s) = {action: prob}`가 주어지면 `V`가 멈출 때까지 Bellman equation을 반복합니다.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 3단계: policy improvement

`π`를 `V`에 대한 greedy policy로 바꿉니다. `π`가 변하지 않았다면 반환하세요. optimum에 도달한 것입니다.

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 4단계: 둘을 이어 붙이기

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4에서 typical convergence는 4~6번의 outer iteration입니다. `V*(0,0) ≈ -6`과 step count를 엄격히 줄이는 policy를 출력합니다.

### 5단계: value iteration(단일 loop 버전)

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

같은 fixed point이고, code는 더 짧습니다.

## 함정

- **Terminal 처리를 잊는 것.** absorbing state에 Bellman을 적용하면 여전히 아무것도 바꾸지 않는 "best action"을 집어 듭니다. `if s == terminal: V[s] = 0`으로 guard하세요.
- **Sup-norm vs L2 convergence.** average가 아니라 `max |V_new - V|`를 사용하세요. 이론적 guarantee는 sup-norm에 있습니다.
- **In-place vs synchronous update.** `V[s]`를 in-place로 업데이트하는 방식(Gauss-Seidel)은 별도의 `V_new` dict를 쓰는 방식(Jacobi)보다 빠르게 수렴합니다. production code는 in-place를 씁니다.
- **Policy tie.** 두 action의 Q-value가 같으면 `argmax`가 iteration마다 tie를 다르게 깨서 "policy stable" check가 oscillate할 수 있습니다. 고정된 순서에서 첫 action을 고르는 stable tie-break를 사용하세요.
- **State-space explosion.** DP는 sweep당 `O(|S| · |A|)`입니다. 약 10⁷ state까지 작동합니다. 그 이상은 function approximation이 필요합니다(Phase 9 · 05 이후).

## 활용하기

2026년에 DP는 correctness baseline이자 planner의 inner loop입니다.

| 사용 사례 | 방법 |
|----------|--------|
| 작은 tabular MDP를 정확히 풀기 | Value iteration(더 단순) 또는 policy iteration(outer step이 더 적음) |
| Q-learning / PPO implementation 검증 | toy environment에서 DP-optimal V*와 비교 |
| Model-based RL (Phase 9 · 10) | learned transition model 위의 Bellman backup |
| AlphaZero / MuZero의 planning | Monte Carlo Tree Search = async Bellman backup |
| Offline RL (CQL, IQL) | Conservative Q-iteration, 즉 OOD action에 penalty를 둔 DP |

누군가 "optimal value function"이라고 말할 때마다 "DP fixed point"를 뜻합니다. paper에서 `V*` 또는 `Q*`를 보면 이 loop를 떠올리세요.

## 결과물 만들기

`outputs/skill-dp-solver.md`로 저장하세요.

```markdown
---
name: dp-solver
description: 작은 tabular MDP를 policy iteration 또는 value iteration으로 정확히 풉니다. 수렴 동작을 보고합니다.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

known model을 가진 MDP가 주어지면 다음을 출력하세요.

1. Choice. Policy iteration vs value iteration. |S|, |A|, γ와 연결한 이유.
2. Initialization. V_0, 시작 policy. 수렴 민감도.
3. Stopping. Sup-norm tolerance ε. 예상 sweep 횟수.
4. Verification. 정확히 계산한 V*(s_0). 추출한 greedy policy.
5. Use. 이 baseline을 sampling-based method 디버그/평가에 어떻게 사용할지.

state space가 10⁷을 넘으면 DP 실행을 거부하세요. sup-norm check 없이 수렴을 주장하지 마세요. infinite-horizon 과제에서 γ ≥ 1이면 guarantee violation으로 표시하세요.
```

## 연습문제

1. **Easy.** 4×4 GridWorld에서 `γ ∈ {0.9, 0.99}`로 value iteration을 실행하세요. `max |ΔV| < 1e-6`까지 몇 sweep이 필요한가요? `V*`를 4×4 grid로 출력하세요.
2. **Medium.** *stochastic* GridWorld(slip probability `0.1`)에서 policy iteration과 value iteration을 비교하세요. sweep 수, wall-clock time, final `V*(0,0)`를 세세요. iteration 기준으로 어느 쪽이 더 빨리 수렴하나요? wall-clock 기준으로는 어떤가요?
3. **Hard.** modified policy iteration을 만드세요. evaluation step에서 수렴할 때까지가 아니라 `k` sweep만 실행합니다. `k ∈ {1, 2, 5, 10, 50}`에 대해 `V*(0,0)` error vs `k`를 plot하세요. 이 curve가 evaluation/improvement tradeoff에 대해 무엇을 말해주나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|--------------------------|-----------|
| Policy iteration | "DP 알고리즘" | policy가 멈출 때까지 evaluation(`V^π`)과 improvement(`V^π`에 대해 greedy한 `π`)를 번갈아 수행. |
| Value iteration | "더 빠른 DP" | Bellman optimality backup을 한 sweep에 적용하며, `V*`로 geometrically 수렴. |
| Bellman operator | "재귀 연산자" | `(T V)(s) = max_a Σ P (r + γ V(s'))`; sup-norm에서 `γ`-contraction. |
| Contraction | "왜 DP가 수렴하는가" | `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|`인 operator `T`는 unique fixed point를 가짐. |
| GPI | "모든 것은 DP" | Generalized Policy Iteration: `V`와 `π`를 mutual consistency로 밀어가는 모든 method. |
| Synchronous update | "Jacobi 방식" | 한 sweep 내내 old `V`를 사용. 분석은 깔끔하지만 느림. |
| In-place update | "Gauss-Seidel 방식" | 업데이트되는 중인 `V`를 바로 사용. 실제로 더 빠르게 수렴. |

## 더 읽을거리

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — policy iteration과 value iteration의 표준 설명입니다.
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook.html) — contraction-mapping argument의 엄밀한 처리입니다.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration과 convergence analysis를 다룹니다.
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — original policy iteration paper입니다.
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 이후 모든 lesson이 쓰는 DP에서 approximate-DP / deep RL로 넘어가는 bridge입니다.
