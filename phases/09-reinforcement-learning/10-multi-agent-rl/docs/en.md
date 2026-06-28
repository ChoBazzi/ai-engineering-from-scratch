# Multi-Agent RL

> Single-agent RL은 environment가 stationary라고 가정합니다. 같은 세계에 학습하는 agent 두 개를 넣으면 이 가정은 깨집니다. 각 agent가 상대 agent의 environment 일부가 되고, 둘 다 변하고 있기 때문입니다. Multi-agent RL은 Markov assumption이 더 이상 성립하지 않을 때도 learning이 수렴하게 만드는 기법들의 집합입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## 문제

방 안을 이동하는 법을 배우는 robot은 single-agent RL 문제입니다. 축구팀은 그렇지 않습니다. StarCraft opponent를 상대하는 AlphaStar도 그렇지 않습니다. bidding agent들의 marketplace도 그렇지 않습니다. 사거리 정지 표지판에서 협상하는 두 대의 차도 그렇지 않습니다. 현실의 many-on-many 문제 대부분은 single-agent가 아닙니다.

모든 multi-agent setting에서, 한 agent의 관점에서는 다른 agent들이 *environment의 일부*입니다. 그들이 학습하고 행동을 바꾸면 environment는 non-stationary가 됩니다. Markov property, 즉 "next state는 current state와 내 action에만 의존한다"는 성질이 위반됩니다. next state는 *다른* agent가 무엇을 선택했는지에도 의존하고, 그들의 policy는 moving target이기 때문입니다.

이것은 tabular convergence proof를 깨뜨립니다(Q-learning의 보장은 stationary environment를 가정합니다). naive deep RL도 깨집니다. agent들이 서로를 쫓는 loop에 빠져 stable policy로 수렴하지 못합니다. multi-agent 전용 기법이 필요합니다. centralized training / decentralized execution, counterfactual baseline, league play, self-play가 그 예입니다.

2026년의 application: robot swarm, traffic routing, autonomous vehicle fleet, market simulator, multi-agent LLM system(Phase 16), 그리고 intelligent player가 둘 이상 있는 모든 game.

## 개념

![네 가지 MARL 체계: independent, centralized critic, self-play, league](../assets/marl.svg)

**형식화: Markov Game.** MDP의 일반화입니다. state `S`, joint action `a = (a_1, …, a_n)`, transition `P(s' | s, a)`, agent별 reward `R_i(s, a, s')`로 구성됩니다. 각 agent `i`는 자기 policy `π_i` 아래에서 자기 return을 최대화합니다. reward가 동일하면 **fully cooperative**입니다. zero-sum이면 **adversarial**입니다. 섞여 있으면 **general-sum**입니다.

**핵심 난제:**

- **Non-stationarity.** agent `i`의 관점에서 `P(s' | s, a_i)`는 변화 중인 `π_{-i}`에 의존합니다.
- **Credit assignment.** shared reward가 있을 때 어떤 agent가 원인을 제공했나요?
- **Exploration coordination.** agent들은 같은 state를 중복 탐색하지 말고 상호 보완적인 strategy를 탐색해야 합니다.
- **Scalability.** joint action space는 `n`에 대해 지수적으로 커집니다.
- **Partial observability.** 각 agent는 자기 observation만 봅니다. global state는 숨겨져 있습니다.

**네 가지 지배적 체계:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** 각 agent는 다른 agent를 environment의 일부로 취급하며 자기 Q 또는 policy를 학습합니다. 단순하고, 때로는 동작합니다(특히 experience replay가 agent-modeling을 부드럽게 하는 trick처럼 작동할 때). 이론적 수렴성: 없음. 실무에서는 loosely-coupled task에는 괜찮고, tightly-coupled task에는 좋지 않습니다.

**2. Centralized training, decentralized execution (CTDE).** 현대에서 가장 흔한 paradigm입니다. 각 agent는 local observation `o_i`에 조건을 거는 자기 *policy* `π_i`를 가집니다. deployment에서는 표준적인 decentralized execution입니다. *training* 중에는 centralized critic `Q(s, a_1, …, a_n)`이 full global state와 joint action에 조건을 겁니다. 예:
- **MADDPG** (Lowe et al. 2017): agent별 centralized critic을 쓰는 DDPG.
- **COMA** (Foerster et al. 2017): counterfactual baseline입니다. "내가 action `a'`를 택했다면 reward가 얼마였을까?"라고 묻고 내 contribution을 분리합니다.
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022): centralized value function을 쓰는 PPO입니다. 2026년 cooperative MARL에서 지배적입니다.
- **QMIX** (Rashid et al. 2018): value decomposition입니다. `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))`이며 monotonic mixing을 사용합니다.

**3. Self-play.** 같은 agent의 두 copy가 서로 플레이합니다. opponent policy는 *과거 snapshot의 내 policy*입니다. AlphaGo / AlphaZero / MuZero. OpenAI Five. zero-sum game에서 가장 잘 작동하며, training signal이 대칭적입니다.

**4. League play.** general-sum / adversarial environment로 self-play를 확장한 것입니다. 과거와 현재 policy population을 유지하고, league에서 opponent를 sampling해 그들을 상대로 학습합니다. exploiter(현재 최고 policy를 이기는 데 특화)와 main exploiter(exploiter를 이기는 데 특화)를 추가합니다. AlphaStar(StarCraft II). game이 "rock-paper-scissors" strategy cycle을 허용할 때 필요합니다.

**Communication.** agent들이 learned message `m_i`를 서로에게 보낼 수 있게 합니다. cooperative setting에서 작동합니다. Foerster et al. (2016)은 differentiable inter-agent communication을 end-to-end로 학습할 수 있음을 보였습니다. 오늘날의 LLM 기반 multi-agent system(Phase 16)은 본질적으로 자연어로 communication합니다.

## 직접 만들기

이 레슨은 두 cooperative agent가 있는 6×6 GridWorld를 사용합니다. 두 agent는 반대쪽 corner에서 시작해 shared goal에 도달해야 합니다. Shared reward: 둘 중 하나라도 아직 움직이는 동안 step당 `-1`, 둘 다 도착하면 `+10`. `code/main.py`를 보세요.

### 1단계: multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*joint* action space는 `|A|² = 16`입니다. global state는 두 위치입니다.

### 2단계: independent Q-learning

각 agent는 joint state를 key로 하는 자기 Q-table을 실행합니다. 각 step에서 둘 다 ε-greedy action을 고르고, joint transition을 수집하고, shared reward로 각자 Q를 update합니다.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

이 task에서는 reward가 dense하고 aligned이므로 동작합니다. tightly-coupled task, 예를 들어 한 agent가 다른 agent를 위해 *기다려야* 하는 경우에는 실패합니다.

### 3단계: decomposed-value update를 쓰는 centralized Q

joint action `Q(s, a_1, a_2)` 위의 Q 하나를 사용합니다. shared reward로 update합니다. execution에서는 marginalization으로 decentralize합니다. `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. exponential joint action space를 대가로 *올바른* global view를 얻습니다.

### 4단계: simple self-play (adversarial 2-agent)

같은 agent, 두 역할입니다. agent A를 agent B 상대로 학습합니다. `K` episode 뒤에 A의 weight를 B에 copy합니다. 대칭적 학습, 일관된 progress. AlphaZero recipe의 축소판입니다.

## 함정

- **Non-stationary replay.** independent agent에서 experience replay는 single-agent보다 더 나쁩니다. 오래된 transition이 지금은 obsolete한 opponent에서 생성되었기 때문입니다. 해결: recency로 relabel하거나 weighting합니다.
- **Credit assignment ambiguity.** 긴 episode 뒤의 shared reward에는 어떤 agent가 기여했는지 말할 명확한 방법이 없습니다. 해결: counterfactual baseline(COMA), 또는 agent별 reward shaping.
- **Policy drift / chasing.** 각 agent의 best response가 서로의 update에 따라 변합니다. 해결: centralized critic, 낮은 learning rate, 또는 한 번에 하나씩 freeze.
- **Reward hacking via coordination.** agent들이 designer가 예상하지 못한 coordinated exploit을 찾아냅니다. auction agent가 bid zero로 수렴하는 식입니다. 해결: 신중한 reward design, behavioral constraint.
- **Exploration redundancy.** 두 agent가 같은 state-action pair를 탐색합니다. 해결: agent별 entropy bonus, 또는 role-conditioning.
- **League cycles.** pure self-play는 dominance cycle에 갇힐 수 있습니다. 해결: 다양한 opponent가 있는 league play.
- **Sample explosion.** `n` agents × state space × joint actions. function approximation과 factored action space(agent마다 하나의 policy output head)로 근사합니다.

## 활용하기

2026년 MARL application map:

| 도메인 | 방법 | 비고 |
|--------|--------|-------|
| 협력 navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| 2인 게임(chess, Go, poker) | MCTS를 쓰는 self-play(AlphaZero) | Zero-sum; symmetric training. |
| 복잡한 multiplayer(Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| 자율주행 fleet | attention을 쓰는 CTDE MAPPO / PPO | Partial obs; variable team sizes. |
| 경매 시장 | Game-theoretic equilibrium + RL | `n` → ∞일 때 mean-field RL. |
| LLM multi-agent systems (Phase 16) | 자연어 communication + role conditioning | agent-planning layer에서의 RL loop. |

2026년에 MARL의 가장 큰 성장 영역은 LLM 기반입니다. language-model agent의 swarm이 협상하고, 토론하고, software를 만듭니다. RL은 token-level이 아니라 *trajectory-level* output 위의 preference optimization으로 나타납니다(Phase 16 · 03).

## 출시하기

`outputs/skill-marl-architect.md`로 저장하세요.

```markdown
---
name: marl-architect
description: 주어진 과제에 맞는 multi-agent RL 체계(IPPO, CTDE, self-play, league)를 선택합니다.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

`n`개의 agent가 있는 과제가 주어지면 다음을 출력하세요.

1. 체계 분류. 협력 / 적대 / general-sum. 근거를 제시합니다.
2. 알고리즘. IPPO / MAPPO / QMIX / self-play / league. 결합 강도와 reward 구조에 연결해 이유를 설명합니다.
3. 정보 접근. centralized training(어떤 전역 정보가 critic에 들어가는가)? decentralized execution?
4. credit assignment. counterfactual baseline, value decomposition, 또는 reward shaping.
5. exploration 계획. agent별 entropy, population-based training, 또는 league.

강하게 결합된 협력 과제에는 independent Q-learning을 거부하세요. cycle 위험이 있는 general-sum 과제에는 self-play 추천을 거부하세요. 고정 opponent 평가가 없는 MARL 파이프라인은 표시하세요(self-play 숫자를 유리하게 골라 제시하는 일이 흔합니다).
```

## 연습 문제

1. **Easy.** 2-agent cooperative GridWorld에서 independent Q-learning을 학습하세요. mean return > 0이 되기까지 episode가 몇 개 필요한가요? joint learning curve를 그리세요.
2. **Medium.** "coordination" task를 추가하세요. 두 agent가 같은 turn에 goal 위로 올라설 때만 goal에 도달한 것으로 간주합니다. independent Q는 여전히 수렴하나요? 무엇이 깨지나요?
3. **Hard.** MAPPO-style training을 위한 centralized critic을 구현하고, coordination task에서 independent PPO와 수렴 속도를 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Markov game | "multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; 각 agent는 자기 reward를 가집니다. |
| CTDE | "centralized training, decentralized execution" | training time에는 joint critic, 각 agent policy는 local obs만 사용합니다. |
| IPPO | "independent PPO" | 각 agent가 PPO를 따로 실행합니다. 단순한 baseline이지만 종종 과소평가됩니다. |
| MAPPO | "multi-agent PPO" | global state에 조건을 건 centralized value function을 쓰는 PPO. |
| QMIX | "monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)`이 decentralized argmax를 가능하게 합니다. |
| COMA | "counterfactual multi-agent" | Advantage = 내 Q에서 내 action에 대해 marginalize한 expected Q를 뺀 값. |
| Self-play | "agent vs past self" | 단일 agent, 두 역할. zero-sum game의 표준입니다. |
| League play | "population training" | 과거 policy를 cache하고 pool에서 opponent를 sampling합니다. strategy cycle을 처리합니다. |

## 더 읽을거리

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — centralized critic을 쓰는 CTDE.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — credit assignment를 위한 counterfactual baseline.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — monotonicity를 쓰는 value decomposition.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO는 MARL에서 놀랄 만큼 강합니다.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — scale 있는 league play.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — zero-sum game에서의 pure self-play.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — multi-agent setting과 CTDE가 해결하려는 non-stationarity 문제에 대한 textbook의 짧은 설명을 포함합니다.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — cooperative, competitive, mixed MARL과 convergence result를 다루는 survey입니다.
