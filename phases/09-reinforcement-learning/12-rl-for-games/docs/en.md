# 게임을 위한 RL — AlphaZero, MuZero, 그리고 LLM-Reasoning 시대

> 1992: TD-Gammon은 pure TD로 backgammon 인간 champion을 이겼습니다. 2016: AlphaGo는 이세돌을 이겼습니다. 2017: AlphaZero는 chess, shogi, Go를 처음부터 학습해 압도했습니다. 2024: DeepSeek-R1은 PPO를 GRPO로 바꾼 같은 recipe가 reasoning에도 작동함을 증명했습니다. game은 이 phase의 모든 breakthrough를 밀어붙이는 benchmark입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## 문제

game에는 RL이 원하는 모든 것이 있습니다. 깨끗한 reward(win/loss). 무한한 episode(self-play reset). 완벽한 simulation(game *자체*가 simulator입니다). discrete 또는 작은 continuous action space. adversarial robustness를 강제하는 multi-agent structure.

그리고 game은 모든 주요 RL breakthrough가 시험된 방식입니다. TD-Gammon(backgammon, 1992). Atari-DQN(2013). AlphaGo(2016). AlphaZero(2017). OpenAI Five(Dota 2, 2019). AlphaStar(StarCraft II, 2019). MuZero(learned model, 2019). AlphaTensor(matrix multiplication, 2022). AlphaDev(sorting algorithms, 2023). DeepSeek-R1(math reasoning, 2025) — game-RL technique이 text에서도 작동한다는 가장 최근의 demonstration입니다.

이 capstone은 세 landmark architecture인 AlphaZero, MuZero, GRPO를 하나의 통합 관점으로 살펴봅니다. **self-play + search + policy improvement**입니다. 각 방법은 이전 방법을 일반화합니다. 특히 GRPO는 token을 action으로, mathematical verification을 win signal로 삼아 AlphaZero의 recipe를 LLM reasoning에 적용한 것입니다.

## 개념

![AlphaZero ↔ MuZero ↔ GRPO: 같은 loop, 다른 environment](../assets/rl-games.svg)

**통합 loop.**

```text
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. 알려진 규칙을 가진 game(chess, shogi, Go)이 주어졌을 때:

- Policy-value network: 하나의 tower `f_θ(s) → (p, v)`. `p`는 legal move 위의 prior입니다. `v`는 expected game outcome입니다.
- Monte Carlo Tree Search (MCTS): 각 move에서 가능한 continuation tree를 확장합니다. `(p, v)`를 prior + bootstrap으로 사용합니다. UCB(PUCT)로 node를 선택합니다. `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: agent-vs-agent로 game을 플레이합니다. move `t`에서 MCTS visit distribution `π_t`가 policy training target이 됩니다.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z`는 game outcome(+1 / 0 / -1)입니다.

인간 지식 0. handcrafted heuristic 0. self-play game을 각각 몇천만 번 수행한 뒤 chess, shogi, Go를 모두 master한 단일 recipe입니다.

**MuZero (2019).** Schrittwieser et al. 규칙을 알아야 한다는 요구를 제거합니다.

- 고정 environment 대신 *latent dynamics model* `(h, g, f)`를 학습합니다.
  - `h(s)`: observation을 latent state로 encode합니다.
  - `g(s_latent, a)`: next latent state + reward를 예측합니다.
  - `f(s_latent)`: policy prior + value를 예측합니다.
- MCTS는 *learned latent space*에서 실행됩니다. 같은 search, 같은 training loop입니다.
- Go, chess, shogi *그리고* Atari에서 작동합니다. 규칙 지식이 없는 하나의 algorithm입니다.

**Stochastic MuZero (2022).** stochastic dynamics와 chance node를 추가해 backgammon류 game으로 확장합니다.

**Muesli, Gumbel MuZero (2022-2024).** sample efficiency와 deterministic search를 개선합니다.

**GRPO (2024-2025).** DeepSeek-R1 recipe입니다. 같은 AlphaZero 모양의 loop를 language-model reasoning에 적용합니다.

- "Game": math / coding / reasoning problem에 답하기. "Win" = verifier(test case passes, numerical answer matches)가 1을 반환.
- Policy: LLM. Action: token. State: prompt + response-so-far.
- critic(PPO-style V_φ)이 없습니다. 대신 각 prompt마다 policy에서 `G`개의 completion을 sample합니다. 각각의 reward를 계산합니다. **group-relative advantage** `A_i = (r_i - mean_r) / std_r`를 REINFORCE-style update의 signal로 사용합니다.
- drift를 막기 위한 reference policy 대비 KL penalty가 있습니다(RLHF와 동일).
- 전체 loss:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Reward model도 critic도 MCTS도 없습니다. group-relative baseline이 세 가지를 모두 대체합니다. reasoning benchmark에서 훨씬 적은 compute로 PPO-RLHF 품질과 같거나 더 좋은 결과를 냅니다.

**R1 recipe 전체.** DeepSeek-R1(DeepSeek 2025)은 한 논문 안의 두 모델입니다.

- **R1-Zero.** DeepSeek-V3 base model에서 시작합니다. SFT가 없습니다. 두 reward component로 GRPO를 바로 적용합니다. *accuracy reward*(규칙 기반 — final answer가 올바른 숫자로 parse되는가 / code가 unit test를 통과하는가)와 *format reward*(completion이 chain-of-thought를 `<think>…</think>` tag로 감쌌는가)입니다. 수천 step 동안 평균 response length는 약 100 token에서 약 10,000 token으로 늘고, math benchmark score는 near-o1-preview level까지 올라갑니다. model이 scratch에서 reasoning을 배웁니다. 단점: chain of thought가 종종 읽기 어렵고, 언어가 섞이며, 스타일이 다듬어져 있지 않습니다.
- **R1.** R1-Zero의 readability 문제를 네 단계 pipeline으로 해결합니다.
  1. **Cold-start SFT.** 깨끗한 formatting의 long-CoT demonstration 수천 개를 모읍니다. base model을 그것들로 supervised-finetune합니다. 읽을 수 있는 starting point를 제공합니다.
  2. **Reasoning-oriented GRPO.** accuracy+format reward에 code-switching을 막는 *language-consistency* reward를 더해 GRPO를 적용합니다.
  3. **Rejection sampling + SFT round 2.** RL checkpoint에서 약 600K reasoning trajectory를 sample하고, final answer가 correct이며 CoT가 읽을 수 있는 것만 남깁니다. 여기에 약 200K non-reasoning SFT example(writing, QA, self-cognition)을 합쳐 base를 다시 fine-tune합니다.
  4. **Full-spectrum GRPO.** reasoning(rule-based rewards)과 general alignment(helpfulness/harmlessness preference-based rewards)를 모두 다루는 RL round를 한 번 더 수행합니다.

결과는 open weights에서 AIME와 MATH-500에서 o1과 맞먹고, distill할 수 있을 만큼 작습니다. 같은 논문은 R1의 reasoning trace로 SFT해 만든 여섯 개의 distilled dense model(Qwen-1.5B부터 Llama-70B까지)도 공개합니다. student에는 RL을 적용하지 않습니다. 강한 RL teacher의 distillation은 student scale에서 scratch부터 RL하는 것보다 일관되게 낫습니다.

**reasoning에 PPO 대신 GRPO를 쓰는 이유.** DeepSeekMath 논문(2024년 2월)의 세 가지 이유는 다음과 같습니다. (1) value network를 학습하지 않아 memory가 절반으로 줄어듭니다. (2) group baseline은 reasoning task가 만드는 sparse end-of-trajectory reward를 자연스럽게 처리합니다. (3) per-prompt normalization은 난이도가 크게 다른 problem 간에도 advantage를 비교 가능하게 만듭니다. PPO의 단일 critic은 이것을 잘 처리하지 못합니다.

**Search-free vs search-based.** game은 갈라졌습니다.

- *Perfect-information games with long horizons* (Go, chess): 여전히 search-based입니다. AlphaZero / MuZero가 지배합니다.
- *LLM reasoning*: production에는 아직 MCTS가 없습니다. full rollout에 GRPO를 적용하고, inference compute에는 best-of-N을 씁니다. Process reward models(PRMs)는 step-level search가 다시 추가될 가능성을 보여 줍니다.

## 직접 만들기

`code/main.py`의 코드는 **GRPO in miniature**를 구현합니다. 여러 sample group이 있는 bandit입니다. algorithm은 LLM에서 쓰는 것과 같습니다. policy와 environment만 더 단순할 뿐입니다. 2025년 innovation인 *loss*와 *group-relative advantage*를 가르칩니다.

### 1단계: tiny verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

실제 GRPO에서는 verifier가 unit test를 실행하거나 math equality를 검사합니다.

### 2단계: policy: prompt마다 K answer token 위의 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

prompt에 조건을 건 LLM의 final-layer output과 같습니다.

### 3단계: group sampling과 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

group-relative advantage가 2024년 DeepSeek trick입니다. critic은 필요 없습니다. "baseline"은 group mean이고, normalization은 group std를 사용합니다.

### 4단계: REINFORCE baseline(value-free)과 비교

같은 setup, 같은 compute, plain REINFORCE입니다. GRPO가 더 빠르고 안정적으로 수렴합니다.

### 5단계: entropy와 KL 관찰

RLHF와 같은 진단 지표입니다. reference 대비 평균 KL, policy entropy, reward-over-time. 이것들이 안정화되면 학습이 끝난 것입니다.

## 함정

- **Reward hacking via verifier gaming.** GRPO는 RLHF의 위험을 물려받습니다. verifier가 틀렸거나 exploit 가능하면 LLM은 exploit을 찾아냅니다. robust verifier(여러 test case, formal proof)가 중요합니다.
- **Group size too small.** group baseline의 variance는 `1/√G`처럼 움직입니다. `G = 4` 아래에서는 advantage signal이 noisy합니다. 표준 선택은 `G = 8`에서 `64`입니다.
- **Length bias.** 길이가 다른 LLM completion은 log-probability도 다릅니다. token count로 normalize하거나, sequence-level log-prob을 쓰거나, max length로 truncate하세요.
- **Pure self-play cycles.** AlphaZero-style training은 general-sum game에서 dominance loop에 갇힐 수 있습니다. 다양한 opponent pool(league play, Lesson 10)로 완화합니다.
- **Search-policy mismatch.** AlphaZero는 policy를 search output에 맞추도록 학습합니다. policy net이 search distribution을 표현하기에 너무 작으면 training이 정체합니다.
- **Compute floor.** MuZero / AlphaZero는 막대한 compute가 필요합니다. 단일 ablation도 수백 GPU-hour인 경우가 많습니다. 학습용 miniature demo(예: Connect Four 위의 AlphaZero)가 있습니다.
- **Verifier coverage.** buggy solution을 통과시키는 unit test는 그 bug를 강화합니다. edge case를 잡는 verifier를 설계하세요.

## 활용하기

2026년 game-RL landscape, domain별:

| 도메인 | 지배적 방법 |
|--------|-----------------|
| 2인 zero-sum board game(Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| imperfect-info card game(poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel game | Muesli / MuZero / IMPALA-PPO |
| 대규모 multiplayer strategy(Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (GRPO 아님; verifier가 preference이지 verifiable이 아님) |
| Robotics | PPO + DR (game-RL은 아니지만 같은 policy-gradient tool을 사용) |
| 조합 최적화 문제 | AlphaZero variants (AlphaTensor, AlphaDev) |

*recipe*, 즉 self-play, search-augmented improvement, policy distillation은 text, pixel, physical control에 걸쳐 있습니다. GRPO는 가장 젊은 instance입니다. 더 많은 instance가 나올 것입니다.

## 출시하기

`outputs/skill-game-rl-designer.md`로 저장하세요.

```markdown
---
name: game-rl-designer
description: 주어진 도메인을 위한 game-RL 또는 reasoning-RL 학습 파이프라인(AlphaZero / MuZero / GRPO)을 설계합니다.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

목표(perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial)가 주어지면 다음을 출력하세요.

1. 환경 적합성. 알려진 규칙인가? Markov인가? stochastic인가? multi-agent인가? AlphaZero vs MuZero vs GRPO 선택에 반영합니다.
2. search 전략. MCTS(learned prior를 쓰는 PUCT), Gumbel-sampled, best-of-N, 또는 없음.
3. self-play 계획. symmetric self-play / league / offline data / verifier-generated.
4. target signal. game outcome / verifier reward / preference / learned model. robustness 계획을 포함합니다.
5. 진단 지표. baseline 대비 승률, ELO curve, verifier pass rate, reference 대비 KL.

imperfect-info game에는 AlphaZero를 거부하세요(CFR로 보냅니다). 신뢰할 수 있는 verifier 없이 GRPO를 거부하세요. 고정 baseline opponent 세트가 없는 game-RL 파이프라인은 모두 거부하세요(self-play ELO는 그렇지 않으면 보정되지 않습니다).
```

## 연습 문제

1. **Easy.** `code/main.py`에서 GRPO bandit을 구현하세요. prompt 2개 × answer token 4개에서 학습하세요. `G=8`로 1,000 update 안에 수렴해야 합니다.
2. **Medium.** PPO(clipped)와 vanilla REINFORCE를 끼워 넣으세요. 같은 bandit에서 GRPO와 sample efficiency 및 reward variance를 비교하세요.
3. **Hard.** 길이 2의 "reasoning chain"으로 확장하세요. agent는 token 두 개를 내고 verifier는 그 pair에 reward를 줍니다. GRPO가 two-step sequence 전반의 credit assignment를 어떻게 처리하는지 측정하세요. (Hint: *full sequence*마다 group advantage를 계산하고, 두 token position 모두에 전파하세요.)

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| MCTS | "learned net을 쓰는 tree search" | Monte Carlo Tree Search; learned `(p, v)` prior를 쓰는 UCB1/PUCT selection. |
| AlphaZero | "self-play + MCTS" | MCTS visit과 game outcome에 맞춰 학습한 policy-value net. |
| MuZero | "learned-model AlphaZero" | 같은 loop이지만 learned dynamics를 통해 latent space에서 작동합니다. |
| GRPO | "critic-free PPO" | Group Relative Policy Optimization; group-mean baseline + KL을 쓰는 REINFORCE. |
| PUCT | "AlphaZero의 UCB" | `Q + c · p · √N / (1 + N_a)` — value estimate와 prior의 균형을 맞춥니다. |
| Self-play | "agent vs past self" | zero-sum의 표준입니다. 대칭적 training signal입니다. |
| League play | "population-based self-play" | past + current + exploiter를 opponent로 sampling합니다. |
| Verifier reward | "검증 가능한 RL" | deterministic checker(test 통과, answer match)에서 reward가 나옵니다. |
| Process reward | "PRM" | final answer뿐 아니라 각 reasoning step에 점수를 매깁니다. |

## 더 읽을거리

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — GRPO와 group-relative baseline을 도입한 논문입니다.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 전체 네 단계 R1 recipe와 R1-Zero ablation.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — scale 있는 CFR + deep-learning.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 모든 것을 시작한 논문입니다.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — custom reward function으로 GRPO를 적용하는 production reference입니다.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 여러 scale에서 R1 recipe를 open replication한 사례입니다.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — R1이 LLM scale에서 구현한 self-play, search, "designed reward"에 대한 textbook framing입니다.
