# Deep Q-Networks (DQN)

> 2013년: Mnih는 원시 픽셀 위에서 하나의 Q-learning 네트워크를 학습시켜 일곱 개 Atari 게임에서 모든 고전적 RL 에이전트를 이겼다. 2015년: 이를 49개 게임으로 확장해 Nature에 발표했고, deep RL 시대를 열었다. DQN은 Q-learning에 함수 근사를 안정화하는 세 가지 요령을 더한 것이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## 문제

표 형태 Q-learning은 모든 (상태, 행동) 쌍마다 별도의 Q-value가 필요하다. 체스판에는 약 10⁴³개의 상태가 있다. Atari 프레임 하나는 210×160×3 = 100,800개의 특징이다. 표 형태 RL은 수십억은커녕 수천 개 상태에서도 무너진다.

해결책은 지나고 보면 명백하다. Q-table을 신경망 `Q(s, a; θ)`로 바꾸면 된다. 하지만 이 명백한 해법에 도달하는 데는 수십 년이 걸렸다. Q-learning에 순진하게 함수 근사를 붙이면 "deadly triad" 아래에서 발산한다. 함수 근사 + bootstrapping + off-policy 학습의 조합이다. Mnih et al. (2013, 2015)은 학습을 안정화하는 세 가지 엔지니어링 요령을 찾아냈다.

1. **Experience replay**는 전이 사이의 상관을 끊는다.
2. **Target network**는 bootstrap target을 고정한다.
3. **Reward clipping**은 gradient 크기를 정규화한다.

Atari에서의 DQN은 단일 아키텍처와 단일 하이퍼파라미터 집합으로 원시 픽셀에서 수십 개 제어 문제를 해결한 첫 사례였다. 이후 만들어진 모든 "deep-RL" 계열, 즉 DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57은 이 세 가지 요령 위에 쌓여 있다.

## 개념

![DQN 학습 loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**목표.** DQN은 신경망 Q-function에서 one-step TD loss를 최소화한다.

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network로, 매 스텝 gradient descent로 갱신된다. `θ^-` = target network로, 주기적으로 `θ`에서 복사된다(대략 10,000 스텝마다). `D` = 과거 전이들의 replay buffer다.

**세 가지 요령, 중요도 순:**

**Experience replay.** `~10⁶`개 전이를 담는 ring buffer다. 각 학습 스텝은 minibatch를 균등 무작위로 샘플링한다. 이렇게 하면 시간적 상관(연속 프레임은 거의 동일하다)이 깨지고, 드문 보상 전이에서 여러 번 학습할 수 있으며, 연속 gradient update 사이의 상관도 줄어든다. 이것이 없으면 신경망을 쓰는 on-policy TD는 Atari에서 발산한다.

**Target network.** Bellman 방정식 양쪽에 같은 네트워크 `Q(·; θ)`를 쓰면 target이 매 update마다 움직인다. 자기 꼬리를 쫓는 셈이다. 해결책은 고정된 가중치를 가진 두 번째 네트워크 `Q(·; θ^-)`를 유지하는 것이다. `C` 스텝마다 `θ → θ^-`를 복사한다. 이렇게 하면 수천 번의 gradient step 동안 regression target이 안정된다. Soft update `θ^- ← τ θ + (1-τ) θ^-`(DDPG, SAC에서 사용)는 더 부드러운 변형이다.

**Reward clipping.** Atari의 보상 크기는 1부터 1000+까지 다양하다. `{-1, 0, +1}`로 clipping하면 어떤 단일 게임도 gradient를 지배하지 못한다. 보상 크기 자체가 중요할 때는 틀린 선택이지만, 부호만 중요한 Atari에서는 괜찮다.

**Double DQN.** Hasselt (2016)는 maximization bias를 고친다. online net으로 행동을 *선택*하고, target net으로 그 행동을 *평가*한다.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

그대로 끼워 넣을 수 있는 대체식이며 일관되게 더 낫다. 기본값으로 사용하라.

**다른 개선(Rainbow, 2017):** prioritized replay(TD-error가 큰 전이를 더 자주 샘플링), dueling architecture(`V(s)`와 advantage head 분리), noisy networks(학습된 exploration), n-step returns, distributional Q(C51/QR-DQN), multi-step bootstrapping. 각각은 몇 퍼센트씩 보태고, 이득은 대체로 더해진다.

## 직접 만들기

여기 코드에는 stdlib만 쓰고 numpy도 쓰지 않는다. 아주 작은 연속형 GridWorld 위에 손으로 만든 단일 hidden layer MLP를 사용하므로 모든 학습 스텝이 마이크로초 단위로 돈다. 알고리즘은 규모만 키우면 Atari DQN과 동일하다.

### 1단계: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari에는 약 50,000 용량이 필요하고, 우리 toy env에는 5,000이면 충분하다.

### 2단계: 작은 Q-network(수동 MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass는 linear → ReLU → linear다. 이것이 전체 네트워크다.

### 3단계: DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

형태는 Lesson 04의 Q-learning과 같고 차이는 두 가지뿐이다. (a) table을 indexing하는 대신 미분 가능한 `Q(·; θ)`를 backprop한다. (b) target은 `Q(·; θ^-)`를 사용한다.

### 4단계: 바깥 loop

각 episode마다 `Q(·; θ)`에 대해 ε-greedy로 행동하고, 전이를 buffer에 넣고, minibatch를 샘플링해 gradient step을 수행하며, 주기적으로 `θ^- ← θ`를 동기화한다. 패턴은 다음과 같다.

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

16차원 one-hot 상태를 쓰는 작은 GridWorld에서는 agent가 약 500 episode 안에 거의 최적인 policy를 배운다. Atari에서는 이를 200M frame으로 확장하고 CNN feature extractor를 추가한다.

## 함정

- **Deadly triad.** 함수 근사 + off-policy + bootstrapping은 발산할 수 있다. DQN은 target net + replay로 이를 완화한다. 둘 중 하나라도 제거하지 말라.
- **Exploration.** ε는 보통 학습 초기 약 10% 동안 1.0에서 0.01로 decay해야 한다. 초기 exploration이 부족하면 Q-net은 국소 basin에 수렴한다.
- **Overestimation.** noisy Q에 대한 `max`는 위쪽으로 bias된다. production에서는 항상 Double DQN을 사용하라.
- **Reward scale.** 보상을 clip하거나 normalize하라. gradient 크기는 보상 크기에 비례한다.
- **Replay buffer coldstart.** buffer에 수천 개 전이가 쌓이기 전에는 학습하지 말라. 약 20개 샘플에 대한 초기 gradient는 overfit된다.
- **Target sync frequency.** 너무 자주 동기화하면 target net이 없는 것과 비슷하고, 너무 드물면 target이 낡는다. Atari DQN은 10,000 env step을 쓴다. 경험칙: 전체 학습 horizon의 약 1/100마다 sync한다.
- **Observation preprocessing.** Atari DQN은 상태가 Markov가 되도록 4 frame을 stack한다. 속도 정보가 필요한 env는 frame stacking이나 recurrent state가 필요하다.

## 활용하기

2026년에 DQN은 거의 state-of-the-art가 아니지만 여전히 기준이 되는 off-policy 알고리즘이다.

| 작업 | 선택할 방법 | DQN이 아닌 이유 |
|------|-------------|-----------------|
| 이산 행동 Atari 계열 | Rainbow DQN 또는 Muesli | 같은 framework에 더 많은 요령이 있다. |
| 연속 제어 | SAC / TD3 (Phase 9 · 07) | DQN에는 policy network가 없다. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | replay buffer가 없고 scale하기 쉽다. |
| Offline RL | CQL / IQL / Decision Transformer | 보수적인 Q target, bootstrapping 폭주 없음. |
| 큰 discrete action space(recommender) | action embedding을 곁들인 DQN, 또는 IMPALA | 가능하다. 세부 설계가 중요하다. |
| LLM RL | PPO / GRPO | step-level이 아니라 sequence-level이고 loss가 다르다. |

교훈은 여전히 이동 가능하다. Replay와 target network는 SAC, TD3, DDPG, SAC-X, AlphaZero의 self-play buffer, 모든 offline RL 방법에 등장한다. Reward clipping은 PPO의 advantage normalization으로 살아남았다. 이 아키텍처가 청사진이다.

## 산출물

`outputs/skill-dqn-trainer.md`로 저장하라.

```markdown
---
name: dqn-trainer
description: 이산 행동 RL 작업을 위한 DQN 학습 config(buffer, target sync, ε schedule, reward clipping)를 만든다.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

이산 행동 환경(observation shape, action count, horizon, reward scale)이 주어지면 다음을 출력하라.

1. 네트워크. Architecture(MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy(C step마다 hard sync 또는 soft τ).
4. 탐색. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. 비활성화할 명시적 이유가 없으면 기본으로 켠다.

target network나 replay buffer가 없거나 ε가 1로 고정된 DQN은 ship하지 말라. continuous-action 작업은 거부하라(SAC / TD3로 보낸다). reward range가 per-step mean의 10×를 넘으면 clipping 또는 scale normalization이 필요하다고 표시하라.
```

## 연습문제

1. **쉬움.** `code/main.py`를 실행하라. episode별 return curve를 그려라. running mean이 -10을 넘기까지 몇 episode가 필요한가?
2. **보통.** target network를 비활성화하라(Bellman target 양쪽에 online net 사용). 학습 불안정성을 측정하라. return이 진동하거나 발산하는가?
3. **어려움.** Double DQN을 추가하라. online net으로 `argmax a'`를 고르고, target net으로 평가하라. noisy-reward GridWorld에서 1,000 episode 후 Double DQN이 있을 때와 없을 때 `Q(s_0, best_a)`의 bias를 true `V*(s_0)`와 비교하라.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|------------------|-----------|
| DQN | "Deep Q-learning" | 신경망 Q-function, replay buffer, target network를 쓰는 Q-learning. |
| Experience replay | "섞은 전이들" | 각 gradient step에서 균등 샘플링하는 ring buffer. 데이터 상관을 끊는다. |
| Target network | "고정된 bootstrap" | Bellman target에 쓰는 Q의 주기적 복사본. 학습을 안정화한다. |
| Deadly triad | "RL이 발산하는 이유" | 함수 근사 + bootstrapping + off-policy = 수렴 보장이 없음. |
| Double DQN | "maximization bias 수정" | online net이 행동을 선택하고 target net이 평가한다. |
| Dueling DQN | "V와 A head" | Q = V + A - mean(A)로 분해한다. 출력은 같고 gradient flow는 더 좋다. |
| Rainbow | "모든 요령" | DDQN + PER + dueling + n-step + noisy + distributional을 하나로 묶은 것. |
| PER | "우선순위 replay" | TD-error 크기에 비례해 전이를 샘플링한다. |

## 더 읽을거리

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — deep RL의 출발점이 된 2013 NeurIPS workshop 논문.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 논문, 49-game DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 여러 요령을 쌓은 논문.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — 명확한 현대적 설명.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — DQN의 target network와 replay buffer가 다루려는 "deadly triad"(함수 근사 + bootstrapping + off-policy)에 대한 교과서 설명.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — ablation study에 쓰이는 reference single-file DQN. 이 lesson의 from-scratch 버전과 함께 읽기 좋다.
