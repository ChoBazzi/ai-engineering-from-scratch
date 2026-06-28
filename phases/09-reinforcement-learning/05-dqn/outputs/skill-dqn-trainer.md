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
