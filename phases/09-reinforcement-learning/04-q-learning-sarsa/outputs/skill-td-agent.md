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
