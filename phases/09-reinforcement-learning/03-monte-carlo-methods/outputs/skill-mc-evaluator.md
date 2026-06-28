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
