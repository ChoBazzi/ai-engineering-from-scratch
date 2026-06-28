---
name: dp-solver
description: 작은 tabular MDP를 policy iteration 또는 value iteration으로 정확히 풉니다. 수렴 동작을 보고합니다.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

known model을 가진 MDP가 주어지면 다음을 출력하세요.

1. 선택. Policy iteration vs value iteration. |S|, |A|, γ와 연결한 이유.
2. 초기화. V_0, 시작 policy. 수렴 민감도.
3. 중단 조건. Sup-norm tolerance ε. 예상 sweep 횟수.
4. 검증. 정확히 계산한 V*(s_0). 추출한 greedy policy.
5. 활용. 이 baseline을 sampling-based method 디버그/평가에 어떻게 사용할지.

state space가 10⁷을 넘으면 DP 실행을 거부하세요. sup-norm check 없이 수렴을 주장하지 마세요. infinite-horizon 과제에서 γ ≥ 1이면 guarantee violation으로 표시하세요.
