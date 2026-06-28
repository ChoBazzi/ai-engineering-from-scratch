---
name: policy-gradient-trainer
description: 주어진 작업에 대한 REINFORCE / actor-critic / PPO 학습 config를 만들고 variance 문제를 진단한다.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

환경(discrete / continuous actions, horizon, reward stats)이 주어지면 다음을 출력하라.

1. Policy head. Softmax(discrete) 또는 Gaussian(continuous), parameter count 포함.
2. Baseline. None(vanilla), running mean, 학습된 `V̂(s)`, 또는 A2C critic.
3. Variance 제어. Reward-to-go는 기본으로 켜고, return normalization, gradient clip value를 정한다.
4. Entropy bonus. Coefficient β와 decay schedule.
5. Batch size. Update당 episode 수와 on-policy data freshness contract.

horizon이 500 step을 넘는 REINFORCE-no-baseline은 거부하라. Softmax head를 쓰는 continuous-action control은 거부하라. `β = 0`이고 관측된 policy entropy가 0.1 미만인 run은 entropy-collapsed로 표시하라.
