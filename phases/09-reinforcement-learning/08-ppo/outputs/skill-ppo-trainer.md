---
name: ppo-trainer
description: 주어진 환경을 위한 PPO training config와 diagnostic plan을 만든다.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

환경과 training budget이 주어지면 다음을 출력하라.

1. Rollout 크기. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate parameter. `ε`(clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. 명시적 `γ`와 `λ`를 포함한 GAE(`λ`).
5. 진단 계획. KL, clip fraction, explained variance threshold와 alert.

`K > 30` 또는 `ε > 0.3`은 거부하라(unsafe trust region). Advantage normalization 또는 KL/clip monitoring이 없는 PPO run은 거부하라. Clip fraction이 지속적으로 0.4를 넘으면 drift로 표시하라.
