---
name: actor-critic-trainer
description: 주어진 환경을 위한 A2C / A3C / GAE configuration을 만들고, advantage estimation과 loss weight를 명시한다.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

환경과 compute budget이 주어지면 다음을 출력하라.

1. 병렬성. A2C(GPU batched) vs A3C(CPU async)와 worker 수.
2. Rollout 길이 T. Update마다 env별 step 수.
3. Advantage 추정기. n-step 또는 GAE(λ). λ를 명시한다.
4. Loss 가중치. `c_v`(value), `c_e`(entropy), gradient clip.
5. Learning rate. Actor와 critic(사용한다면 분리).

horizon이 1000을 넘는 환경에서 single-worker A2C는 거부하라(너무 on-policy이고 너무 느리다). Advantage normalization 없이 ship하지 말라. `c_e = 0`이고 관측된 entropy가 0.1 미만인 run은 entropy-collapsed로 표시하라.
