# Sim-to-Real Transfer

> simulator에서 학습했지만 hardware에서 실패하는 policy는 simulator를 암기한 policy입니다. Domain randomization, domain adaptation, system identification은 learned controller가 reality gap을 넘게 만드는 세 가지 도구입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 minutes

## 문제

실제 robot을 학습시키는 일은 느리고, 위험하고, 비쌉니다. biped가 걷는 법을 배우려면 수백만 training episode가 필요합니다. 실제 biped가 한 번이라도 넘어지면 hardware가 망가집니다. simulation은 무제한 reset, deterministic reproducibility, parallel environment, 물리적 손상 없음이라는 이점을 제공합니다.

하지만 simulator는 틀립니다. bearing은 MuJoCo model보다 마찰이 더 큽니다. camera에는 simulator가 포함하지 않은 lens distortion이 있습니다. motor에는 delay, backlash, saturation이 있으며 99%의 sim model은 이를 생략합니다. 바람, 먼지, 가변 조명은 깨끗한 rendering에서 학습한 policy를 망가뜨립니다. **reality gap**, 즉 sim distribution과 real distribution 사이의 systematic difference가 robotics에 배포되는 RL의 중심 문제입니다.

필요한 것은 *sim-to-real distribution shift에 robust한 policy*입니다. 역사적인 세 접근법은 simulator를 randomize하는 것(domain randomization), 소량의 real data로 policy를 adapt하는 것(domain adaptation / fine-tuning), 또는 실제 system parameter를 식별해 맞추는 것(system identification)입니다. 2026년의 지배적 레시피는 massive parallel simulation(Isaac Sim, Isaac Lab, GPU 위의 Mujoco MJX)과 함께 이 셋을 결합합니다.

## 개념

![세 가지 sim-to-real 체계: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin et al. 2017, Peng et al. 2018. training 중 실제 robot과 다를 수 있는 모든 sim parameter를 randomize합니다. mass, friction coefficient, motor PD gain, sensor noise, camera position, lighting, texture, contact model 등이 포함됩니다. policy는 "오늘 내가 어떤 sim 안에 있는가"에 대한 conditional distribution을 학습하고, 전체 범위에 걸쳐 generalize합니다. 실제 robot이 training envelope 안에 있으면 policy는 작동합니다.

- **장점:** real data가 필요 없습니다. 하나의 recipe로 여러 robot에 적용할 수 있습니다.
- **단점:** over-randomized training은 "universal"하지만 지나치게 조심스러운 policy를 만듭니다. noise가 너무 많다는 것은 regularization이 너무 많다는 것과 비슷합니다.

**System Identification (SI).** training 전에 simulator parameter를 real-world data에 맞춥니다. 실제 robot의 arm-joint friction을 측정할 수 있다면 그 값을 sim에 넣습니다. 그런 뒤 그 값을 예상하는 policy를 학습합니다. 실제 system 접근이 필요하지만 reality gap을 직접 줄입니다.

- **장점:** 정밀하고 low-noise training target을 제공합니다.
- **단점:** 남은 model error는 policy에 보이지 않습니다. 작은 미식별 효과(예: motor deadband)도 deployment를 깨뜨릴 수 있습니다.

**Domain Adaptation.** sim에서 학습하고, 소량의 real data로 fine-tune합니다. 두 가지 flavor가 있습니다.

- **Real2Sim2Real:** real rollout을 사용해 residual simulator `f(s, a, z) - f_sim(s, a)`를 학습하고, 수정된 sim에서 학습합니다. 많은 real data 없이 gap을 좁힙니다.
- **Observation adaptation:** real obs → sim-like obs로 mapping하는 learned feature extractor(예: GAN pixel-to-pixel)를 가진 policy를 학습합니다. controller는 sim 안에 머뭅니다.

**Privileged learning / teacher-student.** Miki et al. 2022(ANYmal quadruped). simulation에서 privileged information(ground truth friction, terrain height, IMU drift)에 접근할 수 있는 *teacher*를 학습합니다. 그런 뒤 real-sensor observation만 보는 *student*로 distill합니다. student는 history에서 privileged feature를 추론하는 법을 배우며, physical parameter 전반에 robust해집니다.

**Massively parallel simulation.** 2024-2026. Isaac Lab, Mujoco MJX, Brax는 모두 단일 GPU에서 수천 개의 parallel robot을 실행합니다. 4,096개의 parallel humanoid를 쓰는 PPO는 몇 시간 만에 수년치 experience를 수집합니다. training distribution이 넓어질수록 "reality gap"은 줄어듭니다. 4,096개 env 각각이 서로 다른 randomized parameter를 가지면 DR은 거의 공짜가 됩니다.

**실제 2026년 recipe(quadruped walking 예):**

1. gravity, friction, motor gain, payload를 domain-randomized한 massive parallel sim.
2. privileged info(terrain map, body velocity ground truth)를 쓰는 teacher policy 학습.
3. proprioception(leg joint encoder)만 쓰도록 teacher에서 student policy distillation.
4. real IMU 위의 autoencoder를 통한 optional observation adaptation.
5. 배포. 10개 이상 environment에서 zero-shot. 실패하면 safety-constrained PPO로 몇 분간 real-world fine-tuning.

## 직접 만들기

이 레슨의 코드는 *noisy* transition을 가진 GridWorld에서 domain randomization을 보여 주는 작은 demo입니다. "sim"에서 randomized slip probability를 경험하는 policy를 학습하고, training 중 본 적 없는 slip level을 가진 "real" 환경에서 평가합니다. 이 형태는 MuJoCo-to-hardware transfer에 그대로 대응됩니다.

### 1단계: parameterized sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip`은 simulator가 노출하는 parameter입니다. 실제 robotics에서는 friction, mass, motor gain처럼 sim과 real 사이에서 shift하는 모든 것이 될 수 있습니다.

### 2단계: DR로 학습하기

각 episode 시작 시 `slip ~ Uniform[0.0, 0.4]`를 sample합니다. PPO / Q-learning / 무엇이든 학습합니다. 많은 episode 동안 반복합니다.

### 3단계: "real" slip에서 zero-shot 평가

`slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`에서 평가합니다. 앞의 네 값은 training support 안에 있고, `0.5`와 `0.7`은 밖에 있습니다. DR-trained policy는 support 안에서 거의 optimal을 유지하고 support 밖에서는 완만하게 성능이 떨어져야 합니다. fixed-slip-trained policy는 training slip 밖에서 brittle해집니다.

### 4단계: narrow training과 비교하기

`slip = 0.0`만으로 두 번째 policy를 학습합니다. 같은 `slip` sweep에서 평가합니다. real slip > 0이 되는 순간 catastrophic drop을 볼 수 있어야 합니다.

## 함정

- **Too much randomization.** `slip ∈ [0, 0.9]`에서 학습하면 policy가 너무 risk-averse해져 optimal path를 시도하지도 않습니다. "무슨 일이든 일어날 수 있다"가 아니라 *예상되는* real-world distribution에 맞추세요.
- **Too little randomization.** 얇은 slice에서만 학습하면 policy가 전혀 generalize하지 못합니다. policy가 개선될수록 distribution을 넓히는 adaptive curriculum(Automatic Domain Randomization)을 사용하세요.
- **Misidentified parameter space.** 잘못된 것을 randomize하면(real gap은 motor delay인데 camera hue를 randomize하는 경우) DR은 도움이 되지 않습니다. 먼저 실제 robot을 profile하세요.
- **Privileged info leakage.** teacher가 observation이 아니라 global state를 action에 사용하면 student가 따라잡을 수 없는 target을 만들 수 있습니다. teacher의 policy가 observation history로 student에게 실현 가능해야 합니다.
- **Sim-to-sim transfer failure.** policy가 더 어려운 sim variant에 robust하지 않다면 real world에도 robust하지 않습니다. 배포 전에 반드시 held-out sim variant에서 test하세요.
- **No real-world safety envelope.** sim에서 작동하고 "real에서도 작동"하는 policy라도 low-level safety shield가 없으면 hardware를 망가뜨릴 수 있습니다. non-learned controller에 rate limit, torque limit, joint limit을 추가하세요.

## 활용하기

2026년 sim-to-real stack:

| 도메인 | Stack |
|--------|-------|
| 다리 locomotion(ANYmal, Spot, humanoid) | Isaac Lab + DR + privileged teacher / student |
| 조작(dexterous hands, pick-and-place) | Isaac Lab + DR + DR-GAN for vision |
| 자율주행 | CARLA / NVIDIA DRIVE Sim + DR + real fine-tune |
| 드론 racing | RotorS / Flightmare + DR + online adaptation |
| 손가락 / in-hand manipulation | OpenAI Dactyl (전례 없는 scale의 DR) |
| 산업용 arm | MuJoCo-Warp + SI + small real fine-tune |

모든 scale의 control에서 workflow는 일관됩니다. sim을 가능한 한 잘 맞추고, 맞출 수 없는 것을 randomize하고, 거대한 policy를 학습하고, distill한 뒤, safety shield와 함께 배포합니다.

## 출시하기

`outputs/skill-sim2real-planner.md`로 저장하세요.

```markdown
---
name: sim2real-planner
description: 주어진 robot + task에 대해 DR, SI, safety를 포함한 sim-to-real transfer 파이프라인을 계획합니다.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

robot platform, task, 실제 hardware time 접근 권한이 주어지면 다음을 출력하세요.

1. reality gap 인벤토리. 예상 영향도 기준으로 정렬한 의심 원인(contact, sensing, actuation delay, vision).
2. DR 파라미터. 정확한 목록, 범위, 분포. 각 범위를 실제 측정값과 비교해 정당화합니다.
3. SI 단계. 측정할 파라미터와 측정 방법.
4. teacher/student 분리. teacher가 쓰는 privileged info와 student가 쓰는 obs.
5. 안전 범위. low-level limit, emergency stop, backup controller.

(a) zero-shot sim-variant test, (b) safety shield, (c) rollback plan 없이는 배포를 거부하세요. 측정된 실제 변동성보다 3배 이상 넓은 DR 범위는 over-randomized 가능성이 높다고 표시하세요.
```

## 연습 문제

1. **Easy.** fixed-slip GridWorld(slip=0.0)에서 Q-learning agent를 학습하세요. slip ∈ {0.0, 0.1, 0.3, 0.5}에서 평가하세요. return vs slip을 plot하세요.
2. **Medium.** `slip ~ Uniform[0, 0.3]`을 sampling하는 DR Q-learning agent를 학습하세요. 같은 sweep에서 평가하세요. slip=0.5(out-of-distribution)에서 DR이 얼마나 도움이 되나요?
3. **Hard.** curriculum을 구현하세요. slip=0.0에서 시작해 policy가 optimal의 90%에 도달할 때마다 DR range를 넓힙니다. slip=0.3 zero-shot에 도달하는 총 environment step을 fixed DR baseline과 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Reality gap | "sim-to-real 차이" | training physics/sensing과 deployment physics/sensing 사이의 distribution shift. |
| Domain randomization (DR) | "무작위 sim 전반에서 학습" | training 중 sim parameter를 randomize해 policy가 generalize하게 합니다. |
| System identification (SI) | "실제를 측정해 sim에 맞추기" | 실제 physical parameter를 추정하고, sim이 그에 맞도록 설정합니다. |
| Domain adaptation | "real data로 fine-tune" | sim training 뒤 소량의 real-world fine-tune을 수행합니다. obs 또는 dynamics를 adapt할 수 있습니다. |
| Privileged info | "teacher를 위한 ground truth" | sim만 가진 정보입니다. student는 obs history에서 이를 추론해야 합니다. |
| Teacher/student | "privileged 정보를 observable로 distill" | shortcut으로 학습한 teacher를, 그 shortcut 없이 mimic하도록 student가 배웁니다. |
| ADR | "Automatic Domain Randomization" | policy가 개선될수록 DR range를 넓히는 curriculum. |
| Real2Sim | "real data로 gap 줄이기" | sim이 real rollout을 모방하도록 residual을 학습합니다. |

## 더 읽을거리

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 원래의 DR 논문(robotics vision)입니다.
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — dynamics와 quadruped locomotion을 위한 DR.
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, scale 있는 ADR.
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal을 위한 teacher-student.
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 2025-2026 deployment를 이끄는 massively parallel sim.
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum method.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — modern sim-to-real pipeline의 바탕이 되는 Dyna framing(model을 planning + rollout에 사용).
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — sim-to-real method taxonomy와 benchmark result를 다루는 survey입니다.
