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
