---
name: vla-action-format-picker
description: 로봇 작업에 맞는 행동 형식(discrete bin, FAST, flow-matching, dual-system)과 VLA 계열(RT-2, OpenVLA, π0, GR00T)을 고른다.
version: 1.0.0
phase: 12
lesson: 21
tags: [vla, rt-2, openvla, pi0, groot, action-tokenization]
---

로봇 작업(조작, 내비게이션, 전신 휴머노이드), DOF 수, 제어율 요구사항, compute 제약이 주어지면 행동 형식과 VLA 계열을 고른다.

생성할 것:

1. 행동 형식. 단순 단일 팔 작업에는 discrete-bin, 속도에 민감한 trajectory에는 FAST, 부드러운 연속 제어에는 flow-matching, 휴머노이드에는 dual-system.
2. VLA 계열 선택. RT-2(폐쇄형), OpenVLA(공개 7B), π0(공개 flow), GR00T N1(공개 dual-system 휴머노이드).
3. 제어율 실현 가능성. 형식의 처리량을 필요한 제어 Hz와 맞춘다. discrete bin은 7B 모델에서 10 Hz를 초과할 수 없다.
4. 학습 데이터 혼합. 공동 미세 조정 비율(웹 VQA : 로봇). 0.5:1에서 시작하고 작업별로 조정한다.
5. 미세 조정 계획. 작업 시연 약 500-1000개에는 LoRA, 약 1만 개 시연에는 전체 미세 조정.
6. 안전 게이트. VLA 바깥에 필요한 제어 계층 검사.

강한 거부:
- 안전 계층 명세 없이 VLA를 추천하는 것. 항상 관절 한계와 속도 clipping을 포함한다.
- discrete-bin 토큰화가 30 Hz 제어에 충분히 빠르다고 주장하는 것. 그렇지 않다.
- 충분한 smoothness 제약 없이 flow-matching을 제안하는 것. 분포 밖 행동은 여전히 발생한다.

거부 규칙:
- 제어율 요구사항이 <=7B 모델의 discrete-bin 형식에서 50 Hz를 초과하면 거부하고, π0 또는 특화 head를 추천한다.
- 로봇이 30 DOF를 초과하면(휴머노이드) 단일 단계 아키텍처를 거부하고 dual-system(GR00T)을 요구한다.
- 예산이 Open X-Embodiment 규모의 사전학습을 감당할 수 없다면 from-scratch VLA를 거부하고 OpenVLA 미세 조정을 추천한다.

출력: 행동 형식, VLA 선택, 제어율 검사, 공동 미세 조정 혼합, 안전 게이트가 포함된 한 페이지 계획. arXiv 2307.15818(RT-2), 2406.09246(OpenVLA), 2410.24164(π0), 2503.14734(GR00T)로 끝낸다.
