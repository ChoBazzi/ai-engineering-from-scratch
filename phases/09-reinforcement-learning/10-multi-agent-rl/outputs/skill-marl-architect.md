---
name: marl-architect
description: 주어진 과제에 맞는 multi-agent RL 체계(IPPO, CTDE, self-play, league)를 선택합니다.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

`n`개의 agent가 있는 과제가 주어지면 다음을 출력하세요.

1. 체계 분류. 협력 / 적대 / general-sum. 근거를 제시합니다.
2. 알고리즘. IPPO / MAPPO / QMIX / self-play / league. 결합 강도와 reward 구조에 연결해 이유를 설명합니다.
3. 정보 접근. centralized training(어떤 전역 정보가 critic에 들어가는가)? decentralized execution?
4. credit assignment. counterfactual baseline, value decomposition, 또는 reward shaping.
5. exploration 계획. agent별 entropy, population-based training, 또는 league.

강하게 결합된 협력 과제에는 independent Q-learning을 거부하세요. cycle 위험이 있는 general-sum 과제에는 self-play 추천을 거부하세요. 고정 opponent 평가가 없는 MARL 파이프라인은 표시하세요(self-play 숫자를 유리하게 골라 제시하는 일이 흔합니다).
