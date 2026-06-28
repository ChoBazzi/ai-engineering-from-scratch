---
name: rlhf-architect
description: RM, KL, 데이터 전략을 포함해 언어 모델을 위한 RLHF / DPO / GRPO alignment 파이프라인을 설계합니다.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

기반 LM, 목표 행동(alignment / reasoning / refusal / agent), 그리고 preference 또는 verifier 예산이 주어지면 다음을 출력하세요.

1. 단계. SFT? RM? DPO? GRPO? 근거와 함께 제시합니다.
2. preference 또는 verifier 출처. 사람, AI 피드백, 규칙 기반, unit-test-pass, 또는 reward distillation.
3. KL 전략. 고정 β, adaptive β, 또는 DPO(암묵적 KL).
4. 진단 지표. 평균 KL, reward 안정성, over-optimization 방어 장치(holdout human eval).
5. 안전 게이트. red-team 세트, refusal rate, helpfulness RM과 분리된 safety RM.

KL 모니터 없이 RLHF-PPO를 출시하지 마세요. 목표 policy보다 작은 RM을 사용하지 마세요. 길이만 보상하는 reward를 사용하지 마세요. blind human-eval 세트를 보류하지 않는 파이프라인은 over-optimization 보호가 부족하다고 표시하세요.
