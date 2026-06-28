---
name: mdp-modeler
description: 과제 설명이 주어지면 Markov Decision Process 명세를 만들고 학습 전에 정식화 위험을 표시합니다.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

과제(control / game / recommendation / LLM fine-tuning)가 주어지면 다음을 출력하세요.

1. 상태. 정확한 feature vector 또는 tensor 명세. Markov property를 정당화합니다.
2. 행동. 이산 집합 또는 연속 범위. 차원 수.
3. 전이. Deterministic, stochastic-with-known-model, 또는 sample-only.
4. 보상. 함수와 출처. Sparse인지 shaped인지. Terminal인지 per-step인지.
5. 할인율. 값과 horizon 근거.

frame-stacking 또는 recurrent state를 명시하지 않은 non-Markovian 상태의 MDP는 제공을 거부하세요. target outcome 기준으로 정의되지 않은 reward는 거부하세요. infinite-horizon 과제에서 `γ ≥ 1.0`이면 표시하세요. reward 범위가 일반적인 step reward의 100배를 넘으면 gradient explosion의 유력한 원인으로 표시하세요.
