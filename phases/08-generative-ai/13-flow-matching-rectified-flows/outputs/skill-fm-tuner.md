---
name: fm-tuner
description: Diffusion 학습 계획을 flow-matching / rectified-flow 설정으로 변환합니다.
version: 1.0.0
phase: 8
lesson: 13
tags: [flow-matching, rectified-flow, diffusion]
---

Diffusion 스타일 학습 계획(data, compute, schedule, target step count, quality bar)이 주어지면 flow-matching에 대응되는 설정을 출력하세요.

1. Schedule + interpolant. Linear(rectified flow), optimal transport(Lipman OT-CFM), variance-preserving, cosine 중 하나를 고릅니다. 한 문장으로 이유를 설명합니다.
2. Time sampling. Uniform, logit-normal(SD3), mode-weighted 중 하나를 고릅니다. 1000 Hz에서 uniform sampling이 끝점에 용량을 낭비하는 경우 경고합니다.
3. Target. Velocity v = x_1 - x_0(rectified flow) 또는 alpha'(t)x_1 + sigma'(t)x_0(CFM). 어떤 것을 쓰는지 명시합니다.
4. Optimizer + lr warmup. Transformer scale에서 안정성을 위해 beta2 = 0.95인 AdamW를 포함합니다.
5. Reflow plan. reflow를 0, 1, 2번 중 몇 번 실행할지 정합니다. 반복당 예산은 선별된 subset 전체를 다시 추론하는 비용과 비슷합니다.
6. Step counts. 학습 step count 목표, 예상 inference steps(20, 4, 2, 1), guidance scale 범위를 제시합니다.
7. 평가. Diffusion baseline 대비 FID / CLIP-score를 보고하고, quality vs step count를 플롯합니다.

v_1이 수렴하기 전에는 reflow를 거부하세요. 나쁜 모델에 reflow를 적용하면 나쁜 방향이 그대로 굳어집니다. 그 위에 consistency distillation이 없으면 1-step inference 추천을 거부하세요. 20 step을 초과하는 inference를 목표로 하는 flow-matching 모델은 모두 표시하세요. 그만큼 많은 step이 필요하다면 재정식화의 이점을 낭비한 것입니다.
