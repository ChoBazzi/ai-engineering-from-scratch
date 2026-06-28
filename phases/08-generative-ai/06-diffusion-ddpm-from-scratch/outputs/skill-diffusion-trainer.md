---
name: diffusion-trainer
description: Diffusion training run의 schedule, prediction target, sampler, eval plan을 구성합니다.
version: 1.0.0
phase: 8
lesson: 06
tags: [diffusion, ddpm, training]
---

Dataset profile(modality, resolution, dataset size), compute budget(GPU hours, VRAM floor), quality bar(FID target 또는 downstream use)가 주어지면 다음을 출력합니다.

1. Schedule. Linear, cosine(Nichol), 또는 sigmoid. Step 수 T(DDPM baseline은 1000, 더 빠른 variant는 256).
2. 예측 target. epsilon, v-prediction, 또는 x_0. Schedule 전반의 resolution과 signal-to-noise에 연결된 이유를 포함합니다.
3. 아키텍처. Pixel diffusion을 위한 U-Net depth + channel width, latent diffusion을 위한 DiT, 또는 video를 위한 3D U-Net / DiT. Time embedding scheme(sinusoidal + MLP, FiLM, 또는 AdaLN)을 포함합니다.
4. Sampler. DDIM(20-50 step), DPM-Solver++(10-20), Euler-A(creative), 또는 distilled 1-4-step. Guidance scale(CFG w) 추천을 포함합니다.
5. 평가 계획. FID / KID / CLIP-score / human-preference, sample count(FID는 &gt;=10k), CFG w sweep protocol.

Latent diffusion이 FLOPs의 1/16로 같은 품질을 달성할 수 있는데도 &gt;=256x256에서 pixel-space diffusion 학습을 추천하지 마세요. Conditional generation에서 CFG 없이 model을 배포하는 것도 거부하세요. Conditional model의 zero-shot unconditional sample은 보통 degenerate합니다. beta_T &gt; 0.1인 schedule은 saturated output이나 불안정한 training을 만들 가능성이 높다고 표시하세요.
