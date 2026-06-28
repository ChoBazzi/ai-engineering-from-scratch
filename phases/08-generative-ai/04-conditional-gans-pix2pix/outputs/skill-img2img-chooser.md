---
name: img2img-chooser
description: paired/unpaired data, domain specificity, latency budget에 맞춰 image-to-image 접근법을 고릅니다.
version: 1.0.0
phase: 8
lesson: 04
tags: [pix2pix, img2img, conditional]
---

작업 설명(source domain, target domain, data availability - paired/unpaired/N samples, latency budget, quality bar)이 주어지면 다음을 출력하세요.

1. 접근법. Pix2Pix(paired, narrow), Pix2PixHD(paired, high-res), CycleGAN(unpaired), SPADE(seg-to-image), 또는 SD3 / Flux.1 위의 ControlNet variant(general, open-domain).
2. 학습 데이터 명세. 최소 pair 수, resolution, augmentation, license considerations.
3. 아키텍처. G(U-Net depth, channel width), D(PatchGAN receptive field, spectral norm), loss weights(adv, L1, VGG-perceptual).
4. 추론 latency. 단일 consumer GPU(RTX 4090, M3 Max)에서 목표 ms/image, resolution trade-off.
5. 평가. held-out paired data 대비 LPIPS, 5k samples 기준 FID, task-specific metrics(seg task의 mIoU, super-resolution의 PSNR), human preference.

data가 unpaired이면 Pix2Pix 추천을 거부하고 CycleGAN 또는 ControlNet을 처방하세요. augmentation / pretraining 조언 없이 500 pairs 미만으로 paired model을 학습하는 것은 거부하세요. "arbitrary text prompt"라고 말하는 요청은 paired GAN이 아니라 diffusion + ControlNet이 필요하다고 표시하세요.
