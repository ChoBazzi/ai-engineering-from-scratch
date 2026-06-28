---
name: vae-trainer
description: 주어진 데이터셋과 downstream 용도에 맞춰 VAE architecture, latent size, beta schedule, eval plan을 지정합니다.
version: 1.0.0
phase: 8
lesson: 02
tags: [vae, latent, generative]
---

데이터셋 프로필(modality, resolution, dataset size)과 downstream 용도(reconstruction only, sampling, latent-diffusion 또는 token-AR 모델의 input-encoder)가 주어지면 다음을 출력하세요.

1. 변형. Plain VAE, beta-VAE, VQ-VAE, RVQ(residual), NVAE 중 하나. modality와 downstream 용도에 연결한 한 문장 이유를 함께 제시합니다.
2. 아키텍처. Encoder / decoder topology(conv downsample factor, channel width, hidden dim, attention blocks). 해당되는 경우 public reference weights(`sd-vae-ft-ema`, Encodec, DAC, WAN-VAE)를 언급합니다.
3. Latent 차원. Spatial dim과 channel dim. sample당 total bits. raw data 대비 compression ratio.
4. Beta schedule. Warmup ramp, final value, 사용한다면 free-bits threshold.
5. 평가 계획. Reconstruction MSE / SSIM / PSNR, dimension별 KL, active-dim count, posterior-collapse alarm threshold, `q(z|x)`와 prior 사이의 Frechet distance.

training 시작 시 beta > 0.5인 VAE 배포는 거부하세요(posterior collapse). 이미지는 plain Gaussian VAE를 최종 generator로 쓰지 마세요. 흐릿해집니다. 대신 diffusion 또는 flow-matching 모델의 latent encoder로 사용하세요. codebook usage가 20% 미만인 VQ-VAE는 codebook reset policy가 잘못 설정된 것으로 표시하세요.
