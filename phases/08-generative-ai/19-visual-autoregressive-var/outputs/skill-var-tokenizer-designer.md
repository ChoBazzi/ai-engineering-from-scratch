---
name: var-tokenizer-designer
description: Next-scale visual autoregressive image generation을 위한 multi-scale residual VQ tokenizer를 설계합니다.
version: 1.0.0
phase: 8
lesson: 19
tags: [var, next-scale-prediction, vq-vae, residual-vq, image-generation, tokenizer]
---

Image target(resolution, channels, color vs grayscale, dataset size, downstream LM compute budget, target FID)이 주어지면 다음을 출력하세요.

1. Scale schedule. 1x1부터 (H/p) x (W/p)까지 K개의 resolution levels를 나열합니다. 기본값은 256x256에서 10 scales, 512x512에서 14 scales입니다. LM의 effective sequence length(scale areas의 합)와 per-pass parallel-within-scale budget에 맞춰 K를 정당화합니다.
2. Codebook. 모든 scale에서 단일 shared codebook size V를 사용합니다(일반적으로 4096 / 8192 / 16384). Dataset size와 decoder capacity로 V를 고릅니다. Calibration batch에서 codebook usage가 50 percent를 넘는지 확인하고, 아니면 V를 줄입니다.
3. Residual sharing. Scales 1..K가 upsampled embeddings의 합으로 latent를 함께 재구성하는지 확인합니다(residual VQ). Patch size p와 VAE backbone(VQGAN-style discriminator on / off, perceptual loss weight)을 명시합니다.
4. Decoder. 합산된 latent를 pixels로 매핑하는 VAE decoder입니다. VQGAN decoder, VAR-paper decoder, 더 가벼운 MAGVIT-style decoder 중에서 고릅니다. FID target과 decoder VRAM을 기준으로 정당화합니다.
5. Position embedding. Scale별 learned embedding과 scale 내부 2D sin-cos를 쓰는 (scale_index, row, col) triple인지 확인합니다. Flat 1D positions는 거부하세요. LM이 올바른 conditional을 적용하려면 scale label이 필요합니다.

VAR에 대해 non-residual multi-scale tokenizer를 거부하세요. Summed residuals가 없으면 next-scale conditional이 제대로 정의되지 않고, LM은 논문이 증명한 것과 다른 objective를 최적화합니다. V가 더 작은 scale의 pixel count에 맞게 보정되고 codebook collapse가 완화된 경우가 아니라면 separate per-scale codebooks를 거부하세요. K x average-scale-area가 text conditioning을 위한 여유분을 뺀 LM의 max sequence length를 초과하면 next-scale prediction 자체를 거부하세요.

예시 입력: "ImageNet class-conditional 256x256, dataset 1.2M, LM budget 1.5B params, target FID under 5.0."

예시 출력:
- Scale schedule: K=10, sizes 1, 2, 3, 4, 5, 6, 8, 10, 13, 16. Total tokens 671.
- Codebook: shared, V=4096. ImageNet 256에서 70-80 percent usage를 예상합니다.
- Residual sharing: confirmed; p=16, perceptual + adversarial losses를 쓰는 VQGAN backbone, residual sum이 f를 재구성합니다.
- Decoder: VQGAN decoder, 4 upsampling blocks, extra refiner 없음.
- Position embedding: (scale, row, col) triple, learned scale token + scale 내부 2D sin-cos.
