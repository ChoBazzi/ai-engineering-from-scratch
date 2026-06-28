---
name: token-gen-cost-analyzer
description: Emu3식 next-token generation의 token count, inference latency, quality ceiling을 계산하고 Emu3 계열과 diffusion 중 하나를 고른다.
version: 1.0.0
phase: 12
lesson: 12
tags: [emu3, next-token-prediction, video-gen, diffusion, cfg]
---

Generation product spec(image 또는 video, target resolution, quality tier, throughput requirement)이 주어지면 Emu3식 next-token generation의 token count를 계산하고, inference cost를 추정하며, Emu3 계열과 diffusion 중 하나를 고르라.

다음을 산출하라.

1. Token count. 선택한 tokenizer reduction에서 image당 token(이미지는 보통 각 dim 8x). 3D VQ의 video당 token(보통 4x4x4 spatiotemporal).
2. Inference latency. Emu3 계열은 tokens / throughput(tokens-per-second), diffusion은 denoise-steps * step-time. Concrete A100 / H100 range를 제시하라.
3. Quality ceiling. Tokenizer reconstruction PSNR(IBQ 계열은 30-32 dB), MJHQ-30K의 FID expectation, video의 FVD.
4. CFG configuration. Task별 recommended guidance weight(gamma). 표준 generation은 보통 3.0, 강한 prompt adherence는 5-7.
5. Pick. Product가 unified understanding + generation 또는 any-modality flexibility를 필요로 하면 Emu3 계열, strict latency가 있는 image-gen-only product면 diffusion(SDXL / SD3 / Flux).

강한 거절:
- Emu3가 inference에서 diffusion보다 빠르다고 주장하는 것. 그렇지 않다. 수천 개 image token에 대한 autoregressive decode가 지속 비용이다.
- CFG weight를 명시하지 않고 Emu3 계열을 추천하는 것. 없으면 품질이 무너진다.
- Strict 4K image generation에 Emu3를 제안하는 것. 2048+ resolution의 token count는 KV cache를 터뜨리고 몇 분이 걸린다.

거절 규칙:
- Latency budget이 이미지당 <5s이면 Emu3를 거부하고 SDXL 또는 SD3를 추천하라.
- Product가 이미지를 emit하고, 설명하고, third-party image에 대해 reason해야 한다면 Emu3 계열을 추천하라. Unified loss가 핵심이며 diffusion은 별도 VLM 없이는 이를 할 수 없다.
- 사용자가 commercial use를 위한 permissive license open weights를 원하면 Emu3를 거부하라. 먼저 license를 확인해야 한다. 일부 version은 research-only다.

출력: token count, latency estimate, quality ceiling, CFG config, 근거가 있는 pick을 담은 one-page analysis. 대안으로 arXiv 2409.18869(Emu3)와 2408.11039(Transfusion)를 끝에 붙여라.
