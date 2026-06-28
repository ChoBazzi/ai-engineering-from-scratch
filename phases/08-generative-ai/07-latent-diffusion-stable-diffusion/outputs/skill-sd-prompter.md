---
name: sd-prompter
description: 주어진 prompt, style, quality bar에 대해 Stable Diffusion / Flux inference를 구성합니다.
version: 1.0.0
phase: 8
lesson: 07
tags: [stable-diffusion, flux, latent-diffusion]
---

Prompt, target style, quality bar(fast preview / portfolio quality / print-ready)가 주어지면 다음을 출력합니다.

1. 모델 + checkpoint. SD 1.5(legacy tools), SDXL-base + refiner, SDXL-Turbo(fast), SD3.5-Large, Flux.1-dev(best open), Flux.1-schnell(fast open), 또는 hosted API(DALL-E 3, Imagen 4, Midjourney v7). 한 문장 이유를 포함합니다.
2. Sampler. Euler A(creative), DPM-Solver++ 2M Karras(stable), LCM(fast), 또는 flow-matching sampler(SD3/Flux). Step count를 포함합니다.
3. CFG scale. Turbo / LCM은 0, Flux는 3-4, SDXL은 5-7, SD1.5는 7-10. Trade-off를 문서화합니다.
4. Add-ons. ControlNet(pose, depth, canny, seg), IP-Adapter(reference image), LoRA(style 또는 subject), SD3+를 위한 T5 toggle.
5. Negative prompt. 명시적 empty string과 채워진 content(artifacts, low quality, wrong anatomy)는 다릅니다. 둘 다 지정합니다.

SDXL+에서 CFG &gt; 10은 거부하세요(saturated output). Non-legacy checkpoint에서 sampler step &gt; 50은 거부하세요(품질은 30 step 근처에서 plateau). 서로 다른 base model에서 학습된 LoRA를 섞는 것도 거부하세요(SD 1.5 LoRA on SDXL은 조용히 깨집니다). Photorealistic human 요청에는 NSFW, deepfake, copyright policy 관련 reminder가 필요하다고 표시하세요.
