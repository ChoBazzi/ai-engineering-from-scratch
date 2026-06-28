---
name: tokenizer-vs-adapter-picker
description: VLM 프로젝트에 Chameleon식 early fusion(shared-vocab tokenizer)과 LLaVA식 late fusion(frozen LLM 위 adapter) 중 무엇을 쓸지 고른다.
version: 1.0.0
phase: 12
lesson: 11
tags: [chameleon, early-fusion, vq-vae, late-fusion, adapter]
---

Product specification(understanding-only 또는 understanding+generation), target image quality(social-post / magazine / print / broadcast), cost budget(training + inference)이 주어지면 Chameleon 계열 또는 LLaVA 계열을 추천하고 concrete architecture outline을 제시하라.

다음을 산출하라.

1. Verdict. Early-fusion(Chameleon / Emu3 / AnyGPT) 또는 late-fusion(LLaVA / BLIP-2 / Qwen-VL) 계열.
2. Tokenizer pick(early-fusion verdict의 경우). VQ-VAE(Chameleon), MAGVIT-v2, IBQ, SBER-MoVQGAN 중 선택하고 예상 reconstruction ceiling을 PSNR로 제시하라.
3. Training-stability plan. Scale 있는 early-fusion을 위한 QK-Norm, dropout 배치, LayerNorm 순서.
4. Cost estimate. Late-fusion 대안 대비 training GPU-hour와 이미지당 inference latency.
5. Generation-quality ceiling. 사용자가 기대할 수 있는 PSNR / FID 범위와 product의 quality bar가 discrete token으로 도달 가능한지, 아니면 continuous(Transfusion식) generation이 필요한지.
6. Migration path. 사용자가 성장해 late-fusion이 제한이 될 때(이미지 출력이 필요해질 때) migration이 어떤 모습인지.

강한 거절:
- Understanding-only product에 Chameleon식을 추천하는 것. Pure understanding에는 late-fusion이 더 단순하고 싸며 ceiling이 더 높다.
- Production image generation에 K<4096인 VQ-VAE를 제안하는 것. Codebook이 너무 작아 artifact가 보인다.
- Early-fusion inference가 공짜라고 주장하는 것. VQ decoder는 생성 이미지마다 50-200ms를 더하며, 종종 LLM output time보다 크다.

거절 규칙:
- 사용자가 frontier-quality image generation(FID < 15, print-ready)을 원하면 discrete token을 거부하고 Transfusion / Stable Diffusion 3 / MMDiT(Lesson 12.13)를 가리켜라.
- Product에 image output이 절대 필요 없으면 early-fusion을 거부하라. Complexity가 정당화되지 않는다.
- 사용자가 기존 Llama / Qwen LLM weight를 plug in하고 싶어 하면 early-fusion을 거부하라. Fresh model pretraining이 필요하다.

출력: verdict, tokenizer pick, stability checklist, cost estimate, quality ceiling, migration path가 있는 one-page plan. 비교 읽을거리로 arXiv 2405.09818(Chameleon)과 2408.11039(Transfusion)를 끝에 붙여라.
