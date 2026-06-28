---
name: unified-gen-model-picker
description: Open weights로 multimodal understanding과 generation을 모두 필요로 하는 product에 Show-o / Transfusion / Emu3 / Janus-Pro 계열 중 하나를 고른다.
version: 1.0.0
phase: 12
lesson: 14
tags: [show-o, masked-diffusion, unified, t2i, inpainting]
---

Open-weights constraint와 latency budget이 있고 unified understanding + generation(VQA, captioning, T2I, optional inpainting)이 필요한 product가 주어지면 model family를 고르고 reference configuration을 내라.

다음을 산출하라.

1. Family verdict. Show-o(masked discrete diffusion), Transfusion / MMDiT(continuous diffusion), Emu3 / Chameleon(autoregressive discrete), 또는 Janus-Pro(decoupled encoders).
2. Inference-step budget. Show-o는 16 step, Transfusion은 20 step, Emu3는 1024+ step. 사용자의 latency budget으로 선택을 정당화하라.
3. Inpainting support. Show-o는 공짜다. Transfusion은 mask channel을 추가한다. Emu3는 별도 fine-tune이 필요하다. 이를 사용자에게 표시하라.
4. Tokenizer pick. Discrete family에는 IBQ / MAGVIT-v2 / SBER를 추천하고, continuous에는 SD3의 VAE를 추천하라.
5. Training stability. Two-loss(Transfusion)는 weight tuning이 필요하고, Show-o의 single loss는 더 깔끔하다.
6. Migration path if user grows. Quality가 limit가 되면 Show-o에서 Transfusion으로 옮긴다.

강한 거절:
- Inference latency가 이미지당 <10s인데 Emu3 / Chameleon을 제안하는 것. 약 1024 token 이상의 autoregressive generation은 너무 느리다.
- Show-o가 frontier image quality에서 Transfusion과 맞먹는다고 주장하는 것. 그렇지 않다. Tokenizer가 ceiling이다.
- VQA가 필요한 product에 Stable Diffusion을 추천하는 것. SD는 image에 대해 reasoning하지 못한다.

거절 규칙:
- 사용자가 이미지 generation당 <2s를 원하면 Show-o를 거부하고 understanding용 별도 VLM + Stable Diffusion을 추천하라. Multi-model complexity를 받아들여라.
- 사용자가 open weights에서 "best-in-class quality"를 원하면 Show-o / Emu3를 거부하고 Transfusion 계열(MMDiT) 또는 JanusFlow를 추천하라.
- 사용자가 tokenizer에 commit할 수 없으면(licensing, quality ceiling 우려) discrete-only family를 거부하고 Transfusion을 추천하라.

출력: family verdict, step budget, inpainting support, tokenizer recommendation, stability plan, migration path가 있는 one-page pick. arXiv 2408.12528(Show-o), 2408.11039(Transfusion), 2501.17811(Janus-Pro)을 끝에 붙여라.
