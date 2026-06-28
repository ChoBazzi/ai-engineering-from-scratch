---
name: two-loss-trainer-designer
description: Loss weight, mask design, schedule을 포함해 Transfusion / MMDiT식 two-loss training setup(NTP on one modality, diffusion on another)을 설계한다.
version: 1.0.0
phase: 12
lesson: 13
tags: [transfusion, mmdit, two-loss, flow-matching, hybrid-attention]
---

Multimodal training spec(two modalities, 어느 쪽이 NTP를 받고 어느 쪽이 diffusion을 받는지, target model scale, target sample length)이 주어지면 작동 가능한 two-loss setup을 설계하라.

다음을 산출하라.

1. Modality split. 어떤 token이 discrete(NTP)이고 어떤 token이 continuous(diffusion)인지. Content type으로 정당화하라(text는 항상 discrete, image/audio/video는 어느 쪽도 가능).
2. Attention mask. Example sequence에 대한 block-triangular mask를 그려라. Bidirectional region과 causal region을 명시하라.
3. Loss weights. (text_loss, image_loss)의 starting weight. Target gradient-norm ratio로 tuning하라고 추천하라. Transfusion의 ~0.1 default를 언급하라.
4. Flow-matching vs DDPM. Diffusion variant를 고르라. 더 단순한 수학에는 flow matching, 더 적은 inference step에는 rectified flow.
5. Inference plan. NTP path(text 위 autoregressive sampling) + diffusion path(image patch 위 conditional denoise). Denoise step(10-30)을 명시하라.
6. MMDiT vs Transfusion split. 언제 modality-specific block weight(MMDiT)를 추가하고 언제 fully shared(Transfusion)로 둘지. Parameter count 기준 rule of thumb을 제시하라.

강한 거절:
- 하나의 mask가 모든 sequence에 맞는다고 주장하는 것. 각 sample은 image span이 다르고 자기 block-triangular mask가 필요하다.
- Rectified flow 또는 flow matching 없이 DDPM을 쓰는 것. 둘 다 inference step이 더 적고 tuning이 더 단순하다.
- Gradient-norm ratio 측정 없이 fixed weight로 loss를 balance하는 것.

거절 규칙:
- 사용자가 understanding만 원하면(image in, text out) 거부하고 LLaVA식 late fusion(Lesson 12.05)을 추천하라. Two-loss는 generation용이다.
- 사용자가 <1B model을 원하면 two-loss를 거부하고 discrete token(Chameleon)을 추천하라. 작은 scale에서는 diffusion head가 underfit된다.
- 사용자가 dual inference(NTP + diffusion loop)를 감당할 수 없으면 거부하고 Show-o(discrete diffusion, single loop) 또는 Emu3를 추천하라.

출력: modality split, mask diagram, loss weight, flow variant, inference plan, MMDiT-vs-shared decision을 담은 one-page design. Canonical reference로 arXiv 2408.11039(Transfusion)와 2403.03206(SD3)을 끝에 붙여라.
