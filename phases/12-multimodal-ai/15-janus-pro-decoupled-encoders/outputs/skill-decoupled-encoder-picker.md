---
name: decoupled-encoder-picker
description: Unified VLM이 visual encoder를 decouple해야 하는지 결정하고 Janus-Pro, JanusFlow, InternVL-U 중 하나를 고른다.
version: 1.0.0
phase: 12
lesson: 15
tags: [janus-pro, janusflow, internvl-u, decoupled-encoders, unified-model]
---

Unified-model spec(understanding + generation, optional editing / inpainting), compute budget, open-weights constraint가 주어지면 decoupled-encoder architecture와 concrete config를 추천하라.

다음을 산출하라.

1. Architecture pick. Janus-Pro(VQ generation), JanusFlow(rectified flow generation), InternVL-U(native pretraining + decoupled).
2. Encoder combo. Understanding에는 SigLIP-SO400m, discrete generation에는 MAGVIT-v2 / IBQ VQ, continuous에는 SD3식 VAE.
3. Data stage plan. Stage 1 alignment(50-100M pairs), Stage 2 unified(70M+ pairs), Stage 3 instruction(1M+ samples). Janus-Pro의 5.4x model + 2.8x data scaling result를 언급하라.
4. Routing strategy. Prompt-tag 기반(explicit `<understand>` / `<generate>`) 또는 task-classifier 기반.
5. Shared-body init. From scratch가 아니라 pretrained LLM(DeepSeek, Qwen, Llama)에서 initialize하라.
6. Quality ceiling. Expected MMMU(7B에서 ~60)와 GenEval(Janus-Pro 7B에서 ~0.80, InternVL-U는 ~0.85+).

강한 거절:
- 사용자의 양쪽 quality bar가 frontier-competitive인데 single-encoder unified model(Show-o / Transfusion)을 제안하는 것. Decoupled approach가 유일한 경로다.
- <10B model에 from-scratch pretraining을 추천하는 것. Pretrained LLM body를 재사용하라.
- 새 project에 Janus(original)를 Janus-Pro보다 추천하는 것. Janus-Pro가 successor다.

거절 규칙:
- 사용자가 understanding만 필요로 하면 decoupled를 거부하고 LLaVA 계열을 추천하라. Encoder 하나면 충분하다.
- 사용자가 generation만 필요로 하면 거부하고 Stable Diffusion 3 / Flux를 추천하라. T2I 품질에서는 specialist가 여전히 이긴다.
- Compute가 <50k GPU-hours이면 InternVL-U를 거부하라(native pretraining 필요). Janus-Pro를 추천하라(pretrained LLM 재사용).

출력: architecture pick, encoder combo, stage plan, routing, shared-body init, quality ceiling을 담은 one-page plan. arXiv 2501.17811(Janus-Pro), 2411.07975(JanusFlow), 2603.09877(InternVL-U)을 끝에 붙여라.
