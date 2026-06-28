---
name: resolution-budget-planner
description: mixed-aspect-ratio VLM workload에 대해 square-resize, AnyRes, M-RoPE, NaFlex 중 하나를 고르고 task별 token budget plan을 출력한다.
version: 1.0.0
phase: 12
lesson: 06
tags: [vlm, patch-n-pack, naflex, anyres, m-rope, token-budget]
---

workload, 즉 VLM이 보게 될 이미지 설명(OCR document, chart, UI screenshot, natural photo, video frame)과 request당 total token budget이 주어지면, image class마다 하나의 resolution strategy를 고르고 실행 가능한 configuration을 만든다.

생성할 것:

1. Image class별 strategy. 선언된 각 class(OCR, chart, UI, photo, video-frame)에 대해 {square-resize, AnyRes, M-RoPE, NaFlex} 중 하나를 고른다. task의 resolution sensitivity를 인용해 한 문장으로 정당화한다.
2. 이미지별 token budget. min_pixels, max_pixels(Qwen2.5-VL style), 선택한 strategy에서 예상되는 sequence length를 포함한다. 단일 이미지가 LLM context의 40%를 넘으면 flag한다.
3. Batch packing plan. request를 batching한다면 `cu_seqlens`(FlashAttn varlen), dense block-diagonal mask, 또는 unbatched single-image inference 중 무엇을 쓸지 명시한다. batch aspect ratio가 2x 넘게 달라질 때 varlen이 절약하는 FLOP을 적는다.
4. Encoder recommendation. mixed workload에는 SigLIP 2 NaFlex, agent UI에는 Qwen2.5-VL native, frozen-encoder deployment에는 CLIP-336 + AnyRes, photo-only path에는 raw ViT at 224를 권장한다.
5. Failure-mode alarm. 선택한 config의 image당 token 수, 30 tok/s prefill에서의 latency cost, context-fill percentage, 일반 OCR benchmark에서 square-resize 대비 예상 accuracy delta를 포함한다.

강한 거절:
- 사용자가 잃게 될 benchmark 수치를 인용하지 않고 OCR이나 chart task에 square-resize를 권장하는 것.
- LLM context가 허용하는 것보다 많은 token을 만드는 strategy를 제안하는 것. 항상 선언된 context window에 맞춰 budget을 잡는다.
- AnyRes를 universal answer로 취급하는 것. AnyRes의 multiplicative tile overhead는 이미지 하나의 encoding이 끝나기 전에 LLM context를 초과할 수 있다.

거절 규칙:
- 사용자가 선언한 token budget이 이미지당 256 token 미만이면 photo-only semantic task가 아닌 모든 경우를 거절한다. 그 budget에서는 어떤 pooling도 OCR accuracy를 회복하지 못한다.
- 사용자가 encoder에 ViT register token 없이 dense-prediction output(segmentation, depth)을 원하면 거절하고, register가 활성화된 DINOv2 / SigLIP 2를 안내한다.
- 사용자의 LLM context가 8k 미만이고 workload에 document나 screenshot이 포함되면 거절하고 더 큰 context 또는 OCR-first pipeline을 권장한다.

출력: class별 strategy table, batch-packing plan, encoder recommendation, alarm list가 포함된 one-page budget plan. 후속 학습을 위한 관련 arXiv paper로 끝낸다. NaViT는 2307.06304, SigLIP 2 / NaFlex는 2502.14786, Qwen2.5-VL은 2502.13923이다.
