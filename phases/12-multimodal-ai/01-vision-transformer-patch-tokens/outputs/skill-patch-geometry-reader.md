---
name: patch-geometry-reader
description: ViT 설정을 읽고 downstream VLM 계획을 위한 patch-token, parameter, VRAM 분석을 생성합니다.
version: 1.0.0
phase: 12
lesson: 01
tags: [vit, patch-tokens, dinov2, siglip, vlm-backbone]
---

vision backbone 설정(patch size, resolution, hidden dim, depth, heads, 선택적 registers)이 주어지면, 이 encoder가 몇 개의 token을 내보내는지, 실행에 필요한 VRAM 비용이 얼마인지, downstream VLM 또는 dense-prediction 작업에 적합한 선택인지 알려주는 geometry 분석을 생성하세요.

생성할 내용:

1. Patch grid와 sequence length. Grid shape (H/P, W/P). CLS, registers, 모든 pooling token을 포함한 sequence length. 선언되어 있으면 multi-resolution 지원(NaFlex, AnyRes)을 강조하세요.
2. Parameter breakdown. Patch embed, position embed, transformer blocks(attention + MLP), final LN, total을 정확한 count와 사람이 읽기 쉬운 표기(예: 86.4M)로 모두 제시하세요.
3. Forward당 FLOPs. Attention(블록당 4 N D^2 + 2 N^2 D)과 MLP(블록당 16 N D^2)를 depth 전체에 합산하세요. 고해상도에서 문제가 될 N에 대한 quadratic 비용을 표시하세요.
4. VRAM estimate. 이미지 한 장의 single forward inference에서 필요한 activation memory와, encoder가 downstream LLM에 공급될 때의 KV-equivalent cache를 추정하세요.
5. Pooling recommendation. 선언된 downstream 작업에 따라 CLS, mean patch, register-based, 또는 skip-pooling-for-VLM 중 추천하세요.

강한 거절 기준:
- Patch token을 입력과 pixel-identical한 것으로 취급하는 모든 분석. Projection은 학습된 linear map입니다. Patch는 pixel이 아니라 추상 vector입니다.
- CLS가 항상 올바른 pooling이라고 주장하는 경우. 최신 dense-feature와 VLM 경로는 CLS를 완전히 건너뜁니다.
- NaFlex-style native-resolution 유연성을 언급하지 않고 2D-RoPE와 learned positional embedding을 서로 교체 가능하다고 취급하는 경우.

거절 규칙:
- 제공된 config가 이미지 크기를 정확히 나누지 못하는 patch size를 선언하면 거절하세요. 선언된 padding scheme 없이는 NaFlex-compatible config가 아닙니다.
- 호출자가 proprietary model(Gemini, Claude, GPT-5)의 정확한 pretrained weight count를 요구하면 거절하세요. 공개된 값이 아닙니다.
- 대상 deployment VRAM이 ViT-g/14-class model에 대해 4GB 미만이면 거절하고 SigLIP SO400m/14 이하의 작은 backbone을 추천하세요.

출력: token count, parameter breakdown, FLOPs estimate, VRAM budget, 추천 pooling strategy를 포함한 한 페이지 geometry analysis. 마지막에는 NaFlex 세부 사항은 SigLIP 2 paper(arXiv:2502.14786), dense feature는 DINOv2 paper, patch-n'-pack 구현은 Lesson 12.06을 가리키는 "what to read next" 문단으로 끝내세요.
