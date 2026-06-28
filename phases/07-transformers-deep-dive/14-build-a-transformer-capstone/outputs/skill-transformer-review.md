---
name: transformer-review
description: 13개의 Phase 7 lesson 기준으로 transformer-from-scratch 구현을 검토한다.
version: 1.0.0
phase: 7
lesson: 14
tags: [transformers, review, capstone]
---

transformer-from-scratch 코드베이스(PyTorch / JAX)가 주어지면 2026년 기본값에 비추어 검토하고, 빠졌거나 잘못된 부분을 표시하라.

1. attention. causal mask가 있다. `sqrt(d_head)`로 스케일한다. Multi-head split이 동작한다. 가능하면 Flash Attention을 쓴다. d_model ≥ 1024이면 GQA를 언급한다.
2. positional encoding. RoPE(2026년 선호) 또는 learned absolute(작은 모델에서는 허용). Sinusoidal은 역사적 방식으로 표시한다.
3. block wiring. pre-norm(post-norm 아님). RMSNorm(LayerNorm 아님). SwiGLU FFN(ReLU/GELU 아님). 모든 sublayer 주변에 residual. Linear layer의 bias는 제거한다(현대 기본값).
4. training. AdamW(또는 2026+에서는 Muon), linear warmup이 있는 cosine LR schedule, 1.0에서 gradient clipping, bf16 autocast. Token embedding과 lm_head 사이 weight tying.
5. loss. 모든 위치에서 shift-by-one cross-entropy. padding이 있으면 mask out한다. 고정 간격으로 train과 val loss를 로그한다.

다음 중 하나라도 있는 코드베이스에는 승인하지 말라. 명시적 이유 없는 post-norm, 정당화 없는 2026년 production code의 LayerNorm, decoder self-attention의 causal mask 누락, 작은 LM의 untied embeddings. 다음을 표시하라. validation split 없음, gradient clipping 없음, warmup 없이 LR > 1e-3, fallback 없이 positional embedding 범위를 초과하는 block_size. `python code/main.py`를 end-to-end로 실행하고 nano config의 tinyshakespeare에서 최종 val loss가 2.5 미만인지 확인하라고 권장하라.
