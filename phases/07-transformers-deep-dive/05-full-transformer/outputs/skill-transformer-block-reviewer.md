---
name: transformer-block-reviewer
description: 2026년 기본값을 기준으로 transformer block 구현을 검토하고 drift를 표시합니다.
version: 1.0.0
phase: 7
lesson: 5
tags: [transformers, architecture, review]
---

transformer block source(PyTorch / JAX / numpy / pseudocode)와 의도한 role(encoder / decoder / encoder-decoder)이 주어지면 다음을 출력하세요.

1. wiring 점검. pre-norm인지 post-norm인지 확인합니다. 각 sublayer 주변의 residual connection을 확인합니다. 작성자가 이유를 명시하지 않았다면 post-norm을 2026년 기준 non-default로 표시합니다.
2. normalization. LayerNorm vs RMSNorm을 확인합니다. RMSNorm을 선호합니다. Q/K/V/O projection에 bias term이 있으면 표시합니다. 2026년 model 대부분은 이를 제거합니다.
3. attention shape. MHA / GQA / MQA / MLA를 확인합니다. decoder block의 경우 causal mask가 적용되었는지 확인합니다. cross-attention의 경우 Q가 decoder에서, K/V가 encoder에서 오는지 확인합니다.
4. FFN. activation(ReLU / GELU / SwiGLU / GeGLU)과 expansion ratio를 확인합니다. 약 2.67×의 SwiGLU가 현대적 기본값이고, 4× ReLU/GELU는 고전적 방식입니다.
5. positional signal. RoPE / ALiBi / absolute가 기대되는 위치(보통 RoPE는 Q,K projection)에 적용되었는지 확인합니다.

post-norm을 사용하고 warmup schedule이 없는 12 layer 초과 block stack에는 승인하지 마세요. training이 diverge합니다. causal masking이 없는 decoder block도 승인하지 마세요. FFN expansion이 2× 아래로 떨어지는 block은 용량 부족 가능성이 높다고 표시하세요. block이 swap-in sizing을 위한 config field 없이 `d_model`을 hard-code하면 경고하세요.
