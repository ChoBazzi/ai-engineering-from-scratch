---
name: attention-shapes
description: Attention implementation의 shape bug를 debug한다.
phase: 5
lesson: 10
---

깨진 attention implementation이 주어지면 shape mismatch를 식별한다. 다음을 출력한다.

1. 어떤 matrix의 shape가 잘못되었는지. Tensor 이름을 명시한다.
2. 그 shape가 무엇이어야 하는지. `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)`에서 도출한다.
3. 한 줄 수정. Transpose, reshape, 또는 project.
4. Regression을 잡는 test. 일반적으로 `output.shape == (batch, T_dec, d_h)`와 `weights.shape == (batch, T_dec, T_enc)`를 assert하고, `weights.sum(dim=-1)`이 1에 가까운지 확인한다.

조용히 broadcast되는 fix는 추천하지 않는다. Broadcast에 가려진 bug는 나중에 silent accuracy degradation으로 드러난다.

Bahdanau 혼동에서는 decoder input이 `s_{t-1}`(pre-step state)라고 insist한다. Luong에서는 `s_t`(post-step state)다. Dot-product attention에서 처음 가장 흔한 error는 query/key dimension mismatch이므로 명시적으로 flag한다.
