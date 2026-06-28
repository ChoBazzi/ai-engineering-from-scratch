---
name: mha-configurator
description: 새 transformer의 head count, KV-head count, projection strategy(MHA / MQA / GQA / MLA)를 추천합니다.
version: 1.0.0
phase: 7
lesson: 3
tags: [transformers, attention, mha, gqa]
---

Transformer spec(parameter budget, hidden size `d_model`, target context length, inference device memory, training vs inference priority)가 주어지면 다음을 출력하세요.

1. projection variant. MHA, GQA, MQA, MLA 중 하나. KV-cache constraint와 연결된 한 문장 이유를 덧붙입니다.
2. head geometry. `n_heads`, `n_kv_heads`, `d_head`. 값은 `d_model = n_heads * d_head`와 `n_heads % n_kv_heads == 0`을 만족해야 합니다.
3. KV cache 추정. 선택한 variant에서 target context length 기준 token당 layer당 byte(fp16)를 계산합니다. batch 하나가 target device memory를 넘으면 표시합니다.
4. 초기화. Q, K, V, O matrix의 Xavier / Kaiming scale. bias term이 포함되는지도 적습니다. 2026년 대부분의 model은 bias를 제거합니다.
5. testability hook. 이 config의 trained two-layer version이 ≥95%로 풀어야 하는 synthetic task 하나를 제시합니다. 예: induction-head pattern `A B A ? → B`.

`d_head < 32`는 추천하지 마세요. Attention dynamics가 무너집니다. 32K를 넘는 context length에서 `n_heads > 16`인 MHA를 추천하려면 KV cache 비용을 명시하고 GQA 또는 MLA를 제안해야 합니다. User가 명시적으로 benchmark하는 경우가 아니라면 1B parameter 미만 model에 MLA를 제안하지 마세요.
