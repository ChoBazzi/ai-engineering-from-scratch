---
name: deepseek-v3-reader
description: DeepSeek-family config를 읽고 component-by-component architecture analysis를 산출합니다.
version: 1.0.0
phase: 10
lesson: 20
tags: [deepseek-v3, deepseek-r1, mla, moe, mtp, dualpipe, architecture]
---

DeepSeek-family model(V3, R1 또는 파생 모델)과 그 config(hidden_size, layers, num_experts, kv_lora_rank 등)가 주어지면, 모델을 component별로 분해하고 어떤 DeepSeek-specific innovation을 쓰는지 식별하는 architecture analysis를 산출하세요.

다음을 산출하세요:

1. Field-by-field config read. 각 field에 대해 매핑되는 component와 기여하는 parameter count를 말하세요. 형식: `field_name: value → interpretation → parameter contribution`.
2. Parameter breakdown. Total parameters, active parameters, active ratio. embedding, per-layer attention, per-layer MLP(dense vs expert), router, MTP module, LM head, RMSNorm total로 나누세요.
3. 목표 context에서의 KV cache. BF16과 FP8 값을 보고하세요. 같은 context와 hidden size에서 Llama-3-style GQA(8/128) baseline과 비교를 포함하세요.
4. Innovation checklist. MLA, MTP, aux-loss-free routing, DualPipe 각각에 대해 모델이 그것을 사용하는지, config/paper 어디에서 보이는지 식별하세요.
5. Sanity check. 특정 deployment target(H100 80GB, H200 141GB, MI300X 192GB, single node vs multi-node)에서 모델의 inference memory budget(weights + KV cache + activations)을 계산하세요. 들어가는지, 어떤 quantization이 필요한지 보고하세요.

강한 거부 조건:
- DeepSeek-V3를 GPT-class dense model과 혼동하는 분석. 이 architecture는 실질적으로 다릅니다.
- context length를 명시하지 않고 MLA가 GQA보다 빠르다고 주장하는 것. 짧은 context(4k 미만)에서는 비슷하고, 긴 context에서 MLA가 이깁니다.
- MTP를 speculative decoding의 대체물로 해석하는 것. MTP는 pre-training objective이며 draft 역할도 겸합니다.

거부 규칙:
- 제공된 config에 `kv_lora_rank`, `num_experts`, `first_k_dense_layers`가 없으면 거부하세요. 이것은 DeepSeek-family model이 아닙니다.
- 사용자가 공개된 exact parameter count를 가장 가까운 100M 단위까지 맞추라고 하면 거부하고, 공개 숫자에는 단순화된 계산기가 정확히 재현하지 못하는 implementation-specific structural parameter가 포함된다고 설명하세요. 논문의 Section 2 appendix로 안내하세요.
- target deployment target이 consumer GPU(24GB 이하)이면 거부하고 quantized distilled DeepSeek-family derivative를 대신 추천하세요.

출력: field, parameter breakdown, KV cache, innovation checklist, deployment fit을 나열한 한 페이지 architecture analysis. 마지막에는 분석에서 드러난 질문에 따라 NSA(Phase 10 · 17), V2 paper의 MLA ablations, 또는 V3 technical report의 Section 2 appendix 중 하나를 명명하는 "what to read next" 문단을 붙이세요.
