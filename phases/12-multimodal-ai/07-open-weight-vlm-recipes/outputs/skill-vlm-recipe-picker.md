---
name: vlm-recipe-picker
description: 모든 선택에 ablation-table citation을 붙여 open-weight VLM recipe(encoder, connector, LLM, data mix, resolution schedule)를 고른다.
version: 1.0.0
phase: 12
lesson: 07
tags: [vlm, mm1, idefics2, molmo, cambrian, prismatic, ablation]
---

task mix(OCR, chart, UI agent, reasoning, grounding), compute budget(LLM params, training GPU hour, 또는 inference latency target), deployment constraint(edge, cloud, on-device)가 주어지면 citation이 포함된 full open-weight VLM recipe를 출력한다.

생성할 것:

1. Encoder pick. 기본은 SigLIP 2 SO400m/14다. task mix에 grounding/segmentation이 있으면 DINOv2 ViT-g/14와 concat한다. MM1 Table 3과 Cambrian-1의 vision encoder match-up을 인용한다.
2. Connector pick. token-constrained가 아니면 기본은 2-layer MLP다. token-constrained이면 Q-Former 32 query를 쓴다. <1 point delta를 보인 Prismatic VLMs의 connector ablation을 인용한다.
3. LLM pick. budget 기반으로 고른다. <10B에는 Qwen2.5-7B, >30B에는 Llama-3.1-70B 또는 Qwen2.5-72B. 70B 이후 MMMU plateau를 flag한다.
4. Data mix. 기본은 PixMo + ShareGPT4V + Cauldron이다. Molmo의 detailed-human-caption 결과를 인용한다(같은 token count에서 distillation 대비 +2-3 MMMU).
5. Resolution schedule. 기본은 dynamic(256-1280)이며 stage-1 fixed-384 alignment pretraining을 둔다. Idefics2 resolution ablation(AnyRes에서 +3-5 DocVQA)과 Qwen2.5-VL dynamic M-RoPE를 인용한다.
6. Training stages. Stage 1 projector-only, Stage 2 full fine-tune, Stage 3 task-specific.

강한 거절:
- 새 project의 default encoder로 CLIP ViT-L/14를 권장하면서 SigLIP 2가 이를 대체했다는 점을 flag하지 않는 것.
- Q-Former를 MLP 대비 quality gain으로 제안하는 것. Q-Former는 token-budget lever이지 quality lever가 아니다.
- human-captioned alternative가 있는데 synthetic GPT-4V caption을 primary training data로 제안하는 것. Molmo를 인용하라.
- 실제로는 token count에서 오는 variance를 connector architecture가 설명한다고 주장하는 것.

거절 규칙:
- 사용자가 reasoning-heavy task에 1-3B VLM을 원하면 거절하고 더 큰 LLM을 권장한다. reasoning ceiling은 LLM이 정한다.
- 사용자가 detailed-human-caption data 비용을 감당할 수 없으면 예상되는 2-3 MMMU ceiling을 명시적으로 flag하고 best-effort distillation fallback을 제안한다.
- task mix에 4K+ document imagery가 있고 frozen-encoder deployment라면 AnyRes를 거절하고 Qwen2.5-VL 같은 native-resolution M-RoPE encoder를 권장한다.

출력: axis별 pick, ablation citation(arXiv ID), training stage plan, expected benchmark range가 포함된 one-page recipe card. 다음에 읽을 세 ablation paper로 끝낸다. arXiv 2403.09611(MM1), 2405.02246(Idefics2), 2409.17146(Molmo).
