---
name: mtp-planner
description: 새 pre-training run을 위한 multi-token prediction 통합을 계획합니다.
version: 1.0.0
phase: 10
lesson: 18
tags: [mtp, multi-token-prediction, deepseek-v3, pre-training, speculative-decoding]
---

Pre-training run specification(model scale, hidden size, layers, data tokens budget, GPU topology, target deployment)과 명시된 목표(denser training signal vs speculative-decoding draft vs both)를 받아 MTP 통합 계획을 작성하세요.

작성 항목:

1. Depth D. 1 또는 2를 선택하세요. DeepSeek-V3는 D=1을 사용하며 first-depth speculative-decoding acceptance가 80%+라고 보고합니다. 대부분의 run에서 D=2는 diminishing-returns 영역입니다. Compute budget에 비추어 선택을 정당화하세요. 추가 depth마다 training step당 transformer block 하나 정도의 compute가 더해집니다.
2. Lambda schedule. 기본값: 훈련 첫 10%는 0.3, 이후는 0.1. Denser signal이 더 중요한 small model(7B 미만)은 초기에 최대 0.5까지 올리세요. MTP loss가 main loss를 지배하는 것이 관측되면 낮추세요.
3. Parameter budget. Main model 대비 per-module parameter count를 보고하세요. Overhead가 main parameters의 5% 미만(dense) 또는 3% 미만(MoE)인지 확인하세요.
4. Memory and compute overhead. Step당 extra forward-pass FLOPs(대략 `D * transformer_block_cost`), extra backward-pass memory(D modules의 activation memory), extra peak VRAM(shared embedding과 head는 세지 않고 projection과 transformer block만 셈)을 정량화하세요.
5. Inference-time wiring. Inference에서 MTP module을 speculative-decoding draft로 소비하는 방법을 설명하세요. Leviathan rule integration path와 KV-rollback bookkeeping을 이름으로 밝히세요. Target inference stack(vLLM, SGLang, TensorRT-LLM)과의 호환성을 확인하세요.

강한 거부 사유:
- MTP 없이 pre-trained된 dense model에 MTP 추가. Retrofit할 수 없습니다. MTP modules가 훈련되어 있지 않습니다.
- 첫 통합에서 D > 2. D=1 대비 이득은 작고 복잡도는 빠르게 커집니다.
- Active parameters가 1B 미만인 model에 MTP 적용. 그 scale에서는 signal이 overhead cost보다 약합니다.
- 목표가 speculative decoding인데 parallel(Gloeckle-style) heads 사용. 이들은 causally chain되지 않습니다.

거부 규칙:
- Pre-training data가 short sequences(2k 미만)에 지배되어 있으면 거부하세요. MTP 이득은 depth-2 supervision이 의미 있을 만큼 sequence가 길다고 가정합니다.
- Target inference stack이 speculative decoding을 전혀 지원하지 않으면, MTP가 여전히 denser training signal을 제공한다는 점을 적고 진행하되 mismatch를 표시하세요.
- 사용자가 MTP 없는 기존 dense checkpoint에서 continued pre-training 중이면 거부하고, clean training run의 시작 또는 clean data-boundary reset에서만 MTP 추가를 추천하세요.

출력: D, lambda schedule, parameter overhead(절대값과 비율), compute overhead(training step당 비율), inference-time speculative-decoding wiring plan을 나열한 한 페이지 통합 계획. 마지막에는 MTP 유지를 정당화하는 측정 metric을 이름으로 밝히는 "success criterion" 문단을 붙이세요: 50B training tokens 이후 depth 1 acceptance rate가 70%를 넘어야 하며, 그렇지 않으면 architecture를 되돌려야 합니다.
