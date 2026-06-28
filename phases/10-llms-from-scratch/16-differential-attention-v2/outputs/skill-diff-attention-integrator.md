---
name: diff-attention-integrator
description: 새 pre-training run 또는 LoRA fine-tune에 Differential Attention V2를 추가하기 위한 통합 계획입니다.
version: 1.0.0
phase: 10
lesson: 16
tags: [differential-attention, diff-transformer, long-context, flash-attention, pre-training, lora]
---

모델 architecture(hidden, heads, KV heads, layers, d_head), 목표 context length, hallucination 또는 long-context profile(기존 eval의 failure mode), training budget(사용 가능한 tokens, GPU-hours)을 받아 DIFF V2 통합 계획을 작성하세요.

작성 항목:

1. 통합 모드. From-scratch pre-training, mid-training architecture swap, Q projection LoRA fine-tune 중 선택하세요. Training budget과 사용 가능한 기존 weights에 비추어 선택을 정당화하세요.
2. Architecture diff. Projection 중 무엇이 커지고 무엇이 그대로인지, 추가되는 parameter count, subtraction이 attention block의 어디에 배치되는지를 field별 구체적 변경 목록으로 쓰세요. Layer depth별 `lambda_init` schedule을 포함하세요(`0.8 - 0.6 * exp(-0.3 * (depth - 1))`가 논문의 기본값입니다. layerwise telemetry가 불안정을 보이면 depth별로 조정하세요).
3. Kernel choice. V2의 head-count doubling에서 FlashAttention 2 또는 3 지원을 확인하세요. 사용자가 재현성을 위해 명시적으로 필요로 하지 않는 한 V1의 custom-kernel 경로는 거부하세요.
4. Memory budget. KV cache는 baseline과 같습니다(KV heads unchanged). Per-token activation memory delta(extra Q heads, extra compute)를 계산하세요. 목표 context에서의 절대 수치를 보고하세요.
5. Training stability plan. 모니터링할 항목을 설명하세요: layer별 `lambda` drift, head별 attention entropy, Q projection의 gradient variance. Telemetry가 divergence를 나타낼 때 baseline attention으로 롤백을 트리거해야 하는 구체적인 metric을 이름으로 밝히세요.

강한 거부 사유:
- Continued pre-training 없이 pre-trained model에 DIFF attention 추가. 출력 분포가 drift합니다. Drop-in fix가 아닙니다.
- 2026년 4월 이후 새 run에 DIFF V1 사용. V2가 측정된 모든 차원에서 엄격히 더 낫습니다.
- Long-context training data를 함께 활성화하지 않고 DIFF 통합. 이점은 32k 이후에만 나타납니다.
- Controlled experiment 없이 `lambda_init`을 음수로 변경. Negative init은 noise floor보다 더 많이 빼서 훈련을 붕괴시킵니다.

거부 규칙:
- Target context가 16k 미만이면 통합을 거부하고 standard attention을 추천하세요. 추가 parameter cost가 noise-floor 논리로 정당화되지 않습니다.
- 사용자가 long-context evaluation data(RULER, needle-in-haystack, MultiNeedle)를 제공할 수 없으면 거부하고 calibration data를 먼저 요청하세요.
- 사용자가 pre-FlashAttention-2 stack에 있으면 거부하고 통합을 시도하기 전에 stack upgrade를 추천하세요.

출력: mode, param count delta, KV cache impact, FlashAttention confirmation, `lambda` schedule, 3-metric monitoring board를 나열한 한 페이지 통합 계획. 마지막에는 DIFF V2를 architecture에 유지할지 되돌릴지 정당화할 구체적 long-context eval 수치(RULER 64k 또는 동등 지표의 percentage point delta)를 이름으로 밝히는 "success criterion" 문단을 붙이세요.
