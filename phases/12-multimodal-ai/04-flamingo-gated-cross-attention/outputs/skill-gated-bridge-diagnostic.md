---
name: gated-bridge-diagnostic
description: open VLM config에서 Flamingo-lineage design element를 식별하고 freezing / gating 문제를 진단합니다.
version: 1.0.0
phase: 12
lesson: 04
tags: [flamingo, idefics, openflamingo, gated-cross-attention, interleaved-inputs]
---

open VLM checkpoint와 그 config(layer structure, cross-attention schedule, gate parametrization, training recipe)가 주어지면, 어떤 Flamingo-lineage element를 사용하는지 식별하고 gating 설정 오류의 흔한 증상을 진단하세요.

생성할 내용:

1. Lineage checklist. (Perceiver resampler Y/N, gated cross-attn frequency M, tanh vs sigmoid gate, alpha init value, LLM freeze depth)의 존재 여부를 표시하세요.
2. Interleaved-input support. 모델이 기대하는 prompt format을 parse하고 multi-image, video, few-shot in-context prompting 지원 여부를 확인하거나 부정하세요.
3. Visual token budget. 이미지당 비용을 계산하세요: K latents x N cross-attn insertion points. 같은 이미지 수에서 BLIP-2-style single-input bridge와 비교하세요.
4. Gate diagnosis. Training-loss curve나 benchmark degradation이 주어지면 gate가 너무 빨리 열렸는지(text capability 손실), 너무 느리게 열렸는지(visual input 사용 실패), 또는 miscalibrated 상태인지(visual token이 보강이 아니라 경쟁함) 제안하세요.
5. Fix recipe. 구체적인 parameter fix: text가 degradation되었으면 alpha를 0에 가깝게 initialize하거나, gate parameter의 learning rate를 올리거나, 처음 N step 동안 gate를 freeze하세요.

강한 거절 기준:
- Resampler와 gate schedule을 확인하지 않고 어떤 open VLM이든 "a Flamingo"로 취급하는 경우. Idefics2는 resampler를 제거했으므로 한정 없이 Flamingo-lineage라고 부르는 것은 틀립니다.
- Zero init이 항상 training을 견딘다고 가정하는 경우. 일부 open reproduction은 초기 안정성을 더 빠른 convergence와 맞바꾸기 위해 작은 non-zero init을 사용합니다.
- Gated cross-attention이 모든 작업에서 single BLIP-2 bridge보다 엄격히 낫다고 주장하는 경우. 작은 LLM의 single-image VQA에서는 추가 cross-attn layer가 순수 비용입니다.

거절 규칙:
- Checkpoint의 training recipe가 공개되어 있지 않으면 거절하고 gate diagnosis에 gate schedule을 알아야 하는 이유를 설명하세요.
- 호출자가 Gemini 또는 Claude(proprietary)와 비교하라고 요청하면 거절하세요. 해당 gating mechanism은 공개되어 있지 않습니다.
- 범위의 VLM이 early-fusion model(Chameleon, Emu3)이면 거절하세요. Gating은 adapter-style VLM에만 적용됩니다.

출력: lineage checklist, interleaved-input capability matrix, token budget, gate diagnosis, concrete fix recipe를 포함한 한 페이지 diagnostic. 마지막에는 alternative projector approach는 Lesson 12.05(LLaVA), early-fusion escape hatch는 Lesson 12.11(Chameleon)을 가리키는 "what to read next" 문단으로 끝내세요.
