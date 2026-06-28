---
name: llava-vibes-eval
description: LLaVA-family VLM에서 10-prompt vibes-eval을 실행하고 사람이 읽기 쉬운 scorecard를 생성합니다.
version: 1.0.0
phase: 12
lesson: 05
tags: [llava, vlm, vibes-eval, instruction-tuning]
---

LLaVA-family VLM(LLaVA-1.5, LLaVA-NeXT, LLaVA-OneVision 또는 community fork)과 test image set이 주어지면 captioning, VQA, reasoning, refusal, format compliance를 포함하는 10-prompt smoke test를 실행하세요. Projector와 LLM이 제대로 연결되는지 확인하는 scorecard를 생성하세요.

생성할 내용:

1. Expected-behavior description이 있는 10개 prompt:
   - Captioning 3개(short, detailed, creative).
   - VQA 3개(counting, color, object presence).
   - Reasoning 2개(compare two regions, cause-and-effect).
   - Refusal 2개(private individual, PII-identifying).
2. Prompt별 score. 한 줄 근거와 함께 pass / partial / fail을 제시하세요.
3. Overall pattern diagnosis. Captioning은 통과하지만 VQA가 실패하면 stage-2 data mix를 의심하세요. Detailed captioning에서 hallucination이 보이면 ShareGPT4V-style data가 부족하다고 의심하세요. Refusal이 실패하면 safety-data gap을 표시하세요.
4. Resolution check. OCR이 필요한 prompt 하나를 336x336 base에서 실행하고 AnyRes에서 다시 실행해 delta를 기록하세요. Low-res failure는 예상 가능하지만, high-res failure는 AnyRes가 mis-configured되었음을 뜻합니다.
5. Suggested follow-up. 특정 category가 실패할 경우 호출자가 실행할 수 있는 구체적인 training-data addition 세 가지를 제시하세요.

강한 거절 기준:
- Vibes suite를 함께 실행하지 않고 benchmark number만으로 VLM을 scoring하는 경우. Benchmark는 조작될 수 있고, vibes는 실제 deployment readiness를 드러냅니다.
- Hallucination을 stylistic verbosity와 혼동하는 경우. 어떤 object가 만들어졌고 어떤 것은 단지 자세히 묘사되었는지 구체적으로 표시하세요.
- Final answer뿐 아니라 reasoning chain을 확인하지 않고 reasoning prompt를 pass로 주장하는 경우.

거절 규칙:
- 호출자가 API access 없이 proprietary VLM(Gemini, Claude, GPT-5V)을 vibes-eval하라고 요청하면 거절하세요. 이 test는 실제 inference가 필요합니다.
- Target use case가 medical diagnosis 또는 legal advice라면 거절하세요. Vibes-eval은 certification이 아니며 high-stakes domain에 사용해서는 안 됩니다.
- 이미지가 제공되지 않았으면 거절하세요. 이 test는 정의상 image-grounded입니다.

출력: 10개 row(prompt, image, expected, actual, pass/partial/fail), overall pattern diagnosis, 3개 항목 follow-up list가 있는 scorecard. 마지막에는 resolution 관련 failure는 Lesson 12.06(AnyRes), data-mixture tuning은 Lesson 12.07(ablations)을 가리키는 "what to read next" 문단으로 끝내세요.
