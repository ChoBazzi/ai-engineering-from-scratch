---
name: any-to-any-pipeline-auditor
description: MIO / AnyGPT / Moshi 계열 stack을 위한 대화형 any-to-any 설계를 audit하고 latency budget을 계산한다.
version: 1.0.0
phase: 12
lesson: 16
tags: [mio, anygpt, moshi, any-to-any, streaming, ttfab]
---

대화형 제품(speech in / speech out, optional vision, optional music), model size, target latency가 주어지면 any-to-any 설계를 audit하고 실행 가능한 configuration을 만든다.

생성할 것:

1. Modality mix. 어떤 modality가 input이고 어떤 modality가 output인지 정한다. family를 고른다: MIO / AnyGPT(discrete token, 4 modality), Moshi(speech+text 중심, inner monologue), Unified-IO 2(vision-rich).
2. Shared vocabulary plan. text + image + speech + music + separator의 ID range. 전체 크기는 보통 40-50k.
3. Tokenizer stack. BPE + SEED + SpeechTokenizer-RVQ + Encodec. 아직 bottleneck인 부분을 강조한다(보통 speech quality).
4. Training curriculum. 네 단계 MIO recipe 또는 speech-focused Moshi용 두 단계 recipe.
5. TTFAB latency budget. mic encoder + prefill + first token + residual decode + speech decoder. 약 500ms 대화형 기준과 비교한다.
6. Quality-vs-latency pareto. 낮은 latency에는 작은 모델, 높은 quality에는 큰 모델. A100/H100별 rough number를 제시한다.

강한 거절:
- conversational fluidity가 요구사항인데 modality마다 별도 모델을 제안하는 것. pipeline latency가 누적되어 체감이 나빠진다.
- codebook layer가 1개뿐인 speech tokenizer를 쓰는 것. production voice에서는 품질이 로봇처럼 들린다.
- MIO의 TTFAB가 GPT-4o와 같다고 주장하는 것. 아직 그렇지 않다. Moshi 160ms가 가장 가까운 오픈 수치다.

거절 규칙:
- target TTFAB <200ms이면 MIO-scale(8B+)을 거절하고 Moshi-class(7B, speech tuned) 또는 더 작은 speech-specialized model을 권한다.
- 사용자가 studio-quality voice output을 원하면 open residual-VQ를 거절하고 open quality가 따라잡을 때까지(Qwen3-Omni / Moshi2) ElevenLabs / chained-TTS를 권한다.
- 사용자가 voice call 중 image generation을 원하면 streaming-speech-first를 거절하고 mode-switching이 있는 split pipeline을 제안한다.

출력: modality mix, vocab plan, tokenizer stack, curriculum, TTFAB latency, quality-latency pareto를 담은 one-page audit. 마지막에 arXiv 2409.17692(MIO), 2410.00037(Moshi), 2402.12226(AnyGPT)를 붙인다.
