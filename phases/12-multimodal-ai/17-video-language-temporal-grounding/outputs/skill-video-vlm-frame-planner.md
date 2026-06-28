---
name: video-vlm-frame-planner
description: video-language model deployment를 위한 frame sampling, per-frame pooling, output format, benchmark target을 계획한다.
version: 1.0.0
phase: 12
lesson: 17
tags: [video-vlm, temporal-grounding, tmrope, dynamic-fps, benchmarks]
---

video task(action recognition, temporal grounding, summarization, monitoring, agent-workflow replay)와 deployment constraint(model context, latency budget, throughput)가 주어지면 frame sampling과 output plan을 만든다.

생성할 것:

1. Frame sampler pick. steady content에는 uniform, mixed motion에는 dynamic-FPS, action-heavy에는 event-driven, cinematic에는 keyframe+context.
2. Per-frame pooling. high-detail에는 2x2, 기본값은 3x3, coverage가 content density보다 중요한 agent workflow에는 4x4 또는 6x6.
3. Temporal encoding. Qwen2.5-VL-family에는 TMRoPE, 작은 모델에는 learned temporal embedding, single-clip task에는 encoding 없음.
4. 출력 형식. grounding에는 `{event, start, end, confidence}` JSON, summarization에는 free text, mixed flow에는 token-delimited.
5. Benchmark plan. general에는 VideoMME, grounding에는 TempCompass, long-horizon에는 EgoSchema. expected accuracy tier를 명시한다.
6. Context / latency budget. 총 token = duration * fps * tokens_per_frame. context의 40%를 넘으면 경고한다.

강한 거절:
- action-heavy video에 uniform sampling을 제안하는 것. peak event를 놓친다.
- downstream parsing에서 token-delimited output이 JSON accuracy와 같다고 주장하는 것. JSON이 더 robust하다.
- 2026년에 시작하는 project에 Video-LLaMA를 추천하는 것. 오래된 architecture는 더 이상 경쟁력이 없다.

거절 규칙:
- duration > 10 minutes이고 context < 32k이면 거절하고 hierarchical summarization 또는 agentic retrieval(Lesson 12.18)을 권한다.
- target accuracy가 frontier(VideoMME에서 Gemini 2.5 Pro와 2 point 이내)이면 open 7B model을 거절하고 32B+ 또는 proprietary를 요구한다.
- 30s 초과 clip에서 7B 기준 dynamic-FPS target이 > 8이면 latency 관점에서 거절하고 더 낮은 cap을 권한다.

출력: sampler, pooling, temporal encoding, output format, benchmark target, context estimate를 담은 one-page frame plan. 마지막에 비교 읽기용 arXiv 2502.13923(Qwen2.5-VL)과 2306.02858(Video-LLaMA)를 붙인다.
