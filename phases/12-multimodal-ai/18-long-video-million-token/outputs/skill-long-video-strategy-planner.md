---
name: long-video-strategy-planner
description: long-video understanding task에 brute-context, ring-attention, token-compression, agentic-retrieval 중 하나를 고르고 latency + recall expectation을 계산한다.
version: 1.0.0
phase: 12
lesson: 18
tags: [long-video, gemini, ring-attention, videoagent, retrieval]
---

video duration, query complexity(single event vs holistic summary), open vs closed constraint가 주어지면 long-video strategy를 고르고 config를 만든다.

생성할 것:

1. Strategy pick. Brute-context, ring-attention(LongVILA), token-compression(Video-XL), agentic-retrieval(VideoAgent).
2. Token budget. Duration * FPS * per-frame-tokens. LLM context를 넘으면 경고한다.
3. Expected recall. video-length percentile별 needle-in-a-haystack recall. 관련이 있으면 Gemini 1.5 report를 cite한다.
4. Latency. brute-context의 prefill time, agentic 방식의 retrieval + VLM latency.
5. Engineering path. 선택한 strategy의 code snippet scaffold.
6. Fallback plan. Hybrid: brute-context global summary + agentic local detail.

강한 거절:
- open 72B model에서 2시간 video에 brute-context를 제안하는 것. context에 들어가지 않는다.
- agentic retrieval이 항상 이긴다고 주장하는 것. holistic-summary question에서는 brute context에 진다.
- recall tax를 표시하지 않고 token compression을 추천하는 것.

거절 규칙:
- target이 90-minute video에서 frontier recall(>95%)이면 open-only option을 거절하고 Gemini 2.5 Pro를 권한다.
- 사용자가 tool-calling loop 비용을 감당할 수 없으면 agentic-retrieval을 거절하고 compressed brute-context를 제안한다.
- real-time(stream-as-it-plays)이 필요하면 retrieval은 너무 느리므로 거절하고 streaming Qwen2.5-VL을 권한다.

출력: strategy, budget, recall, latency, engineering path, fallback을 담은 one-page plan. 마지막에 비교용 arXiv 2403.05530(Gemini 1.5)과 2403.10517(VideoAgent)를 붙인다.
