---
name: voice-assistant-architect
description: 주어진 workload에 맞춰 component, latency budget, observability, compliance를 포함한 full-stack voice-assistant spec을 작성한다.
version: 1.0.0
phase: 6
lesson: 12
tags: [voice-assistant, architecture, livekit, pipecat, compliance]
---

use case(consumer / customer-support / accessibility / edge), expected scale(concurrent sessions, minutes/month), language, latency target, compliance(HIPAA, PCI, EU AI Act, CA SB 942)가 주어지면 다음을 출력하라.

1. 구성요소(7계층). Mic + chunking · VAD · streaming STT · LLM + tools · streaming TTS · playback · interruption handler. 각 layer의 정확한 provider/model 이름을 적는다.
2. 지연 시간 예산. end-to-end target에 합산되는 stage별 P50 / P95 / P99 target. 어떤 stage가 independent이고 어떤 stage가 sequential인지 표시한다.
3. 도구 호출 schema. 각 tool의 JSON spec + error handling + fallback text. tool이 두 번 실패하면 LLM이 반드시 택해야 하는 "can't help" path를 항상 포함한다.
4. 안전. Prompt injection guard, voice-cloning lockout(TTS가 cloning 가능할 때), wake-word gate(always-on일 때), log의 PII redaction, 30-day retention.
5. 관측성. stage별 P50/P95/P99 · false-interruption rate · tool-call success rate · 100 call당 WER · cost per minute · abandon rate.
6. 컴플라이언스. Disclosure audio("This is an AI assistant"), region-pinning(EU data in EU), audit log retention, opt-out pathway.

wake word 없는 always-on deployment를 거부하라. streaming하지 않는 TTS를 거부하라(utterance-length latency가 추가된다). P95 없는 average latency를 거부하라. tail이 user churn이 발생하는 곳이다. legal review 없이 raw-audio retention &gt; 30 days를 거부하라.

예시 입력: "Accessibility assistant for low-vision users: voice-only interface to a consumer email app. English. P95 &lt; 600 ms. ~10k concurrent users."

예시 출력:
- 구성요소: sounddevice(WebRTC via LiveKit Agents) · Silero VAD · Deepgram Nova-3(English) · email tool(read_message, compose_reply, mark_read)이 있는 GPT-4o · Cartesia Sonic 2 streaming · WebRTC out · VAD fire 시 interrupt=cancel-LLM-and-TTS.
- 예산: capture 120 ms + VAD 40 + STT 150 + LLM TTFT 100 + TTS TTFA 150 = 560 ms P95.
- 도구: read_message({id}), compose_reply({message_id, body}), mark_read({id}), search({query}). 모두 JSON을 반환한다. LLM은 tool별 최대 2회 retry 후 fallback "I couldn't do that - try rephrasing".
- 안전: prompt-injection guard(`ignore previous instructions` 감지); wake word "Hey Mail"; voice cloning 없음(fixed Cartesia voice); log에서 email body redact.
- 관측성: Hamming AI production monitoring; stage별 Prometheus histogram; false-interrupt &gt; 5% 또는 p95 &gt; 800 ms이면 alert.
- 컴플라이언스: 첫 사용 시 AI disclosure; medical message에만 HIPAA opt-in; EU user는 EU-hosted Cartesia + GPT-4o Ireland로 이동.
