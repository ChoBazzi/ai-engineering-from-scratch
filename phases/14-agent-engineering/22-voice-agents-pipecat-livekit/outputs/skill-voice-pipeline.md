---
name: voice-pipeline
description: barge-in, confidence gating, latency budget enforcement를 갖춘 Pipecat 형태의 음성 파이프라인(VAD + STT + LLM + TTS + transport)을 scaffold한다.
version: 1.0.0
phase: 14
lesson: 22
tags: [voice, pipecat, livekit, webrtc, latency]
---

음성 제품 spec(language, transport, providers)이 주어지면 frame-based pipeline을 scaffold한다.

생성할 것:

1. `kind`, `payload`, `direction`(downstream / upstream)을 가진 `Frame` type.
2. Processor: `VAD`, `STT`, `LLM`, `TTS`, `Transport`. 각각 `process(frame)`을 가진다.
3. processor를 앞뒤로 chain하는 `link()` helper.
4. Cancel frame handling: transport에서 TTS, LLM, STT로 가는 UPSTREAM path를 두고 각 단계에서 pending work를 버린다.
5. Observer: 단계별 latency metric. processor를 지나는 frame마다 OTel span을 emit한다(Lesson 23).
6. STT confidence gate: 임계값보다 낮으면 transcript 대신 "please repeat" text frame을 emit한다.

강한 거부 조건:

- UPSTREAM handling이 없는 pipeline. 음성에서 barge-in은 선택 사항이 아니다.
- streaming 없는 LLM call. first-token latency가 지배적이므로 반드시 stream해야 한다.
- confidence-blind STT. 잘못된 transcript를 LLM에 넣으면 잘못된 reply가 나온다.

거부 규칙:

- cold run에서 end-to-end latency가 1500ms를 넘으면 출시를 거부한다. chain을 최적화하거나 MultimodalAgent(LiveKit direct-audio)를 사용한다.
- 제품이 telephony-first인데 pipeline에 SIP adapter가 없으면 거부한다. LiveKit SIP 또는 플랫폼(Vapi/Retell)을 통해 route한다.
- 제품이 전송 중 암호화 없이 PII audio를 운반하면 거부한다.

출력: latency budget, barge-in design, transport choice를 설명하는 `frames.py`, `processors.py`, `pipeline.py`, `observers.py`, `README.md`. 마지막은 Lesson 23(OTel), Lesson 24(observability backend), 또는 WebRTC 세부사항을 위한 LiveKit 문서를 가리키는 "what to read next"로 끝낸다.
