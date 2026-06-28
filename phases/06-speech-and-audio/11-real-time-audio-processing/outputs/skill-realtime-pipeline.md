---
name: realtime-voice-pipeline
description: 목표 end-to-end latency에 맞춰 transport, VAD, streaming STT, LLM, streaming TTS, orchestration을 선택한다.
version: 1.0.0
phase: 6
lesson: 11
tags: [voice-agent, livekit, pipecat, silero, streaming, latency]
---

target(latency P50/P95, language, channel, offline vs cloud, call volume)이 주어지면 다음을 출력하라.

1. 전송. WebRTC(LiveKit / Daily) · WebSocket · SIP trunking(Twilio / Telnyx). jitter tolerance + use case와 연결된 이유.
2. VAD + 턴 처리. Silero VAD(open, 99.5% TPR) · Cobra(commercial) · LiveKit turn-detector. Threshold, min speech duration, silence hang-over.
3. Streaming STT. Parakeet TDT(가장 빠른 오픈) · Kyutai STT(flush trick 포함) · Deepgram Nova-3(API, ~150 ms) · Whisper-streaming. 이유.
4. LLM + streaming. TTS가 시작되기 전에 첫 20 token을 pin한다. Model + streaming config + prompt injection guardrail.
5. Streaming TTS. Kokoro-82M(~100 ms TTFA) · Orpheus · Cartesia Sonic · ElevenLabs Turbo. Voice-pack 또는 cloning guard(Lesson 8).
6. 오케스트레이션. LiveKit Agents · Pipecat · Vapi · Retell · custom Rust. team skill + scale과 연결된 이유.
7. 관측성. stage별 P50/P95/P99 histogram; false-positive interruption rate; drop-call rate; call sample의 WER.

STT 전에 전체 utterance를 buffer하는 deploy는 거부하라. streaming하지 않는 TTS를 거부하라. average latency로 평가하는 것을 거부하고 P95를 요구하라. &gt; 100k minutes/month에서는 build-your-own과 cost comparison 없이 managed platform(Vapi / Retell)을 거부하라.

예시 입력: "Voice agent for car insurance quoting. &lt; 500 ms P95. English, US. 50k minutes/week. Compliance: HIPAA-adjacent (no PII in logs)."

예시 출력:
- 전송: LiveKit Agents + Twilio SIP. call-center scale에서 검증되었고 HIPAA-mode opt-in 가능.
- VAD: Silero VAD @ threshold 0.45, min speech 220 ms, silence hang-over 400 ms. LiveKit turn-detector overlay.
- STT: Deepgram Nova-3 English(~150 ms P95); on-prem audit가 필요하면 Parakeet-TDT로 fallback.
- LLM: OpenAI realtime API를 통한 GPT-4o streaming; post-filter로 prompt injection 방어; 첫 20 token을 TTS에 pin.
- TTS: Cartesia Sonic 2(~150 ms TTFA, voice cloning 미사용, predefined voice).
- 오케스트레이션: LiveKit Agents. Production observability는 Hamming AI.
- 로그: persistence 전에 regex + NER pass로 CVV / SSN / DOB를 제거한다. 30일 보관.
