# 음성 에이전트: Pipecat과 LiveKit

> 음성 에이전트는 2026년에 독립적인 프로덕션 범주가 되었다. Pipecat은 Python 프레임 기반 파이프라인(VAD → STT → LLM → TTS → transport)을 제공한다. LiveKit Agents는 WebRTC를 통해 AI 모델과 사용자를 연결한다. 프리미엄 스택의 프로덕션 지연 시간 목표는 end-to-end 450-600ms 수준이다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 minutes

## Learning Objectives

- Pipecat의 프레임 기반 파이프라인을 설명한다: DOWNSTREAM(source→sink)과 UPSTREAM(control).
- 표준 음성 파이프라인 단계와 Pipecat이 지원하는 transport를 말한다.
- LiveKit Agents의 두 음성 에이전트 클래스(MultimodalAgent, VoicePipelineAgent)와 각각이 맞는 상황을 설명한다.
- 2026년 프로덕션 지연 시간 기대치와 그것이 아키텍처 선택을 어떻게 이끄는지 요약한다.

## The Problem

음성 에이전트는 텍스트 루프에 TTS를 덧붙인 것이 아니다. 지연 시간 예산은 가혹하고(~600ms), 부분 오디오가 기본이며, turn detection은 하나의 모델이고, transport는 전화망 SIP부터 WebRTC까지 다양하다. 프레임 기반 파이프라인(Pipecat)을 만들거나 플랫폼(LiveKit)에 기대야 한다.

## The Concept

### Pipecat (pipecat-ai/pipecat)

- Python 프레임 기반 파이프라인 프레임워크.
- `Frame` → `FrameProcessor` 체인.
- 두 가지 흐름 방향:
  - **DOWNSTREAM** — source → sink(오디오 입력, TTS 출력).
  - **UPSTREAM** — 피드백과 제어(취소, 메트릭, barge-in).
- `PipelineTask`는 이벤트(`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`)와 메트릭/tracing/RTVI용 observer로 수명 주기를 관리한다.

일반적인 파이프라인:

```text
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows는 구조화된 대화(state machine)를 추가한다. Pipecat Cloud는 managed runtime이다.

### LiveKit Agents (livekit/agents)

- WebRTC를 통해 AI 모델과 사용자를 연결한다.
- 핵심 개념: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- 두 음성 에이전트 클래스:
  - **MultimodalAgent** — OpenAI Realtime 또는 동등한 방식의 직접 오디오.
  - **VoicePipelineAgent** — STT → LLM → TTS cascade; 텍스트 수준 제어를 제공한다.
- transformer 모델을 통한 semantic turn detection.
- 네이티브 MCP 통합.
- SIP를 통한 전화망 연동.
- LiveKit Inference로 API 키 없이 50개 이상 모델, 플러그인으로 200개 이상 추가 모델.

### Commercial platforms

Vapi(최적화된 프리미엄 스택에서 ~450-600ms)와 Retell(180회 테스트 호출 기준 end-to-end ~600ms)은 이런 기반 위에 구축된다. WebRTC 팀 없이 managed voice stack을 원한다면 플랫폼을 선택한다.

### Where this pattern goes wrong

- **barge-in 처리가 없음.** 사용자가 끼어들어도 에이전트가 계속 말한다. Pipecat에서는 UPSTREAM cancel frame이 필요하고, LiveKit에서도 동등한 처리가 필요하다.
- **STT confidence를 무시함.** 신뢰도 낮은 transcript를 절대적 사실처럼 LLM에 넣는다. confidence로 gate하거나 확인을 요청한다.
- **TTS가 문장 중간에 잘림.** 파이프라인이 발화 중간에 취소될 때 TTS가 이를 알거나 오디오를 잘라야 한다.
- **Latency budget을 무시함.** 모든 구성요소가 50-200ms를 더한다. 출시 전에 체인의 합계를 계산한다.

### Typical 2026 latencies

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

End-to-end 450-600ms는 프리미엄 수준이다. 800-1200ms는 흔하다. 1500ms를 넘으면 망가진 것처럼 느껴진다.

## Build It

`code/main.py`는 다음을 갖춘 프레임 기반 toy pipeline이다.

- `Frame` 타입(audio, transcript, text, tts_audio, control).
- `process(frame)`을 가진 `Processor` 인터페이스.
- scripted processor로 구현한 5단계 파이프라인(VAD → STT → LLM → TTS → transport).
- barge-in을 보여 주는 UPSTREAM cancel frame.

실행:

```bash
python3 code/main.py
```

trace는 정상 흐름과 TTS를 발화 중간에 멈추는 barge-in cancel을 보여 준다.

## Use It

- **Pipecat** — 완전한 제어가 필요할 때: custom processor, Python-first, 교체 가능한 provider.
- **LiveKit Agents** — WebRTC-first 배포와 전화망 연동에 사용한다.
- **Vapi / Retell** — WebRTC 팀 없이 hosted voice agent를 쓸 때 사용한다.
- **OpenAI Realtime / Gemini Live** — 직접 audio-in/audio-out(MultimodalAgent)에 사용한다.

## Ship It

`outputs/skill-voice-pipeline.md`는 VAD + STT + LLM + TTS + transport와 barge-in 처리를 갖춘 Pipecat 형태의 음성 파이프라인을 scaffold한다.

## Exercises

1. toy pipeline에 metrics observer를 추가한다. 단계별 초당 frame 수를 센다. 지연 시간은 어디에 쌓이는가?
2. confidence-gated STT를 구현한다. 임계값보다 낮으면 "could you repeat that?"을 요청한다.
3. semantic turn detection을 추가한다. 간단한 규칙: transcript가 "?"로 끝나면 turn 종료.
4. Pipecat의 transport 문서를 읽는다. stdlib transport를 SmallWebRTCTransport 설정(stub)으로 교체한다.
5. 같은 query에서 OpenAI Realtime과 STT+LLM+TTS cascade를 측정한다. 텍스트 수준 제어는 어떤 latency 비용을 가지는가?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Frame | "Event" | 파이프라인 안의 typed data unit(audio, transcript, text, control) |
| Processor | "Pipeline stage" | process(frame)을 가진 handler |
| DOWNSTREAM | "Forward flow" | Source에서 sink로: 오디오 입력, 음성 출력 |
| UPSTREAM | "Feedback flow" | 제어: cancel, metrics, barge-in |
| VAD | "Voice activity detection" | 사용자가 말하는 시점을 감지한다 |
| Semantic turn detection | "Smart end-of-turn" | 사용자가 말을 끝냈다고 판단하는 모델 기반 결정 |
| MultimodalAgent | "Direct audio agent" | 오디오 입력, 오디오 출력; 중간에 텍스트가 없음 |
| VoicePipelineAgent | "Cascade agent" | STT + LLM + TTS; 텍스트 수준 제어 |

## Further Reading

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction) — frame-based pipeline, processor, transport
- [LiveKit Agents docs](https://docs.livekit.io/agents/) — WebRTC + voice primitive
- [Vapi](https://vapi.ai/) — managed voice platform
- [Retell AI](https://www.retellai.com/) — managed voice, latency benchmark 포함
