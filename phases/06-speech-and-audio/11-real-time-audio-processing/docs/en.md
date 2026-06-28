# 실시간 오디오 처리

> Batch pipeline은 파일 하나를 처리한다. Real-time pipeline은 다음 20밀리초가 도착하기 전에 현재 20밀리초를 처리한다. 모든 conversational AI, 방송 스튜디오, telephony bot의 성패는 이 latency budget에 달려 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 문제

살아 있는 느낌의 voice assistant를 만들고 싶다. 사람의 대화 turn-taking latency는 약 230 ms(침묵에서 응답까지)다. 500 ms를 넘으면 기계적으로 느껴지고, 1500 ms를 넘으면 망가진 것처럼 느껴진다. 2026년 기준 완전한 **듣기 → 이해하기 → 응답하기 → 말하기** loop의 budget은 다음과 같다.

| 단계 | 예산 |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi(Kyutai, 2024)는 full-duplex 200 ms를 기록했다. GPT-4o-realtime(2024)은 약 320 ms다. 2022년에 출시된 cascaded pipeline은 2500 ms였다. 10배 개선은 세 가지 기법에서 왔다. (1) 모든 곳의 streaming, (2) partial result를 이용한 asynchronous pipelining, (3) interruptible generation.

## 개념

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.** Real-time audio는 고정 크기 block으로 흐른다. 흔한 선택은 20 ms(16 kHz에서 320 sample)다. downstream의 모든 단계는 이 cadence를 따라잡아야 한다.

**Ring buffer.** 고정 크기 circular buffer. producer thread가 새 frame을 쓰고 consumer thread가 읽는다. hot path에서 allocation을 막는다. 크기 ≈ maximum-latency × sample-rate; 2초 16 kHz ring = 32,000 sample.

**VAD (Voice Activity Detection).** 아무도 말하지 않을 때 downstream work를 gate한다. Silero VAD 4.0(2024)은 CPU에서 30 ms frame당 <1 ms로 실행된다. `webrtcvad`는 더 오래된 대안이다.

**Streaming ASR.** 오디오가 도착하는 동안 partial transcript를 내보내는 모델. streaming mode의 Parakeet-CTC-0.6B(NeMo, 2024)는 320 ms latency에서 2-5% WER를 낸다. Whisper-Streaming(Macháček et al., 2023)은 Whisper를 chunk로 나눠 약 2 s latency의 near-streaming을 만든다.

**Interruption.** assistant가 말하는 동안 사용자가 말하면 반드시 (a) barge-in을 감지하고, (b) TTS를 멈추고, (c) 남은 LLM output을 버려야 한다. 모두 100 ms 안에 해야 한다. 그렇지 않으면 사용자는 귀먹은 assistant처럼 느낀다.

**WebRTC Opus transport.** 20 ms frame, 48 kHz, adaptive bitrate 8-128 kbps. browser와 mobile의 표준이다. LiveKit, Daily.co, Pion은 2026년 voice app 구축 stack이다.

**Jitter buffer.** Network packet은 순서가 바뀌거나 늦게 도착한다. jitter buffer가 재정렬하고 smoothing한다. 너무 작으면 들리는 gap이 생기고, 너무 크면 latency가 커진다. 일반적으로 60-80 ms다.

### 흔한 함정

- **Thread contention.** Python의 GIL + 무거운 모델은 audio thread를 굶길 수 있다. C-callback audio library(sounddevice, PortAudio)를 사용하고 Python을 hot path 밖에 둔다.
- **Sample-rate conversion latency.** pipeline 안에서 resampling하면 5-20 ms가 추가된다. 앞에서 미리 resample하거나 zero-latency resampler(PolyPhase, `soxr_hq`)를 사용한다.
- **TTS priming.** Kokoro처럼 빠른 TTS도 첫 request에는 100-200 ms warm-up이 있다. 모델을 cache하고 첫 실제 turn 전에 dummy run으로 warm-up한다.
- **Echo cancellation.** AEC가 없으면 TTS output이 mic로 다시 들어가 bot 자신의 음성에 대해 ASR을 trigger한다. WebRTC AEC3가 오픈소스 기본값이다.

```figure
nyquist-aliasing
```

## 직접 만들기

### 1단계: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Capacity는 최대 buffering latency를 결정한다. 16 kHz에서 32,000 sample = 2 s다.

### 2단계: VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Production에서는 Silero VAD로 대체한다.

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 3단계: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 4단계: interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

핵심은 async I/O와 취소 가능한 TTS streaming이다. WebRTC peerconnection.stop()을 audio track에 호출하는 것이 정석이다.

## 활용하기

2026년 stack:

| 계층 | 선택 |
|-------|------|
| Transport | LiveKit (WebRTC) 또는 Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 또는 Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro 또는 ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API 또는 Moshi |

## 함정

- **안전하다고 500 ms buffering하기.** buffer 자체가 latency floor다. 줄여라.
- **Thread를 pin하지 않기.** UI보다 낮은 priority thread에서 audio callback이 돌면 load 아래에서 glitch가 난다.
- **TTS chunk가 너무 작음.** 200 ms 미만 chunk는 vocoder artifact를 들리게 만든다. 320 ms chunk가 적절한 지점이다.
- **Jitter buffer 없음.** 실제 network는 jittery하다. smoothing이 없으면 pop이 생긴다.
- **Single-shot error handling.** Audio pipeline은 crash-proof여야 한다. exception 하나가 session을 죽인다.

## 출시하기

`outputs/skill-realtime-designer.md`로 저장하라. stage별 구체적인 latency budget이 있는 real-time audio pipeline을 설계한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하라. ring buffer + energy VAD를 simulate하고 가짜 10초 stream의 stage latency를 출력한다.
2. **중간.** `sounddevice`를 사용해 mic를 20 ms frame으로 처리하고 각 frame에서 VAD state를 출력하는 passthrough loop를 만들어라.
3. **어려움.** `aiortc`로 full duplex echo test를 만들어라: browser → WebRTC → Python → WebRTC → browser. 1 kHz pulse로 glass-to-glass latency를 측정하라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Ring buffer | Circular queue | audio frame용 고정 크기 lock-free(또는 SPSC-locked) FIFO. |
| VAD | Silence gate | speech vs non-speech를 표시하는 모델 또는 heuristic. |
| Streaming ASR | Real-time STT | 오디오가 도착하는 동안 partial text를 emit한다; bounded lookahead. |
| Jitter buffer | Network smoother | 순서가 뒤섞인 packet을 재정렬하는 queue; 보통 60-80 ms. |
| AEC | Echo cancellation | speaker-to-mic feedback path를 빼낸다. |
| Barge-in | User interrupt | TTS 중간 사용자 발화를 감지하고 playback을 취소해야 한다. |
| Full duplex | 양방향 동시 | user와 bot이 동시에 말할 수 있다; Moshi는 full duplex다. |

## 더 읽을거리

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743): chunked near-streaming Whisper.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf): full-duplex 200 ms latency.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/): production audio agent orchestration.
- [Silero VAD repo](https://github.com/snakers4/silero-vad): sub-1 ms VAD, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/): 오픈소스 echo cancellation.
