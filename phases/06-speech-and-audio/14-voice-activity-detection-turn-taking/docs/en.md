# 음성 활동 감지와 턴 처리 — Silero, Cobra, 그리고 Flush Trick

> 모든 음성 에이전트의 성패는 두 가지 결정에 달려 있다. 사용자가 지금 말하고 있는가, 그리고 말을 끝냈는가? VAD는 첫 번째 질문에 답한다. 턴 감지(VAD + 침묵 행오버 + 의미적 엔드포인트 모델)는 두 번째 질문에 답한다. 둘 중 하나라도 틀리면 어시스턴트는 사용자의 말을 끊거나, 끝없이 말하게 된다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 minutes

## 문제

음성 에이전트가 20ms 청크마다 내리는 서로 다른 세 가지 결정:

1. **이 프레임이 음성인가?** — VAD. 프레임 단위의 이진 판단이다.
2. **사용자가 새 발화를 시작했는가?** — 시작점 감지(onset detection)다.
3. **사용자가 말을 끝냈는가?** — 엔드포인팅(턴 종료)이다.

순진한 답(에너지 임계값)은 교통 소음, 키보드 소리, 군중 웅성거림 같은 어떤 잡음에서도 실패한다. 2026년의 답은 Silero VAD(공개, 딥러닝 기반) + 턴 감지 모델(의미적 엔드포인팅) + VAD로 보정한 침묵 행오버다.

## 개념

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 3단계 VAD 캐스케이드

**1단계: 에너지 게이트.** 가장 저렴하다. RMS를 -40 dBFS에서 임계값 처리한다. 명백한 침묵은 걸러내지만, 임계값보다 큰 모든 잡음에도 반응한다.

**2단계: Silero VAD** (2020-2026, MIT). 매개변수 100만 개. 6000개 이상의 언어로 학습했다. 단일 CPU 스레드에서 30ms 청크당 약 1ms에 실행된다. 5% FPR에서 87.7% TPR. 오픈소스 기본 선택지다.

**3단계: 의미적 턴 감지기.** LiveKit의 턴 감지 모델(2024-2026) 또는 직접 만든 작은 분류기다. "문장 중간의 멈춤"과 "말하기 완료"를 구분한다. 침묵만 보지 않고 언어적 맥락(억양 + 최근 단어)을 사용한다.

### 핵심 매개변수와 기본값

- **임계값.** Silero는 확률을 출력한다. &gt; 0.5(기본값) 또는 &gt; 0.3(민감)에서 음성으로 분류한다. 임계값이 낮을수록 첫 단어 잘림은 줄지만 false positive가 늘어난다.
- **최소 음성 길이.** 250ms보다 짧은 음성은 거부한다. 대개 기침이나 의자 소리다.
- **침묵 행오버(엔드포인팅).** VAD가 0으로 돌아간 뒤 턴 종료를 선언하기 전에 500-800ms 기다린다. 너무 짧으면 사용자를 방해한다. 너무 길면 둔하게 느껴진다.
- **프리롤 버퍼.** VAD가 켜지기 전 300-500ms의 오디오를 보관한다. "hey"가 잘리는 일을 막는다.

### flush trick (Kyutai 2025)

스트리밍 STT 모델에는 look-ahead 지연이 있다(Kyutai STT-1B는 500ms, STT-2.6B는 2.5초). 보통은 발화 종료 후 그만큼 기다려야 전사 결과를 얻는다. flush trick은 VAD가 발화 종료를 감지하는 순간 **STT에 flush 신호를 보내 즉시 출력을 강제**한다. STT는 실시간보다 약 4배 빠르게 처리하므로 500ms 버퍼가 약 125ms에 끝난다.

엔드투엔드로는 125ms VAD + flush STT = 대화형 지연 시간이다.

### 2026년 VAD 비교

| VAD | TPR @ 5% FPR | 지연 시간 | 라이선스 |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero가 올바른 기본값이다. Cobra는 규정 준수/정확도 업그레이드다. 에너지만 쓰는 VAD는 2026년 프로덕션에서 설 자리가 없다.

## 직접 만들기

### 1단계: 에너지 게이트

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 2단계: Python에서 Silero VAD 사용

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 3단계: 턴 종료 상태 머신

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 4단계: flush trick 뼈대

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

이 방식이 동작하려면 STT(Kyutai, Deepgram, AssemblyAI)가 flush를 지원해야 한다. Whisper streaming은 지원하지 않는다. 블록 기반이라 항상 청크를 기다린다.

## 활용하기

| 상황 | VAD 선택 |
|-----------|-----------|
| 개방형, 빠름, 일반 용도 | Silero VAD |
| 상업용 콜센터 | Cobra VAD |
| 온디바이스(휴대폰) | Silero VAD ONNX |
| 연구 / 화자 분리 | pyannote segmentation |
| 의존성 없는 fallback | WebRTC VAD (legacy) |
| 턴 종료 품질 필요 | Silero + LiveKit turn-detector layered |

경험칙: 정말 다른 선택지가 없는 경우가 아니라면 에너지만 쓰는 VAD를 배포하지 말라.

## 함정

- **고정 임계값.** 조용한 곳에서는 동작하지만 시끄러운 곳에서는 실패한다. 디바이스에서 보정하거나 Silero로 전환하라.
- **너무 짧은 침묵 행오버.** 에이전트가 문장 중간에 끼어든다. 대화 음성에서는 500-800ms가 적절한 지점이다.
- **너무 긴 행오버.** 둔하게 느껴진다. 대상 사용자로 A/B 테스트하라.
- **프리롤 버퍼 없음.** 사용자 오디오의 첫 200-300ms가 사라진다. 항상 rolling pre-roll을 유지하라.
- **의미적 엔드포인팅 무시.** "음, 생각해 볼게..."에는 긴 멈춤이 들어 있다. 사용자는 생각 도중 말을 끊기는 것을 싫어한다. LiveKit의 turn-detector나 유사한 방식을 사용하라.

## 내보내기

`outputs/skill-vad-tuner.md`로 저장하라. 워크로드에 맞는 VAD 모델, 임계값, 행오버, 프리롤, 턴 감지 전략을 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. 음성 + 침묵 + 음성 + 기침 시퀀스를 시뮬레이션하고 세 가지 VAD 단계를 테스트한다.
2. **보통.** `silero-vad`를 설치하고 5분 녹음을 처리한 뒤, 첫 단어 잘림과 false trigger를 모두 최소화하도록 임계값을 튜닝하라. precision/recall을 보고하라.
3. **어려움.** 미니 턴 감지기를 만들어라. Silero VAD + 최근 10개 단어 임베딩 위의 3-layer MLP(sentence-transformers 사용). 손으로 라벨링한 턴 종료 데이터셋에서 학습하라. Silero만 쓴 기준보다 F1을 10% 높여라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| VAD | 음성 감지기 | 프레임 단위 이진 판단: 이것이 음성인가? |
| Turn detection | 엔드포인팅 | VAD + 침묵 행오버 + 의미적 엔드포인트. |
| Silence hangover | 음성 뒤 대기 | 턴 종료를 선언하기 전 기다리는 시간. 500-800ms. |
| Pre-roll | 음성 전 버퍼 | VAD가 켜지기 전 300-500ms 오디오를 보관한다. |
| Flush trick | Kyutai 해킹 기법 | VAD → flush-STT → 500ms 지연 대신 125ms. |
| Semantic endpoint | "멈추려는 의도였나?" | 침묵뿐 아니라 단어를 보는 ML 분류기. |
| TPR @ FPR 5% | ROC 지점 | 표준 VAD 벤치마크. Silero 87.7%, WebRTC 50%. |

## 더 읽을거리

- [Silero VAD](https://github.com/snakers4/silero-vad) — 기준이 되는 공개 VAD.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 상용 정확도 선두.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — 200ms 미만을 만드는 엔지니어링 기법.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — 프로덕션의 의미적 엔드포인팅.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 레거시 기준선.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — 화자 분리 수준의 segmentation.
