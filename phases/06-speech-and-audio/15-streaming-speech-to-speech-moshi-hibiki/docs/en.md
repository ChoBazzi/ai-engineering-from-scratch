# 스트리밍 Speech-to-Speech — Moshi, Hibiki, 그리고 Full-Duplex 대화

> 2024-2026년은 음성 AI를 새로 정의했다. Moshi는 200ms 지연 시간으로 동시에 듣고 말하는 단일 모델을 제공한다. Hibiki는 speech-to-speech 번역을 청크 단위로 수행한다. 둘 다 ASR → LLM → TTS 파이프라인을 버리고 Mimi codec token 위의 통합 full-duplex 아키텍처를 사용한다. 이것이 새로운 기준 설계다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제

Lesson 11 + 12로 만든 모든 음성 에이전트에는 약 300-500ms의 근본적인 지연 하한이 있다. VAD가 켜지고, STT가 처리하고, LLM이 추론하고, TTS가 생성한다. 각 단계마다 자체 최소 지연 시간이 있다. 튜닝하고 병렬화할 수는 있지만, 파이프라인 형태 자체가 한계를 만든다.

Moshi(Kyutai, 2024-2026)는 다른 질문을 던진다. 파이프라인이 없다면 어떨까? 한 모델이 오디오를 입력으로 받아 오디오를 직접, 연속적으로 출력하고, 텍스트는 필수 단계가 아니라 중간 "내적 독백"으로만 쓰면 어떨까?

그 답이 **full-duplex speech-to-speech**다. 이론적 지연 시간은 160ms(80ms Mimi 프레임 + 80ms 음향 지연). 실제 지연 시간은 단일 L4 GPU에서 200ms다. 최상급 파이프라인형 음성 에이전트가 달성하는 지연의 절반이다.

## 개념

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 아키텍처

**입력.** 두 개의 Mimi codec stream. 둘 다 12.5Hz × 8 codebooks:

- Stream 1: 사용자 오디오(Mimi 인코딩, 계속 도착)
- Stream 2: Moshi 자신의 오디오(Moshi가 생성)

**트랜스포머.** 70억 매개변수 Temporal Transformer가 두 스트림과 텍스트 "내적 독백" 스트림을 처리한다. 각 80ms step에서 다음을 수행한다.

1. 최신 사용자 Mimi token(8 codebooks)을 소비한다.
2. 가장 최근 Moshi Mimi token(생성된 8 codebooks)을 소비한다.
3. 다음 Moshi 텍스트 token(내적 독백)을 생성한다.
4. 작은 Depth Transformer를 통해 다음 Moshi Mimi token(8 codebooks)을 생성한다.

세 스트림, 즉 사용자 오디오, Moshi 오디오, Moshi 텍스트가 모두 병렬로 실행된다. Moshi는 말하는 동안 사용자의 말을 들을 수 있고, 사용자가 끼어들면 스스로 말을 멈출 수 있으며, 주요 발화를 깨지 않고 back-channel("mhm")을 할 수 있다.

**Depth transformer.** 한 프레임 안에서 8개 codebook은 병렬로 예측되지 않는다. codebook 사이 의존성이 있기 때문이다. 작은 2-layer "depth transformer"가 80ms 안에서 순차적으로 예측한다. 이는 AR codec LM의 표준 분해 방식이다(VALL-E, VibeVoice도 사용).

### 내적 독백 텍스트가 도움이 되는 이유

명시적 텍스트가 없으면 모델은 음향 스트림 안에서 언어를 암묵적으로 모델링해야 한다. Moshi의 통찰은 오디오와 함께 텍스트 token도 출력하도록 강제하는 것이다. 텍스트 스트림은 본질적으로 Moshi가 말하는 내용의 전사다. 이는 의미적 일관성을 높이고, 언어 모델 head를 교체하기 쉽게 만들며, 전사문을 무료로 제공한다.

### Hibiki: 스트리밍 speech-to-speech 번역

같은 아키텍처를 번역 쌍으로 학습한 것이다. 원본 오디오가 들어오면 대상 언어 오디오가 연속적으로 나온다. Hibiki-Zero(2026년 2월)는 단어 수준 정렬 학습 데이터의 필요를 없앴다. 문장 수준 데이터 + 지연 시간 최적화를 위한 GRPO 강화학습을 사용한다.

초기에는 네 가지 언어 쌍을 지원한다. 약 1000시간의 데이터로 새 언어에 적응시킬 수 있다.

### 더 넓은 Kyutai 스택(2026)

- **Moshi** — full-duplex 대화(프랑스어 우선, 영어도 잘 지원)
- **Hibiki / Hibiki-Zero** — 동시 음성 번역
- **Kyutai STT** — 스트리밍 ASR(500ms 또는 2.5초 look-ahead)
- **Kyutai Pocket TTS** — CPU에서 실행되는 100M-param TTS(2026년 1월)
- **Unmute** — 이들을 공개 서버에서 결합한 전체 파이프라인

L40S GPU에서 처리량은 실시간 3배 속도로 동시 세션 64개다.

### Sesame CSM — 사촌 모델

Sesame CSM(2025)은 비슷한 아이디어를 사용한다. Llama-3 backbone에 Mimi codec head를 붙인다. 하지만 CSM은 full-duplex가 아니라 단방향이다(맥락 + 텍스트를 받아 음성을 생성). 시장에서 가장 좋은 "voice presence" TTS지만, Moshi의 full-duplex 기능과는 완전히 같지 않다.

### 2026년 성능 수치

| 모델 | 지연 시간 | 사용 사례 | 라이선스 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | full-duplex English / French dialogue | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | French ↔ English streaming translation | CC-BY 4.0 |
| Hibiki-Zero | same | 5 language-pairs, no aligned data | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | context-conditioned TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | closed, OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | closed, Google API | commercial |

## 직접 만들기

### 1단계: 인터페이스

Moshi는 80ms 청크의 Mimi 인코딩 오디오를 받아 80ms 청크의 Mimi 인코딩 오디오를 반환하는 WebSocket 서버를 노출한다. 양방향으로, 끊임없이 동작한다.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 2단계: full-duplex 루프

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

두 방향은 동시에 실행된다. Python asyncio나 Rust futures가 표준 전송 방식이다.

### 3단계: 학습 목표(개념)

모든 80ms 프레임 `t`에 대해:

- Input: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Predict: `moshi_text[t]`, then `moshi_mimi[t, codebook_0..7]`

텍스트가 오디오보다 먼저 예측되고(내적 독백), 오디오는 depth transformer 안에서 codebook 순서대로 예측된다.

### 4단계: Moshi가 이기는 곳과 그렇지 않은 곳

Moshi가 이기는 곳:

- 저렴한 하드웨어에서 엔드투엔드 250ms 미만.
- 자연스러운 back-channel과 끼어들기.
- 파이프라인 glue code가 없다.

Moshi가 이기지 못하는 곳:

- 도구 호출(그렇게 학습되지 않았다. 별도 LLM 경로가 필요하다).
- 긴 추론(Moshi는 8B급 대화 모델이지 Claude/GPT-4가 아니다).
- 틈새 주제의 사실 정확도.
- 대부분의 프로덕션 엔터프라이즈 사용 사례(2026년에도 여전히 파이프라인을 사용).

## 활용하기

| 상황 | 선택 |
|-----------|------|
| 최저 지연 음성 companion | Moshi |
| 실시간 번역 통화 | Hibiki |
| 음성 데모 / 연구 | Moshi, CSM |
| 도구가 있는 엔터프라이즈 에이전트 | Pipeline (Lesson 12), not Moshi |
| 맥락 안의 커스텀 음성 TTS | Sesame CSM |
| speech-to-speech, 모든 언어 | GPT-4o Realtime or Gemini 2.5 Live (commercial) |

## 함정

- **제한적인 도구 호출.** Moshi는 대화 모델이지 에이전트 프레임워크가 아니다. 도구에는 파이프라인과 결합하라.
- **특정 음성 조건화.** Moshi는 하나의 학습된 persona를 사용한다. 클로닝은 별도 학습 run이다.
- **언어 범위.** 프랑스어 + 영어는 훌륭하지만 다른 언어는 제한적이다. Hibiki-Zero가 도움이 되지만 여전히 학습 데이터가 필요하다.
- **리소스 비용.** 전체 Moshi 세션은 GPU 슬롯을 점유한다. 저렴한 shared-tenant 배포 패턴이 아니다.

## 내보내기

`outputs/skill-duplex-pipeline.md`로 저장하라. 음성 에이전트 워크로드에 대해 파이프라인과 full-duplex 아키텍처 중 하나를 이유와 함께 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. 두 스트림 + 내적 독백 아키텍처를 기호적으로 시뮬레이션한다.
2. **보통.** HuggingFace에서 Moshi를 가져와 서버를 실행하고 대화 하나를 테스트하라. 사용자 발화 종료부터 Moshi 응답 시작까지의 wall-clock 지연 시간을 측정하라.
3. **어려움.** Lesson 12의 파이프라인 에이전트를 가져와, 서로 맞춘 테스트 발화 20개에서 Moshi와 P50 지연 시간을 비교하라. 그럼에도 파이프라인이 아키텍처상 이기는 경우를 정리하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Full-duplex | 동시에 듣고 말하기 | 같은 모델에서 두 오디오 스트림이 동시에 활성화된다. |
| Inner monologue | 모델의 텍스트 스트림 | Moshi는 오디오 출력과 함께 텍스트 token을 낸다. |
| Depth transformer | codebook 간 예측기 | 한 80ms 프레임 안에서 8개 codebook을 예측하는 작은 transformer. |
| Mimi | Kyutai의 codec | 12.5Hz × 8 codebooks. 의미+음향. Moshi의 기반. |
| Streaming S2S | 오디오 → 오디오 실시간 | 파이프라인 단계 없이 청크 단위 번역/대화. |
| Back-channeling | "Mhm" 반응 | Moshi는 자기 턴을 깨지 않고 작은 맞장구를 낼 수 있다. |

## 더 읽을거리

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — 논문.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 정렬 데이터 없는 스트리밍 번역.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 사양.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — 설치 + 서버.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — 폐쇄형 상용 peer.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — 내부의 STT/TTS 프레임워크.
