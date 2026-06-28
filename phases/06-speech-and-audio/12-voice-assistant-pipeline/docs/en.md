# 음성 어시스턴트 파이프라인 만들기: Phase 6 Capstone

> Lesson 01-11의 모든 것을 하나로 엮는다. 듣고, 추론하고, 다시 말하는 voice assistant를 만든다. 2026년에는 이것이 research 문제가 아니라 해결된 engineering 문제다. 다만 integration detail이 실제 출시 여부를 결정한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 minutes

## 문제

End-to-end assistant를 만든다.

1. mic input(16 kHz mono)을 capture한다.
2. user speech의 시작/끝을 감지한다.
3. streaming으로 transcribe한다.
4. transcript를 tool(timer, weather, calendar)을 호출할 수 있는 LLM에 전달한다.
5. LLM text를 TTS로 stream한다.
6. audio를 user에게 playback한다.
7. user가 응답 중간에 끼어들면 멈춘다.

Latency target: laptop CPU에서 user가 utterance를 마친 뒤 800 ms 안에 첫 TTS audio byte를 낸다. Quality target: 놓친 단어 없음, silence에서 hallucinated subtitle 없음, voice cloning leakage 없음, prompt injection 성공 없음.

## 개념

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 일곱 구성요소

1. **Audio capture.** Mic → 16 kHz mono → 20 ms chunk. Python에서는 보통 `sounddevice`, production에서는 native AudioUnit/ALSA/WASAPI.
2. **VAD (Lesson 11).** Silero VAD @ threshold 0.5, min speech 250 ms, silence hang-over 500 ms. "start"와 "end" signal을 낸다.
3. **Streaming STT (Lesson 4-5).** Whisper-streaming, Parakeet-TDT, 또는 Deepgram Nova-3(API). Partial + final transcript.
4. **Tool calling이 있는 LLM.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. tool용 JSON schema. token을 stream한다.
5. **Streaming TTS (Lesson 7).** Kokoro-82M(가장 빠른 오픈) 또는 Cartesia Sonic(상업). LLM token 20개 뒤 TTS를 시작한다.
6. **Playback.** Speaker out; 저대역폭 network에서는 opus-encode.
7. **Interruption handler.** TTS playback 중 VAD가 fire되면 playback을 멈추고, LLM을 cancel하고, STT를 다시 시작한다.

### 반드시 마주칠 세 가지 실패 모드

1. **첫 단어 잘림.** VAD가 한 박자 늦게 시작한다. user의 "hey"가 사라진다. start threshold를 0.5가 아니라 0.3으로 둔다.
2. **응답 중간 interruption 혼란.** user가 끼어든 뒤에도 LLM이 계속 생성해서 assistant가 user 위에 말한다. VAD → cancel-LLM을 연결하라.
3. **Silence hallucination.** Whisper가 silent warm-up frame에서 "Thanks for watching"을 출력한다. 항상 VAD-gate하라.

### 2026년 프로덕션 참조 스택

| 스택 | 지연 시간 | 라이선스 | 비고 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026년 industry default |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | DIY-friendly |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Single-model; 다른 architecture, lesson 15 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 가장 빠른 launch; customization 제한 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | Privacy / edge |

## 직접 만들기

### 1단계: 청크 처리가 있는 마이크 캡처(pseudocode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 2단계: VAD-gated 턴 캡처

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 3단계: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 4단계: LLM 루프 안의 도구 호출

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 5단계: 끼어들기 처리

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 활용하기

hardware 없이도 pipeline shape를 볼 수 있도록 stub model로 일곱 구성요소를 모두 연결한 실행 가능한 simulation은 `code/main.py`를 보라. 실제 구현에서는 stub을 다음으로 바꾼다.

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` 또는 `openai-whisper`
- `openai` (`gpt-4o`) 또는 `anthropic`
- `kokoro` 또는 `cartesia`
- I/O용 `sounddevice`

## 함정

- **PII를 영원히 logging.** 대부분의 jurisdiction에서 full-turn audio는 PII다. 30일 보관, at-rest encryption.
- **Barge-in 없음.** 사용자는 끼어든다. assistant는 말을 멈춰야 한다.
- **Block하는 TTS.** Synchronous TTS는 event loop를 block한다. async 또는 별도 thread를 사용하라.
- **Tool-call error handling 없음.** Tool은 실패한다. LLM은 error를 돌려받고 한 번 retry한 뒤 gracefully degrade해야 한다.
- **과도한 hallucination filter.** 너무 세게 걸면 assistant가 "I can't help with that."만 반복한다. 약하게 걸면 아무 말이나 한다. held-out set으로 보정하라.
- **Wake-word option 없음.** Always-listening은 privacy liability다. wake-word gate(Porcupine 또는 openWakeWord)를 추가하라.

## 출시하기

`outputs/skill-voice-assistant-architect.md`로 저장하라. budget + scale + language + compliance constraint가 주어졌을 때 full stack spec을 만든다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하라. stub module로 end-to-end full turn 하나를 simulate하고 stage별 latency를 출력한다.
2. **중간.** 미리 녹음한 `.wav`에서 STT stub을 실제 Whisper model로 바꿔라. WER와 end-to-end latency를 측정하라.
3. **어려움.** tool calling을 추가하라. `get_weather`(아무 API)와 `set_timer`를 구현한다. LLM을 tool로 route하고, user가 "set a 5 minute timer"라고 말하면 올바른 function이 fire되고 spoken reply가 이를 확인하는지 검증하라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Turn | user + assistant round-trip | VAD로 경계 지어진 user speech 하나 + LLM-TTS response 하나. |
| Barge-in | Interruption | assistant가 말하는 동안 user가 말하고, assistant가 멈춘다. |
| Wake word | "Hey assistant" | 짧은 keyword detector; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Turn ending | user가 끝냈다고 판단하는 VAD + min-silence decision. |
| Pre-roll | Pre-speech buffer | first-word clip을 피하기 위해 VAD fire 전 200-400 ms audio를 보관한다. |
| Tool call | Function invocation | LLM이 JSON을 emit하고 runtime이 dispatch하며 result가 loop 안으로 되돌아간다. |

## 더 읽을거리

- [LiveKit: voice agent quickstart](https://docs.livekit.io/agents/): production-grade reference.
- [Pipecat: voice agent examples](https://github.com/pipecat-ai/pipecat): DIY-friendly framework.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime): managed voice-native path.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi): full-duplex reference (Lesson 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/): wake-word gating.
- [Anthropic: tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use): LLM function calling.
