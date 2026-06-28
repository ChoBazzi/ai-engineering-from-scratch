# Audio Transformer — Whisper 아키텍처

> 오디오는 시간에 따른 주파수 이미지입니다. Whisper는 mel spectrogram을 먹고 텍스트로 답하는 ViT입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 minutes

## 문제

Whisper(OpenAI, Radford et al. 2022) 이전의 최첨단 자동 음성 인식(ASR)은 wav2vec 2.0과 HuBERT였습니다. 자기지도 특징 추출기에 fine-tuned head를 붙이는 방식입니다. 품질은 높지만 데이터 파이프라인이 비싸고, 도메인 변화에 취약했습니다. 다국어 음성 인식에는 언어 계열마다 별도 모델이 필요했습니다.

Whisper는 세 가지에 베팅했습니다.

1. **모든 것에 학습한다.** 인터넷에서 수집한 97개 언어의 약한 레이블 오디오 680,000시간입니다. 깨끗한 학술 코퍼스도, 음소 레이블도 없습니다.
2. **멀티태스크 단일 모델.** 하나의 디코더를 작업 토큰을 통해 전사, 번역, 음성 활동 감지, 언어 ID, 타임스탬핑에 공동 학습합니다.
3. **표준 인코더-디코더 트랜스포머.** 인코더는 log-mel spectrogram을 소비합니다. 디코더는 텍스트 토큰을 자기회귀적으로 생성합니다. vocoder, CTC, HMM은 없습니다.

결과적으로 Whisper large-v3는 억양, 잡음, 깨끗한 레이블 데이터가 전혀 없는 언어 전반에서 견고합니다. 2026년에는 모든 오픈소스 음성 어시스턴트와 대부분의 상용 음성 어시스턴트에서 기본 음성 프런트엔드입니다.

## 개념

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 1단계 — resample + window

오디오는 16 kHz입니다. 30초로 자르거나 패딩합니다. log-mel spectrogram을 계산합니다. 80 mel bins, 10 ms stride이므로 약 3,000 frames × 80 features가 됩니다. 이것이 Whisper가 보는 "입력 이미지"입니다.

### 2단계 — convolutional stem

커널 3, stride 2인 Conv1D 레이어 두 개가 3,000 frames를 1,500으로 줄입니다. 파라미터를 많이 추가하지 않고 시퀀스 길이를 절반으로 줄입니다.

### 3단계 — encoder

1,500 timestep 위에서 동작하는 24-layer(large 기준) 트랜스포머 인코더입니다. 사인파 위치 인코딩, self-attention, GELU FFN을 사용합니다. 1,500 × 1,280 은닉 상태를 만듭니다.

### 4단계 — decoder

24-layer 트랜스포머 디코더입니다. GPT-2 어휘의 상위 집합에 몇 가지 오디오 전용 특수 토큰을 더한 BPE vocabulary에서 토큰을 자기회귀적으로 생성합니다.

### 5단계 — task tokens

디코더 프롬프트는 모델이 무엇을 해야 하는지 알려주는 control token으로 시작합니다.

```text
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

또는

```text
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

모델은 이 관례로 학습되었습니다. prefix로 작업을 제어합니다. 2026년식 instruction-tuning에 해당하지만, 음성에 적용된 형태입니다.

### 6단계 — output

로그 확률 임계값을 둔 beam search(width 5)를 사용합니다. `<|notimestamps|>` 토큰이 없으면 오디오 0.02초마다 타임스탬프가 예측됩니다.

### Whisper 크기

| 모델 | 파라미터 | 레이어 | d_model | 헤드 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo(2024)는 디코더를 32 layers에서 4 layers로 줄였습니다. WER 회귀는 1포인트 미만이면서 디코딩은 8× 빨라졌습니다. 이 디코딩 속도 향상 때문에 2026년 실시간 음성 에이전트의 기본값은 Whisper-turbo입니다.

### Whisper가 하지 않는 것

- 화자 분리(diarization, 누가 말하는지)는 하지 않습니다. 필요하면 pyannote와 함께 쓰세요.
- 네이티브 실시간 스트리밍은 없습니다. 30초 window가 고정입니다. 현대 wrapper(`faster-whisper`, `WhisperX`)는 VAD + overlap으로 스트리밍을 덧붙입니다.
- 외부 chunking 없이는 30초를 넘는 long-form context가 없습니다. 실제로는 사람 말의 전사에 장거리 문맥이 거의 필요하지 않아 잘 동작합니다.

### 2026년 지형

| 작업 | 모델 | 비고 |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine은 엣지에서 4× 빠릅니다. |
| Multilingual ASR | Whisper-large-v3 | 97개 언어 |
| Streaming ASR | faster-whisper + VAD | 150 ms 지연 시간 목표도 달성 가능합니다. |
| TTS | Piper, XTTS-v2, Kokoro | 인코더-디코더 패턴이지만 Whisper와 닮았습니다. |
| Audio + language | AudioLM, SeamlessM4T | 하나의 트랜스포머 안에 텍스트 토큰 + 오디오 토큰이 있습니다. |

## 직접 만들기

`code/main.py`를 보세요. Whisper를 학습하지는 않습니다. log-mel spectrogram 파이프라인 + 작업 토큰 프롬프트 포매터를 만듭니다. 실제 프로덕션에서 직접 만지는 부분은 이것들입니다.

### 1단계: 오디오 합성

16 kHz로 샘플링한 440 Hz 1초 사인파를 생성합니다. 샘플은 16,000개입니다.

### 2단계: log-mel spectrogram(단순화)

완전한 mel spectrogram에는 FFT가 필요합니다. 여기서는 `librosa` 없이 파이프라인을 보여주기 위해 단순화한 framing + frame별 energy 버전을 사용합니다.

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame은 25 ms, hop은 10 ms입니다. Whisper의 windowing과 일치합니다. 교육 목적상 frame별 energy가 mel bin을 대신합니다.

### 3단계: 30초로 패딩

Whisper는 항상 30초 chunk를 처리합니다. spectrogram을 3,000 frames로 패딩하거나 자릅니다.

### 4단계: prompt token 만들기

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

이것이 전체 작업 제어 표면입니다. 4-token prefix입니다.

## 사용하기

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

더 빠른 OpenAI 호환 방식입니다.

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026년에 Whisper를 고를 때:**

- 하나의 모델로 다국어 ASR을 해야 할 때.
- 잡음이 많고 다양한 오디오를 견고하게 전사해야 할 때.
- 연구 / 프로토타입 ASR이 필요할 때. 가장 빠른 출발점입니다.

**다른 것을 고를 때:**

- 엣지에서 초저지연 스트리밍이 필요할 때. 동일 품질에서는 Moonshine이 Whisper보다 빠릅니다.
- <200 ms가 필요한 실시간 대화형 AI일 때. 전용 streaming ASR을 쓰세요.
- 화자 분리가 필요할 때. Whisper는 이를 하지 않으므로 pyannote를 덧붙입니다.

## 실전 적용

`outputs/skill-asr-configurator.md`를 보세요. 이 스킬은 새 음성 애플리케이션에 맞는 ASR 모델, 디코딩 파라미터, 전처리 파이프라인을 고릅니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 16 kHz의 1초 신호에서 10 ms hop을 쓰면 frame 수가 약 100개인지 확인하세요. 30초라면 약 3,000 frames입니다.
2. **보통.** `numpy.fft`로 완전한 log-mel spectrogram을 만드세요. 80 mel bins가 수치 오차 범위 안에서 `librosa.feature.melspectrogram(n_mels=80)`과 일치하는지 확인하세요.
3. **어려움.** 스트리밍 추론을 구현하세요. 오디오를 2초 overlap이 있는 10초 window로 나누고, 각 chunk에 Whisper를 실행한 뒤 transcript를 병합합니다. 5분짜리 팟캐스트 샘플에서 single-pass 대비 word-error rate를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 보통 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Mel spectrogram | "오디오 이미지" | 한 축은 주파수 bin, 다른 축은 시간 frame인 2D 표현입니다. 각 cell에는 로그 스케일 energy가 있습니다. |
| Log-mel | "Whisper가 보는 것" | Mel spectrogram에 log를 통과시킨 것입니다. 사람이 느끼는 loudness를 근사합니다. |
| Frame | "하나의 시간 조각" | 샘플의 25 ms window입니다. 10 ms stride로 겹칩니다. |
| Task token | "음성용 프롬프트 prefix" | 디코더 프롬프트에 들어가는 `<\|transcribe\|>` / `<\|translate\|>` 같은 특수 토큰입니다. |
| Voice activity detection (VAD) | "음성을 찾기" | ASR 전에 침묵을 제거하는 gate입니다. 비용을 크게 줄입니다. |
| CTC | "Connectionist Temporal Classification" | alignment-free 학습을 위한 고전적 ASR 손실입니다. Whisper는 이를 사용하지 않습니다. |
| Whisper-turbo | "작은 디코더, 전체 인코더" | large-v3 인코더 + 4-layer 디코더입니다. 디코딩이 8× 빠릅니다. |
| Faster-whisper | "프로덕션 wrapper" | CTranslate2 재구현입니다. int8 quantization을 쓰며 OpenAI reference보다 4× 빠릅니다. |

## 더 읽을거리

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 논문입니다.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — reference code + model weights입니다. `whisper/model.py`를 읽으면 Conv1D stem + encoder + decoder를 처음부터 끝까지 약 400줄로 볼 수 있습니다.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — Steps 5–6에서 설명한 beam-search + task-token 로직이 여기에 있습니다. 500줄이며 충분히 읽을 수 있습니다.
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 선행 모델입니다. 일부 설정에서는 여전히 SOTA 특징입니다.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 프로덕션 wrapper이며 reference보다 4× 빠릅니다.
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024년 엣지 친화적 ASR입니다. Whisper와 닮았지만 더 작습니다.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — mel spectrogram preprocessor와 token-timestamp 처리를 포함한 표준 fine-tuning recipe입니다.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — lesson의 아키텍처 다이어그램과 대응되는 전체 구현입니다(encoder, decoder, cross-attention, generation).
