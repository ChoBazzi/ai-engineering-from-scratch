# Whisper — 아키텍처와 파인튜닝

> Whisper는 30초 창을 쓰는 Transformer encoder-decoder이며, 68만 시간의 다국어 약지도 음성-텍스트 쌍으로 학습되었다. 하나의 아키텍처로 여러 작업을 처리하고, 99개 언어에서 견고하게 동작한다. 2026년 기준 ASR의 표준 참조점이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제

OpenAI가 2022년 9월 공개한 Whisper는 ASR을 범용 상품처럼 사용할 수 있게 만든 첫 모델이었다. 오디오를 넣으면 텍스트가 나오고, 99개 언어를 지원하며, 잡음에 강하고, 노트북에서도 실행된다. 2024년에는 OpenAI가 Large-v3와 Turbo 변형을 내놓았고, 2026년에는 Whisper가 팟캐스트 전사, 음성 비서, YouTube 자막까지 거의 모든 음성 인식 작업의 기본 기준선이 되었다.

하지만 Whisper를 영원히 블랙박스 파이프라인으로 취급할 수는 없다. 도메인 이동은 Whisper를 망가뜨린다. 기술 전문 용어, 화자 억양, 고유명사, 짧은 클립, 침묵 구간이 모두 문제를 만든다. 알아야 할 것은 다음과 같다.

1. 내부에서 실제로 무엇이 일어나는가.
2. 청크 처리, 스트리밍, 장문 오디오를 어떻게 올바르게 넣는가.
3. 언제 파인튜닝해야 하며 어떻게 해야 하는가.

## 개념

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**아키텍처.** 표준 Transformer encoder-decoder다.

- 입력: 30초 log-mel 스펙트로그램, 80개 mel, 10 ms hop → 3000프레임. 더 짧은 클립은 0으로 패딩하고, 더 긴 클립은 청크로 나눈다.
- Encoder: conv-downsample(stride 2) + `N`개 Transformer 블록. Large-v3 기준 32층, 1280차원, 20개 head.
- Decoder: causal self-attn과 encoder 출력에 대한 cross-attn을 갖는 `N`개 Transformer 블록. encoder와 같은 크기다.
- 출력: 51,865개 토큰 어휘 위의 BPE 토큰.

Large-v3는 1.55B 파라미터를 가진다. Turbo는 decoder를 32층에서 4층으로 줄여, WER 손실을 1% 미만으로 유지하면서 지연 시간을 8배 줄인다.

**프롬프트 형식.** Whisper는 decoder 프롬프트의 특수 토큰으로 조종되는 멀티태스크 모델이다.

```text
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 언어 태그. 번역과 전사 동작을 강제로 지정한다.
- `<|transcribe|>` 또는 `<|translate|>` — 어떤 언어의 입력이든 영어 출력으로 번역하거나, 원문 그대로 전사한다.
- `<|notimestamps|>` — 단어 단위 타임스탬프를 건너뛴다. 더 빠르다.

하나의 모델이 여러 작업을 수행할 수 있는 이유가 바로 이 프롬프트다. `<|en|>`을 `<|fr|>`로 바꾸면 프랑스어를 전사한다.

**30초 창.** 모든 것이 30초에 고정되어 있다. 더 긴 클립은 청크로 나누어야 하고, 더 짧은 클립은 패딩된다. 창은 기본적으로 스트리밍되지 않는다. WhisperX, Whisper-Streaming, faster-whisper가 존재하는 이유가 여기에 있다.

**Log-mel 정규화.** `(log_mel - mean) / std`이며, 통계값은 Whisper 자체 학습 말뭉치에서 온다. `librosa.feature.melspectrogram`이 아니라 Whisper의 전처리(`whisper.audio.log_mel_spectrogram`)를 반드시 사용해야 한다.

### 2026년의 변형

| 변형 | 파라미터 | 지연 시간(A100) | WER(LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### 파인튜닝

2026년의 표준 workflow는 다음과 같다.

1. 목표 도메인 오디오 10-100시간과 정렬된 전사문을 수집한다.
2. `generate_with_loss` callback을 붙여 `transformers.Seq2SeqTrainer`를 실행한다.
3. 파라미터 효율 방식: attention layer의 `q_proj`, `k_proj`, `v_proj`에 LoRA를 적용하면 WER 비용을 0.3 미만으로 유지하면서 GPU 메모리를 4배 줄인다.
4. 데이터가 10시간 미만이면 encoder를 freeze한다. decoder만 튜닝한다.
5. Whisper 자체 tokenizer와 프롬프트 형식을 사용한다. tokenizer를 절대 바꾸지 않는다.

커뮤니티 결과: 의료 받아쓰기 20시간으로 Medium을 파인튜닝하면 의료 어휘에서 WER가 12%에서 4.5%로 떨어진다. 아이슬란드어 4시간으로 Turbo를 파인튜닝하면 WER가 18%에서 6%로 떨어진다.

## 직접 만들기

### 1단계: Whisper를 그대로 실행하기

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # runaway repetition을 막는다
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

항상 override해야 하는 주요 기본값은 `temperature=0.0`(sampling은 기본적으로 0.0 → 0.2 → 0.4 … fallback chain을 탄다), `condition_on_previous_text=False`(연쇄 환각 문제를 막는다), `no_speech_threshold=0.6`(침묵 감지)이다.

### 2단계: 청크 기반 장문 처리

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX는 (1) Silero VAD gating, (2) wav2vec 2.0을 통한 단어 단위 alignment, (3) `pyannote.audio`를 통한 diarization을 추가한다. 2026년 production 전사 작업의 주력 도구다.

### 3단계: LoRA로 파인튜닝하기

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

이후에는 표준 Trainer loop를 사용한다. 1000 step마다 checkpoint를 저장한다. held-out 데이터에서 WER로 평가한다.

### 4단계: 각 layer가 무엇을 학습하는지 검사하기

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

heatmap으로 시각화하면 decoder step이 encoder frame을 훑어가며 대각선 alignment를 만드는 것을 볼 수 있다. 그 대각선이 Whisper가 이해하는 단어 타임스탬프다.

## 사용하기

2026년의 stack:

| 상황 | 선택 |
|-----------|------|
| 일반 영어, 오프라인 | `whisperx`를 통한 Large-v3-turbo |
| 모바일 / edge | quantized Whisper-Tiny(int8) 또는 Moonshine |
| 다국어 장문 | Large-v3 + `whisperx` + diarization |
| low-resource language | LoRA로 Medium 또는 Turbo 파인튜닝 |
| 스트리밍(2초 지연) | Whisper-Streaming 또는 Parakeet-TDT |
| 단어 단위 타임스탬프 | WhisperX(wav2vec 2.0 기반 forced alignment) |

`faster-whisper`(CTranslate2 backend)는 2026년 가장 빠른 CPU+GPU inference runtime이다. 출력은 동일하면서 vanilla보다 4배 빠르다.

## 2026년에도 여전히 배포되는 함정

- **침묵 구간의 환각 텍스트.** Whisper는 caption으로 학습되어 "Thanks for watching!", "Subscribe!", 노래 가사를 포함한다. 호출 전에 항상 VAD-gate를 적용한다.
- **`condition_on_previous_text` cascade.** 한 번의 환각이 뒤따르는 window를 오염시킨다. 청크 사이의 유창성이 꼭 필요하지 않다면 `False`로 둔다.
- **짧은 클립 패딩.** 2초 클립을 30초로 패딩하면 뒤쪽 침묵에서 환각이 생길 수 있다. `pad=False`를 쓰거나 VAD-gate를 적용한다.
- **잘못된 mel 통계.** Whisper의 mel 대신 librosa의 mel을 쓰면 거의 무작위 출력이 나온다. `whisper.audio.log_mel_spectrogram`을 사용한다.

## 배포하기

`outputs/skill-whisper-tuner.md`로 저장한다. 주어진 도메인에 맞는 Whisper 파인튜닝 또는 inference pipeline을 설계한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행한다. Whisper 스타일 프롬프트를 tokenize하고, decode shape budget을 계산하며, 10분짜리 클립의 chunk schedule을 출력한다.
2. **중간.** `faster-whisper`를 설치하고 10분짜리 팟캐스트를 전사한 뒤, 사람 전사문과 WER를 비교한다. `language="auto"`와 강제 `language="en"`을 비교해 본다.
3. **어려움.** HF `datasets`를 사용해 Whisper가 어려워하는 언어(예: Urdu)를 고르고, 2시간 데이터로 Medium을 LoRA 방식으로 2 epoch 파인튜닝한 뒤 WER delta를 보고한다.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-------------------|-----------|
| 30-sec window | Whisper의 한계 | 하드 입력 상한. 더 긴 오디오는 청크로 나눈다. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>`가 decoder prompt를 시작한다. |
| Timestamps token | 시간 alignment | 0.02초 간격의 모든 offset이 51k vocab의 특수 토큰이다. |
| Turbo | 빠른 변형 | decoder 4층, 8배 빠름, WER regression 1% 미만. |
| WhisperX | 장문 wrapper | VAD + Whisper + wav2vec alignment + diarization. |
| LoRA fine-tune | 효율적 튜닝 | attention에 low-rank adapter를 추가하고 파라미터 약 0.3%만 학습한다. |
| Hallucination | 조용한 실패 | Whisper가 잡음/침묵에서 유창한 영어를 만들어낸다. |

## 더 읽기

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — 원래 아키텍처와 학습 recipe.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-layer decoder, 8배 속도 향상.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — 장문, 단어 alignment, diarization.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 기반, 4배 빠름.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — 표준 LoRA / full-FT walkthrough.
