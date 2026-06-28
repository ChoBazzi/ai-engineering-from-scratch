# 텍스트 음성 변환(TTS) — Tacotron에서 F5와 Kokoro까지

> ASR은 음성을 텍스트로 뒤집고, TTS는 텍스트를 음성으로 뒤집는다. 2026년 stack은 세 부분이다. text → token, token → mel, mel → waveform. 각 부분마다 노트북에 들어가는 기본 모델이 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 minutes

## 문제

"Please remind me to water the plants at 6 pm."라는 문자열이 있다. 자연스럽게 들리고, 올바른 prosody(쉼, 강세)를 가지며, "plants"를 알맞은 모음으로 발음하고, 실시간 음성 비서를 위해 CPU에서 300 ms 안에 실행되는 3초짜리 오디오 클립이 필요하다. 또한 voice를 바꾸고, code-switched input("remind me at 6 pm, daijoubu?")을 처리하며, 이름 발음에서 민망한 결과를 내지 않아야 한다.

현대 TTS pipeline은 다음과 같다.

1. **Text frontend.** 텍스트를 정규화하고(날짜, 숫자, 이메일), phoneme 또는 subword token으로 변환하며, prosody feature를 예측한다.
2. **Acoustic model.** Text → mel spectrogram. Tacotron 2(2017), FastSpeech 2(2020), VITS(2021), F5-TTS(2024), Kokoro(2024).
3. **Vocoder.** Mel → waveform. WaveNet(2016), WaveRNN, HiFi-GAN(2020), BigVGAN(2022), 2024년 이후 neural codec vocoder.

2026년에는 end-to-end diffusion과 flow-matching model 때문에 acoustic + vocoder 분리가 흐려진다. 하지만 세 부분이라는 mental model은 디버깅할 때 여전히 유효하다.

## 개념

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2(2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder가 mel frame을 낸다. 느리고(AR), 긴 텍스트에서 흔들린다. 여전히 baseline으로 인용된다.

**FastSpeech 2(2020).** Non-autoregressive. duration predictor가 각 phoneme에 몇 개의 mel frame을 배정할지 출력한다. 1-pass이며 Tacotron보다 10배 빠르다. naturalness 일부를 잃지만(monotonic alignment) 거의 모든 곳에 배포된다.

**VITS(2021).** encoder + flow-based duration + HiFi-GAN vocoder를 variational inference로 end-to-end 공동 학습한다. 품질이 높고 단일 모델이다. 2022-2024년 dominant open-source TTS였다. 변형으로 YourTTS(multi-speaker zero-shot), XTTS v2(2024, Coqui)가 있다.

**F5-TTS(2024).** flow matching 위의 diffusion transformer다. 자연스러운 prosody와 5초 reference audio 기반 zero-shot voice cloning을 제공한다. 2026년 open-source TTS leaderboard 최상위권이다. 335M 파라미터다.

**Kokoro(2024).** 작고(82M), CPU에서 실행 가능하며, real-time use에서 best-in-class English TTS다. English-only closed vocabulary이며 apache-2.0이다.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** 상용 state of the art. ElevenLabs v2.5의 emotion tag("[whispered]", "[laughing]")와 character voice는 2026년 audiobook production을 지배한다.

### Vocoder의 진화

| 시대 | Vocoder | 지연 시간 | 품질 |
|-----|---------|---------|---------|
| 2016 | WaveNet | offline only | SOTA at release |
| 2018 | WaveRNN | ~realtime | good |
| 2020 | HiFi-GAN | 100× realtime | near-human |
| 2022 | BigVGAN | 50× realtime | generalizes across speakers/langs |
| 2024 | SNAC, DAC (neural codecs) | integrated with AR models | discrete tokens, bit-efficient |

2026년에는 대부분의 "TTS" 모델이 text에서 waveform까지 end-to-end로 간다. mel spectrogram은 내부 표현이 되었다.

### 평가

- **MOS(Mean Opinion Score).** 1-5 척도, crowd-sourced. 여전히 gold standard지만 매우 느리다.
- **CMOS(Comparative MOS).** A-vs-B preference. annotation당 confidence interval이 더 좁다.
- **UTMOS, DNSMOS.** reference-free neural MOS predictor. leaderboard에 사용된다.
- **ASR 기반 CER(Character Error Rate).** TTS 출력을 Whisper에 넣고, 입력 텍스트와 CER를 계산한다. intelligibility의 proxy다.
- **SECS(Speaker Embedding Cosine Similarity).** voice-cloning 품질.

LibriTTS test-clean의 2026년 숫자:

| 모델 | UTMOS | CER(Whisper 기준) | 크기 |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 직접 만들기

### 1단계: 입력을 phonemize하기

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phoneme은 보편적인 다리다. VITS 수준 아래의 모델에는 raw text를 직접 넣지 않는다.

### 2단계: Kokoro 실행하기(2026년 CPU 기본값)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

오프라인으로 실행되고, 단일 파일이며, 82M 파라미터다.

### 3단계: voice cloning과 함께 F5-TTS 실행하기

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

5초 reference clip과 그 transcript를 넘긴다. F5는 prosody와 timbre를 clone한다.

### 4단계: HiFi-GAN vocoder를 처음부터 보기

튜토리얼 스크립트에 넣기에는 너무 크지만, shape은 다음과 같다.

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

학습은 adversarial(short window에 discriminator) + mel-spectrogram reconstruction loss + feature-matching loss로 진행한다. 이미 commodity가 되었으므로 `hifi-gan` repo나 nvidia-NeMo의 pretrained checkpoint를 사용한다.

### 5단계: 전체 pipeline(pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 사용하기

2026년의 stack:

| 상황 | 선택 |
|-----------|------|
| Real-time English voice assistant | Kokoro(CPU) 또는 XTTS v2(GPU) |
| 5초 reference 기반 voice cloning | F5-TTS |
| Commercial character voices | ElevenLabs v2.5 |
| Audiobook narration | ElevenLabs v2.5 또는 XTTS v2 + fine-tune |
| Low-resource language | target-lang data 5-20시간으로 VITS 학습 |
| Expressive / emotion tags | ElevenLabs v2.5 또는 StyleTTS 2 fine-tune |

2026년 기준 open-source 선두는 **품질의 F5-TTS, 효율의 Kokoro**다. 역사가가 아니라면 Tacotron을 집어 들지 않는다.

## 함정

- **Text normalizer 없음.** "Dr. Smith"를 "Doctor"로 읽어야 하는가, "Drive"로 읽어야 하는가? "2026"은 "twenty twenty six"인가, "two zero two six"인가? phonemizer 전에 정규화한다.
- **OOV proper noun.** "Ghumare" → "ghyu-mair"? unknown token을 위한 fallback grapheme-to-phoneme model을 배포한다.
- **Clipping.** Vocoder output은 드물게 clipping되지만, inference에서 mel scaling mismatch가 있으면 ±1.0을 넘을 수 있다. 항상 `np.clip(wav, -1, 1)`을 적용한다.
- **Sample-rate mismatch.** Kokoro는 24 kHz를 출력하지만 downstream pipeline이 16 kHz를 기대한다면 resample해야 한다. 그렇지 않으면 aliasing이 생긴다.

## 배포하기

`outputs/skill-tts-designer.md`로 저장한다. 주어진 voice, latency, language target에 맞는 TTS pipeline을 설계한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행한다. toy vocab에서 phoneme dictionary를 만들고, phoneme별 duration을 추정하며, 가짜 "mel" schedule을 출력한다.
2. **중간.** Kokoro를 설치하고 같은 문장을 `af_bella`와 `am_adam` voice로 합성한다. audio duration과 주관적 품질을 비교한다.
3. **어려움.** 자신의 5초 reference clip을 녹음한다. F5-TTS로 clone한다. reference와 cloned output 사이의 SECS를 보고한다.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-------------------|-----------|
| Phoneme | 소리 단위 | 추상적 소리 class. 영어에는 39개(ARPABet)가 있다. |
| Duration predictor | 각 phoneme이 지속되는 시간 | non-AR model output. phoneme당 integer frame 수. |
| Vocoder | Mel → waveform | mel-spec을 raw sample로 mapping하는 neural net. |
| HiFi-GAN | 표준 vocoder | GAN 기반. 2020-2024년 dominant. |
| MOS | 주관적 품질 | human rater의 1-5 mean opinion score. |
| SECS | Voice-clone metric | target과 output speaker embedding 사이의 cosine similarity. |
| F5-TTS | 2024 open-source SOTA | Flow-matching diffusion. zero-shot cloning. |
| Kokoro | CPU English leader | 82M-param model, Apache 2.0. |

## 더 읽기

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — seq2seq baseline.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end flow-based.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 현재 open-source SOTA.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026년에도 배포되는 vocoder.
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 CPU-friendly English TTS.
