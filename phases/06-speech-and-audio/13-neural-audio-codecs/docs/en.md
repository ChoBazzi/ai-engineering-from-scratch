# 신경 오디오 코덱: EnCodec, SNAC, Mimi, DAC와 Semantic-Acoustic Split

> 2026년 audio generation은 거의 모두 token이다. EnCodec, SNAC, Mimi, DAC는 연속 waveform을 transformer가 예측할 수 있는 discrete sequence로 바꾼다. semantic-vs-acoustic token split, 즉 첫 codebook은 semantic으로, 나머지는 acoustic으로 쓰는 방식은 audio에서 Transformer 이후 가장 중요한 architecture 전환이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 문제

Language model은 discrete token 위에서 동작한다. Audio는 continuous하다. speech / music을 위한 LLM 스타일 모델(MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus)을 원한다면 먼저 **neural audio codec**이 필요하다. 즉 audio를 작은 vocabulary의 token으로 discretize하는 학습된 encoder와 waveform을 reconstruct하는 짝 decoder가 필요하다.

두 family가 등장했다.

1. **Reconstruction-first codec**: EnCodec, DAC. perceptual audio quality를 optimize한다. token은 "acoustic"이다. speaker identity, timbre, background noise를 포함해 모든 것을 포착한다.
2. **Semantic-first codec**: Mimi(Kyutai), SpeechTokenizer. 첫 codebook이 linguistic / phonetic content를 encode하도록 강제한다(대개 WavLM에서 distill). 이후 codebook은 acoustic detail이다.

2024-2026년의 통찰: **순수 reconstruction codec으로 text에서 생성하려 하면 흐릿한 speech가 나온다.** codec token 위의 LLM이 같은 codebook 안에서 language structure와 acoustic structure를 모두 배워야 하는데, 이는 scale되지 않는다. 이를 분리하는 것, 즉 semantic codebook 0과 acoustic codebooks 1-N으로 나누는 것이 Moshi와 Sesame CSM을 작동하게 만든다.

## 개념

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 핵심 기법: Residual Vector Quantization(RVQ)

좋은 품질을 내려면 수백만 code가 필요한 하나의 큰 codebook을 쓰는 대신, 모든 현대 audio codec은 **RVQ**를 사용한다. 작은 codebook을 cascade로 쌓는 방식이다. 첫 codebook은 encoder output을 quantize하고, 두 번째는 residual을 quantize하는 식이다. 각 codebook은 1024 code다. 8개 codebook = effective vocabulary 1024^8 = 10^24.

Inference 때 decoder는 frame마다 선택된 모든 code를 더해 reconstruct한다.

### 2026년에 중요한 네 codec

**EnCodec (Meta, 2022).** baseline. waveform 위의 encoder-decoder, RVQ bottleneck. 24 kHz, 최대 32 codebook 가능, 기본 4 codebook @ 1.5 kbps. `1D conv + transformer + 1D conv` architecture를 사용한다. MusicGen이 사용한다.

**DAC (Descript, 2023).** L2-normalized codebook, periodic activation function, 개선된 loss를 갖춘 RVQ. 오픈 codec 중 reconstruction fidelity가 가장 높다. 12 codebook을 쓰면 원본 speech와 구분이 어려울 때도 있다. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** Multi-scale RVQ. coarse codebook은 fine codebook보다 낮은 frame rate에서 동작한다. 사실상 audio를 계층적으로 모델링한다. 약 12 Hz의 coarse "sketch"와 50 Hz의 detail이다. hierarchical structure가 LM 기반 generation과 잘 맞기 때문에 Orpheus-3B가 사용한다.

**Mimi (Kyutai, 2024).** 2026년의 game-changer. 12.5 Hz frame rate(매우 낮음), 8 codebook @ 4.4 kbps. Codebook 0은 **WavLM에서 distill**된다. WavLM의 speech-content feature를 예측하도록 학습된다. Codebook 1-7은 acoustic residual이다. 이 split이 Moshi(Lesson 15)와 Sesame CSM을 구동한다.

### Frame rate는 language modeling에서 중요하다

낮은 frame rate = 짧은 sequence = 빠른 LM.

| 코덱 | Frame rate | 1 s = N frames | 적합한 용도 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | music, general audio |
| DAC-44.1k | 86 Hz | 86 | high-fidelity music |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM efficient |
| Mimi | 12.5 Hz | 12.5 | streaming speech |

12.5 Hz에서는 10초 utterance가 codec frame 125개뿐이다. transformer가 쉽게 예측할 수 있다.

### Semantic vs acoustic token

```text
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token(Mimi의 codebook 0).** 무엇을 말했는지 encode한다. phoneme, word, content. auxiliary prediction loss를 통해 WavLM에서 distill된다.
- **Acoustic token(codebook 1-7).** timbre, speaker identity, prosody, background noise, fine detail을 encode한다.

AR LM은 먼저 semantic token을 예측하고(text에 conditioned), 그다음 acoustic token을 예측한다(semantic + speaker reference에 conditioned). 이 factorization 때문에 현대 TTS가 zero-shot voice cloning을 할 수 있다. semantic model은 content를 처리하고, acoustic model은 timbre를 처리한다.

### 2026년 재구성 품질(bits per sec, 낮은 bitrate가 좋음)

| 코덱 | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Opus 같은 전통 codec은 bit당 perceptual quality에서는 여전히 이긴다. Neural codec은 **discrete token**(Opus는 만들지 않음)과 **generative-model quality**(LM이 그 token으로 무엇을 할 수 있는가)에서 이긴다.

## 직접 만들기

### 1단계: EnCodec으로 encode

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps에서는 `n_codebooks=8`이다. 각 code는 0-1023(10-bit)이다.

### 2단계: decode하고 reconstruction 측정

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 3단계: semantic-acoustic split(Mimi style)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0은 WavLM-aligned다. text-to-semantic transformer를 학습할 수 있다. direct-to-audio보다 훨씬 작은 vocabulary다. 그런 다음 별도의 acoustic-to-waveform decoder가 speaker reference에 condition한다.

### 4단계: codec token 위의 AR LM이 작동하는 이유

Mimi의 12.5 Hz × 8 codebook에서 10 s speech clip:

```text
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 token은 transformer에게 사소한 context다. 256M parameter transformer도 현대 GPU에서는 10초 speech를 millisecond 단위로 생성할 수 있다.

## 활용하기

문제 → codec mapping:

| 작업 | 코덱 |
|------|-------|
| 범용 음악 생성 | EnCodec-24k |
| 최고 충실도 재구성 | DAC-44.1k |
| speech 위의 AR LM(TTS) | SNAC 또는 Mimi |
| Streaming full-duplex speech | Mimi (12.5 Hz) |
| text가 있는 sound-effect library | EnCodec + T5 condition |
| 세밀한 오디오 편집 | DAC + inpainting |

경험칙: **generative model을 만든다면 Mimi나 SNAC으로 시작하라. compression pipeline을 만든다면 Opus를 사용하라.**

## 함정

- **Codebook이 너무 많음.** codebook을 추가하면 fidelity도 선형으로 늘지만 LM sequence length도 선형으로 늘어난다. 8-12에서 멈춰라.
- **Frame-rate mismatch.** 12.5 Hz Mimi로 LM을 학습한 뒤 50 Hz EnCodec으로 fine-tune하면 조용히 실패한다.
- **모든 codebook이 같다고 가정.** Mimi에서 codebook 0은 content를 담는다. 이것을 잃으면 intelligibility가 망가진다. codebook 7을 잃는 것은 거의 티가 나지 않는다.
- **Reconstruction quality만 metric으로 사용.** codec은 reconstruction이 훌륭해도 semantic structure가 나쁘면 LM 기반 generation에는 쓸모없을 수 있다.

## 출시하기

`outputs/skill-codec-picker.md`로 저장하라. 주어진 generative 또는 compression task에 맞는 codec을 선택한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하라. toy scalar + residual quantizer를 구현하고 codebook을 추가할수록 reconstruction error가 어떻게 변하는지 측정한다.
2. **중간.** `encodec`을 설치하고 held-out speech clip에서 1, 4, 8, 32 codebook을 비교하라. PESQ 또는 MSE vs bitrate를 plot하라.
3. **어려움.** Mimi를 load하라. clip을 encode하라. codebook 0을 random integer로 바꾸고 decode하라. 그런 다음 codebook 7도 똑같이 바꿔라. 두 corruption을 비교하라. codebook 0 corruption은 intelligibility를 망가뜨려야 하고, codebook 7 corruption은 거의 바꾸지 않아야 한다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| RVQ | Residual quantization | 작은 codebook의 cascade; 각 단계가 이전 residual을 quantize한다. |
| Frame rate | Codec speed | 초당 token-frame 수. 낮을수록 LM이 빠르다. |
| Semantic codebook | Codebook 0 (Mimi) | SSL feature에서 distill된 codebook; content를 encode한다. |
| Acoustic codebooks | 나머지 전부 | Timbre, prosody, noise, fine detail. |
| PESQ / ViSQOL | Perceptual quality | MOS와 상관되는 objective metric. |
| EnCodec | Meta codec | RVQ baseline; MusicGen이 사용한다. |
| Mimi | Kyutai codec | 12.5 Hz frame rate; semantic-acoustic split; Moshi를 구동한다. |

## 더 읽을거리

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438): RVQ baseline.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546): highest-fidelity open.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411): multi-scale RVQ.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer): semantic-acoustic split, WavLM distillation.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143): 2-stage semantic/acoustic paradigm.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312): original streamable RVQ codec.
