# 오디오 생성

> 오디오는 16-48 kHz의 1-D 신호다. 5초 클립은 80-240k samples다. 어떤 transformer도 그 sequence에 직접 attention을 걸지 않는다. 2026년의 모든 프로덕션 오디오 모델은 같은 해법을 쓴다. neural codec(Encodec, SoundStream, DAC)이 오디오를 50-75 Hz의 discrete tokens로 압축하고, transformer 또는 diffusion 모델이 tokens를 생성한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 문제

오디오 생성 작업은 세 가지다.

1. **Text-to-speech.** 텍스트가 주어지면 음성을 만든다. 깨끗한 speech는 narrow-band이고 phonetic structure가 강하다. transformer-over-tokens로 잘 해결된다. VALL-E(Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **음악 생성.** prompt(text, melody, chord progression, genre)가 주어지면 음악을 만든다. 훨씬 더 넓은 분포다. MusicGen(Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **오디오 효과 / 사운드 디자인.** prompt가 주어지면 ambient sound 또는 Foley를 만든다. AudioGen, AudioLDM 2, Stable Audio Open.

세 작업 모두 같은 기반에서 실행된다. neural audio codec + token-AR 또는 diffusion generator다.

## 개념

![오디오 생성: codec tokens + transformer 또는 diffusion](../assets/audio-generation.svg)

### Neural audio codec

Encodec(Meta, 2022), SoundStream(Google, 2021), Descript Audio Codec(DAC, 2023). convolutional encoder가 waveform을 timestep별 vector로 압축한다. residual vector quantization(RVQ)은 각 vector를 K개 codebook index의 cascade로 바꾼다. decoder는 이를 되돌린다. 8개 RVQ codebook을 75 Hz로 사용하는 2 kbps의 24 kHz 오디오는 초당 600 tokens다.

```text
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 그 위의 두 가지 생성 패러다임

**Token-autoregressive.** RVQ tokens를 sequence로 flatten하고 decoder-only transformer를 실행한다. MusicGen은 K개 codebook stream을 stream별 offset과 함께 병렬로 내보내기 위해 "delayed parallel"을 사용한다. VALL-E는 text prompt + 3초 voice sample로 speech tokens를 생성한다.

**Latent diffusion.** codec tokens를 continuous latents로 pack하거나 categorical diffusion으로 모델링한다. Stable Audio 2.5는 continuous audio latents에 flow matching을 사용한다. AudioLDM 2는 text-to-mel-to-audio diffusion을 사용한다.

2024-2026년의 추세는 음악에서는 flow matching이 이기고 있다는 것이다(더 빠른 inference, 더 깨끗한 samples). 반면 speech에서는 token-AR가 여전히 우세하다. 자연스럽게 causal하고 stream이 잘 되기 때문이다.

## 프로덕션 지형

| 시스템 | 작업 | Backbone | Latency |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms first token |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Non-streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching on audio latents | 1분 클립에 ~10s |
| Suno v4 | Full songs | 미공개, token-AR로 추정 | 곡당 ~30s |
| Udio v1.5 | Full songs | 미공개 | 곡당 ~30s |
| MusicGen 3.3B | Music | Token-AR on Encodec 32kHz | Real-time |
| AudioCraft 2 | Music + SFX | Flow matching | 5초 클립에 ~5s |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

## 직접 만들기

`code/main.py`는 핵심 아이디어를 시뮬레이션한다. 두 가지 뚜렷한 "style"에서 생성된 synthetic "audio token" sequence에 작은 next-token transformer를 학습한다. style A는 낮은 token과 높은 token이 번갈아 나오고, style B는 monotonic ramp다. style을 조건으로 sample한다.

### 1단계: synthetic audio token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": 교대 패턴
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 2단계: 작은 token predictor 학습

style을 조건으로 하는 bigram-style predictor다. 핵심은 패턴이다. codec tokens -> cross-entropy training -> autoregressive sampling.

### 3단계: 조건부 sample

style token과 시작 token이 주어지면 예측 분포에서 다음 token을 sample한다. 20-40 tokens 동안 계속한다.

## 함정

- **Codec 품질이 출력 품질의 상한을 정한다.** codec이 어떤 소리를 충실히 표현하지 못하면 generator 품질을 아무리 높여도 도움이 되지 않는다. DAC가 현재 open best다.
- **RVQ 오류 누적.** 각 RVQ layer는 이전 layer의 residual을 모델링한다. layer 1의 오류가 전파된다. 높은 layer에는 temperature 0으로 sample하는 것이 도움이 된다.
- **음악 구조.** 30초 tokens는 75 Hz에서 20k+ tokens다. transformer에는 어렵다. MusicGen은 sliding window + prompt continuation을 쓰고, Stable Audio는 더 짧은 clip + crossfading을 쓴다.
- **경계 아티팩트.** 생성된 clip 사이를 crossfade하려면 세심한 overlap-add가 필요하다.
- **깨끗한 데이터에 대한 갈증.** 음악 generator에는 라이선스가 확보된 수만 시간의 음악이 필요하다. Suno / Udio RIAA lawsuit(2024)가 이 문제를 수면 위로 끌어올렸다.
- **Voice cloning 윤리.** 3초 sample과 text prompt만으로 VALL-E / XTTS / ElevenLabs는 voice를 clone할 수 있다. 모든 프로덕션 모델에는 abuse detection + opt-out lists가 필요하다.

## 사용하기

| 작업 | 2026년 stack |
|------|------------|
| Commercial TTS | ElevenLabs, OpenAI TTS 또는 Azure Neural |
| Voice cloning(동의 검증됨) | XTTS v2(open) 또는 ElevenLabs Pro |
| 빠른 배경 음악 | Stable Audio 2.5 API, Suno 또는 Udio |
| 가사가 있는 음악 | Suno v4 또는 Udio v1.5 |
| Sound effects / Foley | AudioCraft 2, ElevenLabs SFX 또는 Stable Audio Open |
| Real-time voice agent | GPT-4o realtime 또는 Gemini Live |
| Open-weights music research | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Dubbing / translation | HeyGen, ElevenLabs Dubbing |

## 출시하기

`outputs/skill-audio-brief.md`를 저장한다. 이 스킬은 audio brief(task, duration, style, voice, license)를 입력받아 다음을 출력한다. model + hosting, prompt format(genre tags, style descriptors, structural markers), codec + generator + vocoder chain, seed protocol, eval plan(MOS / CLAP score / CER for TTS / user A/B).

## 연습문제

1. **쉬움.** `code/main.py`를 실행하고 style을 명시적으로 설정하라. 생성된 sequence가 해당 style의 패턴과 맞는지 확인하라.
2. **중간.** delayed parallel decoding을 추가하라. 1 step offset을 유지해야 하는 2개 token stream을 시뮬레이션하라. joint predictor를 학습하라.
3. **어려움.** HuggingFace transformers로 MusicGen-small을 로컬에서 실행하라. 서로 다른 prompt 세 개로 10초 clip을 생성하고, style adherence에 대해 A/B하라.

## 핵심 용어

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| Codec | "Neural compression" | 오디오용 encoder / decoder이며 보통 출력은 50-75 Hz tokens다. |
| RVQ | "Residual VQ" | K개 quantizer의 cascade이며, 각각 이전 결과의 residual을 모델링한다. |
| Token | "하나의 codec symbol" | codebook의 discrete index이며 보통 1024 또는 2048개다. |
| Delayed parallel | "Offset codebooks" | sequence 길이를 줄이기 위해 K개 token stream을 staggered offset으로 내보낸다. |
| Flow matching | "오디오에서의 2024년 승리" | diffusion의 더 곧은 경로 대안이며 sampling이 더 빠르다. |
| Voice prompt | "3초 sample" | cloned voice를 조향하는 speaker embedding 또는 token prefix. |
| Mel spectrogram | "시각화된 것" | log-magnitude perceptual spectrogram이며 많은 TTS 시스템에서 사용된다. |
| Vocoder | "Mel to wave" | mel spectrogram을 다시 audio로 변환하는 neural component. |

## 프로덕션 노트: 오디오는 streaming 문제다

오디오는 사용자가 한꺼번에 받는 것이 아니라 *생성되는 대로* 도착하기를 기대하는 유일한 출력 modality다. 프로덕션 용어로는 TPOT(Time Per Output Token)이 중요하다는 뜻이다. 목표 throughput은 사용자의 읽기 속도가 아니라 듣기 속도이기 때문이다. 16kHz 오디오를 약 75 tokens/second(Encodec)로 tokenize하면, playback을 매끄럽게 유지하기 위해 서버는 사용자당 ≥75 tokens/sec를 생성해야 한다.

아키텍처적 결과는 두 가지다.

- **Flow-matching 오디오 모델은 쉽게 stream할 수 없다.** Stable Audio 2.5와 AudioCraft 2는 고정 clip 길이를 한 번에 render한다. streaming하려면 clip을 chunk로 나누고 경계를 overlap해야 한다. sliding-window diffusion을 떠올리면 된다. codec AR 모델 대비 100-300ms latency overhead가 추가된다.

제품이 "live voice chat" 또는 "real-time music continuation"이라면 codec AR 경로를 선택한다. "제출하면 30초 clip을 render"하는 제품이라면 flow-matching이 품질과 total latency에서 이긴다.

## 더 읽을거리

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec standard.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 널리 쓰인 첫 neural audio codec.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — flow matching을 사용한 2025년 text-to-music.
