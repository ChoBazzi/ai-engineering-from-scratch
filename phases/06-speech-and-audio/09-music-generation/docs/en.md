# 음악 생성: MusicGen, Stable Audio, Suno, 그리고 라이선스 지각변동

> 2026년 음악 생성: 상업 영역은 Suno v5와 Udio v4가 장악하고, 오픈소스는 MusicGen, Stable Audio Open, ACE-Step이 앞선다. 기술 문제는 거의 해결되었다. 법적 문제(Warner Music의 5억 달러 합의, UMG 합의)가 2025-2026년에 판도를 바꿨다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 minutes

## 문제

텍스트 → 가사, 보컬, 구조가 포함된 30초에서 4분 길이의 음악 클립. 세 가지 하위 문제가 있다.

1. **인스트루멘털 생성.** "lo-fi hip-hop drums with warm keys" 같은 텍스트 → 오디오. MusicGen, Stable Audio, AudioLDM.
2. **노래 생성(보컬 + 가사 포함).** "Country song about rainy Texas nights" → 완성곡. Suno, Udio, YuE, ACE-Step.
3. **조건부 / 제어 가능 생성.** 기존 클립을 연장하거나, 브리지를 다시 생성하거나, 장르를 바꾸거나, stem을 분리하거나, inpaint한다. Udio의 inpainting + stem separation은 2026년에 맞춰야 할 기능이다.

## 개념

![음악 생성: token-LM vs diffusion, 2026 모델 지도](../assets/music-generation.svg)

### 신경 codec 토큰 위의 토큰 LM

Meta의 **MusicGen**(2023, MIT)과 많은 파생 모델: 텍스트/멜로디 embedding에 조건화하고, EnCodec 토큰(32 kHz, codebook 4개)을 autoregressive하게 예측한 뒤 EnCodec으로 decode한다. 300M - 3.3B 파라미터. 강한 baseline이지만 30초를 넘기면 약해진다.

**ACE-Step**(오픈소스, 2026년 4월 4B XL 공개)은 이를 완성곡 가사 조건 생성으로 확장한다. 오픈 커뮤니티에서 Suno에 가장 가까운 모델이다.

### mel 또는 latent 위의 diffusion

**Stable Audio (2023)**와 **Stable Audio Open (2024)**: 압축 오디오에 대한 latent diffusion. loop, sound design, ambient texture에 강하다. 구조적인 완성곡에는 뛰어나지 않다.

**AudioLDM / AudioLDM2**: T2I 스타일 latent diffusion을 통한 text-to-audio이며, 음악, sound effect, speech로 일반화된다.

### 하이브리드(프로덕션): Suno, Udio, Lyria

가중치는 비공개다. 특화된 음성 / 드럼 / 멜로디 head를 갖춘 AR codec LM + diffusion 기반 vocoder일 가능성이 높다. Suno v5(2026)는 ELO 1293의 품질 선두다. Udio v4는 inpainting + stem separation(베이스, 드럼, 보컬 개별 다운로드)을 추가했다.

### 평가

- **FAD (Fréchet Audio Distance).** VGGish 또는 PANNs feature를 사용해 생성 오디오 분포와 실제 오디오 분포 사이의 embedding 수준 거리를 측정한다. 낮을수록 좋다. MusicGen small: MusicCaps에서 FAD 4.5, SOTA는 약 3.0.
- **음악성(주관).** 사람 선호도. Suno v5가 ELO 1293으로 앞선다.
- **텍스트-오디오 정렬.** prompt와 output 사이의 CLAP 점수.
- **음악성 artifact.** 박자를 벗어난 전환, 보컬 phrase drift, 30초 이후 구조 손실.

## 2026 모델 지도

| 모델 | 파라미터 | 길이 | 보컬 | 라이선스 |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 법적 지형(2025-2026)

- **Warner Music vs Suno 합의.** 5억 달러. 이제 WMG는 Suno의 AI 유사성, 음악 권리, 사용자 생성 트랙을 감독한다. Udio에도 유사한 UMG 합의가 있다.
- **EU AI Act** + **California SB 942**: AI 생성 음악은 공개되어야 한다.
- **Riffusion / MusicGen**은 MIT라 컴플라이언스 부담은 없지만, 상업용 보컬도 없다.

출시해도 안전한 패턴:

1. 인스트루멘털만 생성한다(MusicGen, Stable Audio Open, MIT/CC0 output).
2. 생성별 라이선스가 있는 상업 API(Suno, Udio, ElevenLabs Music)를 사용한다.
3. 소유 또는 라이선스 확보 catalog로 학습한다(대부분의 enterprise는 결국 여기에 도달한다).
4. 생성물에 watermark + metadata를 태그한다.

## 직접 만들기

### 1단계: MusicGen으로 생성하기

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

세 가지 크기: `small`(300M, 빠름), `medium`(1.5B), `large`(3.3B). Small만으로도 "아이디어가 먹히는지" 확인하기에는 충분하다.

### 2단계: 멜로디 조건화

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody는 chromagram을 받아 음색을 바꾸면서 선율을 보존한다. "이 멜로디를 현악 사중주로 들려줘" 같은 요청에 유용하다.

### 3단계: FAD 평가

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

VGGish embedding 거리를 계산한다. 장르 수준 회귀 테스트에는 유용하지만, 사람 청취자를 대체하지는 못한다.

### 4단계: LLM-음악 워크플로에 추가하기

Lesson 7-8의 아이디어와 결합한다.

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 활용하기

| 목표 | Stack |
|------|-------|
| 인스트루멘털 sound design | Stable Audio Open |
| 게임 / adaptive music | Google Lyria RealTime (closed) |
| 보컬이 있는 완성곡(상업) | 명시적 라이선스가 있는 Suno v5 또는 Udio v4 |
| 보컬이 있는 완성곡(오픈) | ACE-Step XL 또는 YuE |
| 짧은 광고 jingle | humming reference로 melody-conditioned한 MusicGen |
| 뮤직비디오 배경 | MusicGen + Stable Video Diffusion |

## 2026년에도 여전히 출시되는 함정

- **저작권 세탁 prompt.** "Song in the style of Taylor Swift": 상업용 Suno/Udio는 이제 이를 필터링하지만, 오픈 모델은 그렇지 않다. 자체 필터 목록을 추가하라.
- **30초 이후 반복 / drift.** AR 모델은 loop된다. 여러 생성을 crossfade하거나, 구조적 일관성을 위해 ACE-Step을 사용하라.
- **Tempo drift.** 모델이 BPM에서 벗어난다. prompt에 BPM tag를 넣고 librosa의 `beat_track`으로 post-filter하라.
- **보컬 명료도.** Suno는 뛰어나지만, 오픈 모델은 단어가 흐릿한 경우가 많다. 가사가 중요하면 상업 API나 fine-tune을 사용하라.
- **Mono output.** 오픈 모델은 mono 또는 가짜 stereo를 생성한다. 적절한 stereo reconstruction(ezst, Cartesia의 stereo diffusion)으로 업그레이드하라.

## 출시하기

`outputs/skill-music-designer.md`로 저장하라. music-gen deployment를 위해 모델, 라이선스 전략, 길이 / 구조 계획, 공개 metadata를 선택한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하라. ASCII symbol로 "generative" chord progression + drum pattern을 만드는 음악 생성 만화다. 원한다면 아무 MIDI renderer로 재생해 보라.
2. **중간.** `audiocraft`를 설치하고, MusicGen-small로 4개 genre prompt에 대해 10초 clip을 생성한 뒤 reference genre set과 FAD를 측정하라.
3. **어려움.** ACE-Step(또는 MusicGen-melody)을 사용해 같은 tune의 세 가지 variation을 서로 다른 timbre prompt로 생성하라. prompt와의 정렬을 검증하기 위해 CLAP similarity를 계산하라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| FAD | Audio FID | 실제 vs 생성 embedding 분포 사이의 Fréchet distance. |
| Chromagram | pitch로 표현한 melody | frame별 12차원 vector; melody conditioning의 input. |
| Stems | 악기 track | 분리된 bass / drums / vocals / melody WAV. |
| Inpainting | 구간 재생성 | time window를 mask하고, 모델이 그 부분만 재생성한다. |
| CLAP | Text-audio CLIP | contrastive audio-text embedding; text-audio alignment 평가. |
| EnCodec | Music codec | MusicGen이 사용하는 Meta의 neural codec; 32 kHz, codebook 4개. |

## 더 읽을거리

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284): 오픈 autoregressive benchmark.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358): sound-design 기본 선택지.
- [ACE-Step](https://github.com/ace-step/ACE-Step): 2026년 4월 공개된 오픈 4B 완성곡 generator.
- [Suno v5 platform docs](https://suno.com): 상업 품질 선두.
- [AudioLDM2](https://arxiv.org/abs/2308.05734): 음악 + sound effect용 latent diffusion.
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/): 2025년 11월 precedent.
