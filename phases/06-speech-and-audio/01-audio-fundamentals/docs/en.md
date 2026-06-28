# 오디오 기초 — 파형, 샘플링, 푸리에 변환

> 파형은 원시 신호입니다. 스펙트로그램은 표현입니다. Mel 특징은 ML에 친화적인 형태입니다. 모든 현대 ASR 및 TTS 파이프라인은 이 사다리를 올라가며, 첫 번째 단은 샘플링과 푸리에를 이해하는 것입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 minutes

## 문제

마이크는 시간에 따른 압력 신호를 만듭니다. 신경망은 텐서를 소비합니다. 그 사이에는 여러 관례가 쌓여 있고, 이를 어기면 조용한 버그가 생깁니다. 모델은 잘 학습되는 것처럼 보이지만 WER가 두 배가 되거나, TTS가 잡음을 내보내거나, 음성 복제 시스템이 화자가 아니라 마이크를 외워 버립니다.

음성 시스템의 모든 버그는 다음 세 질문 중 하나로 거슬러 올라갑니다.

1. 데이터는 어떤 샘플링 레이트로 녹음되었고, 모델은 무엇을 기대하는가?
2. 신호에 aliasing이 생겼는가?
3. 원시 샘플을 다루고 있는가, 아니면 주파수 표현을 다루고 있는가?

이 세 가지를 맞추면 Phase 6의 나머지는 다룰 수 있습니다. 틀리면 Whisper-Large-v4조차 엉망인 결과를 냅니다.

## 개념

![파형, 샘플링, DFT, 주파수 bin 시각화](../assets/audio-fundamentals.svg)

**파형.** `[-1.0, 1.0]` 범위의 float로 이루어진 1차원 배열입니다. 샘플 번호로 인덱싱합니다. 초 단위로 바꾸려면 샘플링 레이트로 나눕니다. `t = n / sr`. 16 kHz의 10초 클립은 160,000개 float 배열입니다.

**샘플링 레이트(sr).** 초당 샘플 개수입니다. 2026년에 흔히 쓰이는 레이트는 다음과 같습니다.

| 레이트 | 용도 |
|------|-----|
| 8 kHz | 전화, 레거시 VOIP. Nyquist가 4 kHz라 자음 정보가 죽습니다. ASR에는 피하세요. |
| 16 kHz | ASR 표준. Whisper, Parakeet, SeamlessM4T v2는 모두 16 kHz를 입력으로 받습니다. |
| 22.05 kHz | 구형 모델의 TTS vocoder 학습. |
| 24 kHz | 현대 TTS(Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD 오디오, 음악. |
| 48 kHz | 영화, 전문 오디오, 고충실도 TTS(VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** 샘플링 레이트 `sr`은 `sr/2`까지의 주파수를 모호함 없이 표현할 수 있습니다. `sr/2` 경계가 *Nyquist frequency*입니다. Nyquist를 넘는 에너지는 *aliased*되어 더 낮은 주파수로 접히고 신호를 오염시킵니다. 다운샘플링 전에는 항상 저역 통과 필터를 적용하세요.

**비트 깊이.** 16-bit PCM(signed int16, 범위 ±32,767)은 보편적인 교환 형식입니다. 음악에는 24-bit, 내부 DSP에는 32-bit float를 씁니다. `soundfile` 같은 라이브러리는 int16을 읽지만 `[-1, 1]` 범위의 float32 배열로 노출합니다.

**푸리에 변환.** 모든 유한 신호는 서로 다른 주파수의 사인파 합입니다. 이산 푸리에 변환(DFT)은 `N`개 샘플에 대해 `N`개의 복소 계수를 계산합니다. 주파수 bin마다 하나씩입니다. `bin k`는 `k · sr / N` Hz 주파수에 대응합니다. 크기는 해당 주파수의 진폭이고, 각도는 위상입니다.

**FFT.** Fast Fourier Transform: `N`이 2의 거듭제곱일 때 DFT를 계산하는 `O(N log N)` 알고리즘입니다. 모든 오디오 라이브러리는 내부에서 FFT를 사용합니다. 16 kHz에서 1024샘플 FFT를 하면 0-8 kHz 범위를 15.6 Hz 해상도로 덮는 512개의 사용 가능한 주파수 bin이 나옵니다.

**프레이밍 + 윈도우.** 전체 클립을 한 번에 FFT하지 않습니다. 겹치는 *프레임*으로 자르고(보통 25 ms 프레임, 10 ms hop), 각 프레임에 윈도우 함수(Hann, Hamming)를 곱해 가장자리 불연속을 줄인 다음 각 프레임에 FFT를 적용합니다. 이것이 Short-Time Fourier Transform(STFT)입니다. Lesson 02는 여기서 이어집니다.

```figure
mel-scale
```

## 직접 만들기

### 1단계: 클립을 읽고 파형 그리기

`code/main.py`는 데모에 의존성이 없도록 stdlib `wave` 모듈만 사용합니다. 프로덕션에서는 `soundfile` 또는 `torchaudio.load`를 사용할 것입니다. 둘 다 `(waveform, sr)` 튜플을 반환합니다.

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 2단계: 원리부터 사인파 합성하기

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

16 kHz에서 1초 길이의 440 Hz 사인파(기준음 A)는 16,000개 float입니다. 16-bit PCM 인코딩으로 `wave.open(..., "wb")`를 사용해 저장합니다.

### 3단계: DFT를 손으로 계산하기

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`입니다. `N=256`에서 정합성을 확인하기에는 괜찮지만 실제 오디오에는 쓸 수 없습니다. 실제 코드는 `numpy.fft.rfft`나 `torch.fft.rfft`를 호출합니다.

### 4단계: 지배적인 주파수 찾기

크기 피크 인덱스 `k_star`는 `k_star * sr / N` 주파수에 대응합니다. 440 Hz 사인파에 적용하면 `440 * N / sr` bin에서 피크가 나와야 합니다.

### 5단계: 에일리어싱 보여 주기

7 kHz 사인파를 10 kHz로 샘플링해 보세요(Nyquist = 5 kHz). 7 kHz 톤은 Nyquist보다 높아서 `10 − 7 = 3 kHz`로 접힙니다. FFT 피크는 3 kHz에 나타납니다. 이것이 고전적인 aliasing 데모이며 모든 DAC/ADC에 급격한 저역 통과 필터가 들어가는 이유입니다.

## 활용하기

2026년에 실제로 배포하게 될 스택은 다음과 같습니다.

| 작업 | 라이브러리 | 이유 |
|------|---------|-----|
| WAV/FLAC/OGG 읽기/쓰기 | `soundfile`(libsndfile wrapper) | 가장 빠르고 안정적이며 float32를 반환합니다. |
| 리샘플링 | `torchaudio.transforms.Resample` 또는 `librosa.resample` | 올바른 anti-aliasing이 내장되어 있습니다. |
| STFT / Mel | `torchaudio` 또는 `librosa` | GPU 친화적이며 PyTorch 생태계와 잘 맞습니다. |
| 실시간 스트리밍 | `sounddevice` 또는 `pyaudio` | 크로스플랫폼 PortAudio 바인딩입니다. |
| 파일 검사 | `ffprobe` 또는 `soxi` | CLI이고 빠르며 sr/채널/코덱을 보고합니다. |

결정 규칙: **무엇보다 먼저 샘플링 레이트를 맞추세요**. Whisper는 16 kHz mono float32를 기대합니다. 44.1 kHz stereo를 넣으면 모델 버그처럼 보이는 엉망인 결과가 나옵니다.

## 결과물

`outputs/skill-audio-loader.md`로 저장합니다. 이 스킬은 오디오 입력이 다운스트림 모델의 기대와 일치하는지 확인하고, 일치하지 않을 때 올바르게 리샘플링하도록 돕습니다.

## 연습문제

1. **쉬움.** 16 kHz에서 220 Hz + 440 Hz + 880 Hz를 섞은 1초 신호를 합성하세요. DFT를 실행하세요. 예상 bin에서 세 개의 피크가 나오는지 확인하세요.
2. **보통.** 48 kHz로 자신의 목소리 3초 WAV를 녹음하세요. `torchaudio.transforms.Resample`을 사용해 16 kHz로 다운샘플링하고(anti-aliasing 포함), 그다음 naive decimation(세 번째 샘플마다 선택)으로 16 kHz를 만드세요. 둘 다 FFT하세요. aliasing은 어디에 나타나나요?
3. **어려움.** `math`와 Step 3의 DFT만 사용해 STFT를 처음부터 만드세요. 프레임 크기 400, hop 160, Hann window를 사용합니다. `matplotlib.pyplot.imshow`로 크기를 그리세요. 이것이 Lesson 02의 스펙트로그램입니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Sample rate | 초당 샘플 개수 | ADC가 신호를 측정하는 Hz 단위 주파수입니다. |
| Nyquist | 표현할 수 있는 최대 주파수 | `sr/2`; 이를 넘는 에너지는 아래쪽으로 aliasing됩니다. |
| Bit depth | 각 샘플의 해상도 | `int16` = 65,536단계; `float32` = `[-1, 1]`에서 24-bit 정밀도. |
| DFT | 시퀀스용 푸리에 변환 | `N`개 샘플 → `N`개 복소 주파수 계수. |
| FFT | 빠른 DFT | `N` = 2의 거듭제곱을 요구하는 `O(N log N)` 알고리즘. |
| Bin | 주파수 열 | `k · sr / N` Hz; 해상도 = `sr / N`. |
| STFT | 스펙트로그램 내부 원리 | 시간에 따라 프레임화하고 윈도우를 씌운 FFT. |
| Aliasing | 이상한 주파수 유령 | Nyquist 위의 에너지가 낮은 bin으로 거울처럼 접혀 내려오는 현상. |

## 더 읽을거리

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 샘플링 정리의 배경이 되는 논문.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 무료로 볼 수 있는 표준 DSP 교재.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — 코드로 따라가는 실용 안내서.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 실제 오디오가 깨끗한 사인파가 아닌 이유를 설명하는 참고 자료.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 주파수 bin 직관을 10분 안에 정리해 주는 자료.
