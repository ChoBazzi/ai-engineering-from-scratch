# 스펙트로그램, Mel 스케일 및 오디오 특징

> 신경망은 원시 파형을 잘 소비하지 못합니다. 신경망은 스펙트로그램을 소비합니다. Mel 스펙트로그램은 더 잘 소비합니다. 2026년의 모든 ASR, TTS, 오디오 분류기는 이 단 하나의 전처리 선택에 성패가 달려 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 minutes

## 문제

10초짜리 16 kHz 클립을 생각해 봅시다. 이는 `[-1, 1]` 범위의 float 160,000개이며, "개 짖는 소리"나 "cat이라는 단어" 같은 레이블과는 거의 완전히 상관이 없어 보입니다. 원시 파형에는 정보가 있지만 모델이 쉽게 추출할 수 없는 형태로 들어 있습니다. 100 ms 차이로 발화된 동일한 음소 두 개는 원시 샘플이 완전히 다릅니다.

스펙트로그램은 이 문제를 해결합니다. 인간 지각이 무시하는 시간 세부사항(마이크로초 단위 jitter)은 접고, 인간 지각이 주목하는 구조(약 10-25 ms 시간 창에서 어떤 주파수에 에너지가 있는지)는 보존합니다.

Mel 스펙트로그램은 한 걸음 더 나아갑니다. 인간은 음높이를 로그적으로 지각합니다. 100 Hz와 200 Hz의 차이는 1000 Hz와 2000 Hz의 차이와 "같은 거리"처럼 들립니다. Mel 스케일은 이에 맞게 주파수 축을 휘게 만듭니다. Mel 스케일 스펙트로그램은 2010년부터 2026년까지 음성 ML에서 가장 중요한 단일 특징입니다.

## 개념

![파형에서 STFT, Mel 스펙트로그램, MFCC로 이어지는 사다리](../assets/mel-features.svg)

**STFT(Short-Time Fourier Transform).** 파형을 겹치는 프레임으로 자릅니다(일반적으로 25 ms window, 10 ms hop = 16 kHz에서 400샘플 / 160샘플). 각 프레임에 윈도우 함수를 곱합니다(Hann이 기본값이고, Hamming은 약간 다른 절충을 합니다). 각 프레임에 FFT를 적용합니다. 크기 스펙트럼을 `(n_frames, n_freq_bins)` 형태의 행렬로 쌓습니다. 그것이 스펙트로그램입니다.

**로그 크기.** 원시 크기는 5-6자릿수 범위에 걸쳐 있습니다. `log(|X| + 1e-6)` 또는 `20 * log10(|X|)`를 취해 동적 범위를 압축합니다. 모든 프로덕션 파이프라인은 원시 크기가 아니라 로그 크기를 사용합니다.

**Mel 스케일.** Hz 단위 주파수 `f`는 `m = 2595 * log10(1 + f / 700)`로 mel `m`에 매핑됩니다. 이 매핑은 1 kHz 아래에서는 대략 선형이고, 그 위에서는 대략 로그적입니다. 0-8 kHz를 덮는 80개 mel bin이 표준 ASR 입력입니다.

**Mel filterbank.** Mel 스케일에서 균등하게 배치된 삼각 필터 집합입니다. 각 필터는 인접한 FFT bin의 가중합입니다. STFT 크기에 filterbank 행렬을 곱하면 한 번의 행렬곱으로 Mel 스펙트로그램을 얻습니다.

**Log-mel 스펙트로그램.** `log(mel_spec + 1e-10)`. Whisper의 입력입니다. Parakeet의 입력입니다. SeamlessM4T의 입력입니다. 2026년의 범용 오디오 프런트엔드입니다.

**MFCC.** Log-mel 스펙트로그램에 DCT(type II)를 적용하고 처음 13개 계수를 유지합니다. 특징 간 상관을 줄이고 더 압축합니다. 원시 log-mel 위의 CNN/Transformer가 따라잡기 전인 2015년 무렵까지 지배적인 특징이었습니다. 여전히 화자 인식(x-vectors, ECAPA)에 사용됩니다.

**해상도 절충.** FFT가 클수록 주파수 해상도는 좋아지지만 시간 해상도는 나빠집니다. 25 ms / 10 ms는 오디오 ML 기본값이고, 음악에는 50 ms / 12.5 ms, transient 감지(드럼 타격, 파열음)에는 5 ms / 2 ms를 씁니다.

```figure
spectrogram-window
```

## 직접 만들기

### 1단계: 파형을 프레임으로 나누기

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

`frame_len=400, hop=160`인 10초 16 kHz 클립은 998개 프레임을 만듭니다.

### 2단계: Hann 윈도우

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

FFT 전에 원소별로 곱합니다. 0이 아닌 끝점에서 잘라내며 생기는 spectral leakage를 줄입니다.

### 3단계: STFT 크기

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

프로덕션에서는 `torch.stft` 또는 `librosa.stft`를 사용합니다(FFT 기반, 벡터화됨). 여기의 루프는 교육용이며 `code/main.py`의 짧은 클립에서 실행됩니다.

### 4단계: Mel 필터뱅크

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

`n_fft=400`으로 0-8 kHz를 덮는 80개 mel은 `(80, 201)` 행렬을 만듭니다. `(n_frames, 201)` STFT 크기에 이 행렬의 전치를 곱하면 `(n_frames, 80)` Mel 스펙트로그램을 얻습니다.

### 5단계: log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

흔한 대안은 `librosa.power_to_db`(reference-normalized dB), `10 * log10(power + eps)`입니다. Whisper는 더 복잡한 clip + normalize 루틴을 사용합니다(Whisper의 `log_mel_spectrogram` 참고).

### 6단계: MFCC

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

각 log-mel 프레임에 DCT를 적용하고 처음 13개 계수를 유지합니다. 그것이 MFCC 행렬입니다. 첫 번째 계수는 보통 버립니다(전체 에너지를 인코딩하기 때문입니다).

## 활용하기

2026년 스택은 다음과 같습니다.

| 작업 | 특징 |
|------|----------|
| ASR(Whisper, Parakeet, SeamlessM4T) | 80 log-mel, 10 ms hop, 25 ms window |
| TTS acoustic model(VITS, F5-TTS, Kokoro) | 정밀한 시간 제어를 위한 80 mel, 5-12 ms hop |
| 오디오 분류(AST, PANNs, BEATs) | 128 log-mel, 10 ms hop |
| 화자 임베딩(ECAPA-TDNN, WavLM) | 80 log-mel 또는 원시 파형 SSL |
| 음악(MusicGen, Stable Audio 2) | EnCodec 이산 토큰(mel 아님) |
| 키워드 spotting | 작은 기기용 40 MFCC |

경험칙: **음악을 다루는 것이 아니라면 80 log-mel부터 시작하세요.** 다른 선택을 하려면 그쪽이 입증 책임을 져야 합니다.

## 2026년에도 배포되는 함정

- **Mel 개수 불일치.** 학습은 80 mel로 하고 추론은 128 mel로 하는 경우입니다. 조용히 실패합니다. 양쪽 끝에서 특징 shape를 로그로 남기세요.
- **업스트림 샘플링 레이트 불일치.** 22.05 kHz에서 계산한 mel은 16 kHz와 다르게 보입니다. 특징화 *전에* SR을 고치세요.
- **dB vs log.** Whisper는 dB-mel이 아니라 log-mel을 기대합니다. 일부 HF 파이프라인은 자동 감지하지만, 직접 작성한 코드는 그렇지 않습니다.
- **정규화 drift.** 학습 중에는 발화별 정규화, 추론 중에는 전역 정규화를 쓰는 경우입니다. WER를 두 배로 만드는 프로덕션 버그입니다.
- **패딩 누출.** 클립 끝을 zero-padding하면 뒤쪽 프레임에 평평한 스펙트럼이 생깁니다. 대칭으로 패딩하거나 복제하세요.

## 결과물

`outputs/skill-feature-extractor.md`로 저장합니다. 이 스킬은 주어진 모델 목표에 맞춰 특징 유형, mel 개수, 프레임/hop, 정규화를 고릅니다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 코드는 chirp(주파수가 200 → 4000 Hz로 sweep됨)를 합성하고 프레임마다 argmax mel bin을 출력합니다. (선택 사항으로) 그려 보고 sweep과 일치하는지 확인하세요.
2. **보통.** `n_mels`를 `{40, 80, 128}`, `frame_len`을 `{200, 400, 800}`으로 바꿔 다시 실행하세요. 시간 축을 따라 sharp peak bandwidth를 측정하세요. 어떤 조합이 chirp를 가장 잘 분해하나요?
3. **어려움.** `power_to_db`를 구현하고 AudioMNIST에서 작은 CNN 분류기의 ASR 정확도를 (a) 원시 log-mel, (b) `ref=max`를 사용한 dB-mel, (c) MFCC-13 + delta + delta-delta로 비교하세요. top-1 정확도를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Frame | 조각 | 하나의 FFT에 들어가는 25 ms 파형 덩어리. |
| Hop | Stride | 연속 프레임 사이의 샘플 수; ASR 기본값은 10 ms입니다. |
| Window | Hann/Hamming 같은 것 | 프레임 가장자리를 0으로 가늘게 줄이는 점별 곱셈자. |
| STFT | 스펙트로그램 생성기 | 프레임화 + 윈도우 적용 FFT; 시간 × 주파수 행렬을 만듭니다. |
| Mel | 휘어진 주파수 | 로그 지각 스케일; `m = 2595·log10(1 + f/700)`. |
| Filterbank | 그 행렬 | STFT를 mel bin으로 투영하는 삼각 필터. |
| Log-mel | Whisper의 입력 | `log(mel_spec + eps)`; 2026년에 표준화됨. |
| MFCC | 구식 특징 | log-mel의 DCT; 13개 계수, 상관 제거됨. |

## 더 읽을거리

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — MFCC 논문.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — 원래 Mel 스케일.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — 참조 구현을 읽어 보세요.
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — `mfcc`, `melspectrogram`, hop/window 참고 자료.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — Parakeet + Canary 모델용 프로덕션 규모 파이프라인.
