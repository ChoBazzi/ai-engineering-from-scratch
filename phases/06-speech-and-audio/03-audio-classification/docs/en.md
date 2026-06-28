# 오디오 분류 — MFCC 기반 k-NN부터 AST와 BEATs까지

> "개 짖는 소리 vs 사이렌"부터 "이 언어가 무엇인가"까지 모두 오디오 분류입니다. 특징은 mel입니다. 아키텍처는 10년마다 바뀝니다. 평가는 AUC, F1, 클래스별 recall로 남아 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 minutes

## 문제

10초짜리 클립을 받았습니다. 알고 싶은 것은 "이게 무엇인가?"입니다. 도시 소리(사이렌, 드릴, 개), 음성 명령(yes/no/stop), language ID(en/es/ar), 화자 감정(angry/neutral), 환경음(indoor/outdoor, babble)이 모두 여기에 해당합니다. 이 모든 것이 *오디오 분류*이며, 2026년의 baseline 아키텍처는 성숙해 있습니다. log-mel → CNN 또는 Transformer → softmax입니다.

핵심 난점은 네트워크가 아닙니다. 데이터입니다. 오디오 데이터셋에는 심한 클래스 불균형, 강한 도메인 shift(clean vs noisy), 레이블 noise("urban babble"과 "restaurant noise"를 누가 구분했나요?)가 있습니다. 문제의 80%는 CNN을 Transformer로 바꾸는 것이 아니라 큐레이션, 증강, 평가입니다.

## 개념

![오디오 분류 사다리: MFCC 기반 k-NN에서 AST와 BEATs까지](../assets/audio-classification.svg)

**MFCC 기반 k-NN(1990년대 baseline).** 클립별 MFCC를 평탄화하고, 레이블이 있는 bank와 cosine similarity를 계산한 뒤, 상위 K개의 다수결을 반환합니다. 깨끗하고 작은 데이터셋(Speech Commands, ESC-50)에서 놀랄 만큼 강합니다. GPU 없이 실행됩니다.

**Log-mel 위의 2D CNN(2015-2019).** `(T, n_mels)` log-mel을 이미지처럼 다룹니다. ResNet-18 또는 VGG 스타일을 적용합니다. 시간 축에 global mean pool을 적용합니다. 클래스에 대해 softmax를 수행합니다. 2026년 대부분의 Kaggle 대회에서도 여전히 baseline입니다.

**Audio Spectrogram Transformer, AST(2021-2024).** Log-mel을 패치화하고(예: 16×16 패치), position embedding을 더한 뒤 ViT에 넣습니다. 지도학습 기준 AudioSet에서 state of the art(mAP 0.485)였습니다.

**BEATs와 WavLM-base(2024-2026).** 수백만 시간 규모로 자기지도 사전학습합니다. 원래 필요했을 지도 데이터의 1-10%만으로도 자신의 작업에 fine-tune할 수 있습니다. 2026년에는 비음성 오디오의 기본 출발점입니다. BEATs-iter3는 AudioSet에서 AST보다 1-2 mAP 높으면서 compute는 1/4만 사용합니다.

**Frozen backbone으로 쓰는 Whisper encoder(2024).** Whisper의 encoder를 가져오고 decoder를 버린 뒤 linear classifier를 붙입니다. 오디오 증강 없이도 language ID와 단순 event classification에서 near-SOTA입니다. "공짜 점심" baseline입니다.

### 클래스 불균형이 진짜 과제입니다

ESC-50: 50개 클래스, 클래스마다 40개 클립입니다. 균형 잡혀 있고 쉽습니다. UrbanSound8K: 10개 클래스, 10:1 불균형입니다. AudioSet: 632개 클래스와 100,000:1 long tail이 있습니다. 잘 작동하는 기법은 다음과 같습니다.

- 학습 중 balanced sampling을 사용합니다(평가에는 사용하지 않음).
- Mixup: 두 클립과 그 레이블을 선형 보간하여 증강합니다.
- SpecAugment: 무작위 시간 및 주파수 band를 mask합니다. 단순하지만 중요합니다.

### 평가

- 배타적 다중 클래스(Speech Commands): top-1 accuracy, top-5 accuracy.
- 다중 클래스 multi-label(AudioSet, UrbanSound 스타일): mean average precision(mAP).
- 심한 불균형: 클래스별 recall + macro F1.

알아 두어야 할 2026년 수치는 다음과 같습니다.

| 벤치마크 | 기준선 | SOTA 2026 | 출처 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs 논문(2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 결과 |

## 직접 만들기

### 1단계: 특징화하기

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 2단계: 고정 길이 요약

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

단순하지만 강력합니다. 시간 축의 mean + variance는 13계수 MFCC에 대해 26차원 고정 임베딩을 만듭니다. 즉시 실행됩니다. 2017년에도 ESC-50에서 state-of-the-art NN baseline을 이겼습니다.

### 3단계: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 4단계: log-mel 위의 CNN으로 업그레이드하기

PyTorch에서는 다음과 같습니다.

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

파라미터 3M개입니다. 단일 RTX 4090에서 ESC-50을 약 10분 만에 학습합니다. 정확도는 80% 이상입니다.

### 5단계: 2026년 기본값 — BEATs fine-tune

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

BEATs는 `beats` 라이브러리를 통해 `microsoft/BEATs-base`를 사용하세요. transformers API와 형태가 같습니다.

## 활용하기

2026년 스택은 다음과 같습니다.

| 상황 | 시작점 |
|-----------|-----------|
| 작은 데이터셋(<1000 clips) | MFCC mean 기반 k-NN(자신의 baseline) + 오디오 증강 |
| 중간 데이터셋(1K-100K) | BEATs 또는 AST fine-tune |
| 큰 데이터셋(>100K) | 처음부터 학습하거나 Whisper-encoder fine-tune |
| 실시간, edge | int8로 quantize한 40-MFCC CNN(KWS 스타일) |
| Multi-label(AudioSet) | BCE loss + mixup + SpecAugment를 적용한 BEATs-iter3 |
| Language ID | MMS-LID, SpeechBrain VoxLingua107 baseline |

결정 규칙: **새 모델이 아니라 frozen backbone으로 시작하세요**. BEATs head를 fine-tune하면 몇 주가 아니라 몇 시간 안에 SOTA의 95%에 도달합니다.

## 결과물

`outputs/skill-classifier-designer.md`로 저장합니다. 주어진 오디오 분류 작업에 맞춰 아키텍처, 증강, 클래스 균형 전략, 평가 지표를 고릅니다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 코드는 4클래스 합성 데이터셋(서로 다른 pitch의 순수 tone)에서 k-NN MFCC baseline을 학습합니다. confusion matrix를 보고하세요.
2. **보통.** `summarize`를 [mean, var, skew, kurtosis]로 바꾸세요. 같은 합성 데이터셋에서 4-moment pooling이 mean+var보다 나은가요?
3. **어려움.** `torchaudio`를 사용해 ESC-50 fold 1에서 2D CNN을 학습하세요. 5-fold cross-validation accuracy를 보고하세요. SpecAugment(time mask = 20, freq mask = 10)를 추가하고 delta를 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| AudioSet | 오디오의 ImageNet | Google의 2M 클립, 632클래스 weakly-labeled YouTube 데이터셋. |
| ESC-50 | 작은 분류 benchmark | 환경음 50클래스 × 40클립. |
| AST | Audio Spectrogram Transformer | log-mel 패치 위의 ViT; 2021년 SOTA. |
| BEATs | 자기지도 오디오 | Microsoft 모델, iter3가 2026년 기준 AudioSet 선두. |
| Mixup | 쌍 증강 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Mask 기반 증강 | 스펙트로그램의 무작위 시간 및 주파수 band를 0으로 만듭니다. |
| mAP | 주요 multi-label 지표 | 클래스와 threshold 전반의 mean average precision. |

## 더 읽을거리

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021-2024년의 대표 아키텍처.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024년 이후 기본값.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 지배적인 오디오 증강법.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — 여전히 쓰이는 50클래스 benchmark.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 632클래스 YouTube taxonomy; 여전히 gold standard입니다.
