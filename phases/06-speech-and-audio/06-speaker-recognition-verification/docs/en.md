# 화자 인식과 검증

> ASR은 "무엇을 말했는가?"를 묻는다. speaker recognition은 "누가 말했는가?"를 묻는다. 수학은 embedding과 cosine으로 비슷해 보이지만, production 결정은 모두 하나의 EER 숫자에 달려 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 minutes

## 문제

사용자가 passphrase를 말한다. 알고 싶은 것은 이것이다. 이 사람이 주장하는 그 사람인가(*verification*, 1:1), 아니면 enrollment bank의 첫 번째 사람인가(*identification*, 1:N)? 혹은 둘 다 아니어서 알 수 없는 화자인가(*open-set*)?

2018년 이전에는 GMM-UBM + i-vector가 쓰였다. EER는 괜찮았지만 channel shift(전화 vs 노트북)와 감정에 취약했다. 2018-2022년에는 x-vector(angular margin으로 학습한 TDNN backbone)가 중심이었다. 2022년 이후에는 ECAPA-TDNN과 WavLM-large embedding이 주류가 되었다. 2026년에는 이 분야가 세 모델과 하나의 metric으로 정리된다.

그 metric은 **EER**(Equal Error Rate)이다. False Accept Rate = False Reject Rate가 되도록 decision threshold를 설정한다. 그 교차점이 EER다. 모든 paper, leaderboard, procurement call에서 사용된다.

## 개념

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline.** Enrollment: 목표 화자의 음성을 5-30초 녹음하고, 고정 차원 embedding을 계산한다(ECAPA-TDNN은 192-d, WavLM-large는 256-d). Verification: test utterance embedding을 얻고, cosine similarity를 계산한 뒤 threshold와 비교한다.

**ECAPA-TDNN(2020, 2026년에도 지배적).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. squeeze-excitation이 있는 1D conv block, multi-head attention pooling, 192-d로 가는 linear layer로 구성된다. VoxCeleb 1+2(화자 2,700명, utterance 1.1M)에서 Additive Angular Margin loss(AAM-softmax)로 학습되었다.

**WavLM-SV(2022+).** 사전학습된 WavLM-large SSL backbone을 AAM loss로 파인튜닝한다. 품질은 더 높지만 느리다. 300+ MB 대 15 MB다.

**x-vector(baseline).** TDNN + statistics pooling. 고전적인 방식이며, CPU / edge에서는 여전히 유용하다.

**AAM-softmax.** angular space에서 correct class에 margin `m`을 더한 표준 softmax다. 즉 `cos(θ + m)`을 사용한다. class 사이의 angular separation을 강제한다. 일반적으로 `m=0.2`, scale `s=30`을 쓴다.

### 점수화

- **Cosine**: enrollment embedding과 test embedding 사이의 값. threshold 기반 decision에 사용한다.
- **PLDA(Probabilistic LDA).** 같은 화자와 다른 화자의 likelihood ratio가 closed form으로 계산되는 latent space로 embedding을 project한다. cosine 위에 얹으면 EER를 10-20% 줄일 수 있다. 2020년 이전 표준이었고, 지금은 closed-set setup에서만 주로 사용된다.
- **Score normalization.** `S-norm` 또는 `AS-norm`: 각 score를 imposter cohort의 평균과 표준편차에 대해 정규화한다. cross-domain 평가에 필수다.

### 알아야 할 숫자(2026)

| 모델 | VoxCeleb1-O EER | 파라미터 | 처리량(A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### 화자 분리

다중 화자 클립에서 "누가 언제 말했는가"를 찾는 작업이다. Pipeline: VAD → segment → 각 segment embedding → clustering(agglomerative 또는 spectral) → boundary smoothing. 현대 stack은 speaker segmentation + embedding + clustering을 한 번의 호출로 묶는 `pyannote.audio` 3.1이다. 2026년 AMI 기준 SOTA DER는 약 15%다(2022년 23%에서 하락).

## 직접 만들기

### 1단계: MFCC 통계로 toy embedding 만들기

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

SOTA와는 한참 거리가 있다. 교육용일 뿐이다. `code/main.py`는 synthetic speaker data에서 이를 proof-of-concept로 사용한다.

### 2단계: cosine similarity + threshold

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 3단계: similarity pair에서 EER 계산하기

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

`(eer, threshold_at_eer)`를 반환한다. 둘 다 보고한다.

### 4단계: SpeechBrain으로 production 구성하기

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 5단계: pyannote로 diarization하기

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 사용하기

2026년의 stack:

| 상황 | 선택 |
|-----------|------|
| Closed-set 1:1 verification, edge | ECAPA-TDNN + cosine threshold |
| Open-set verification, cloud | WavLM-SV + AS-norm |
| Diarization(meetings, podcasts) | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing(replay / deepfake detection) | AASIST 또는 RawNet2 |
| Tiny embedded(KWS + enrollment) | Titanet-Small(NeMo) |

## 함정

- **Channel mismatch.** VoxCeleb(web video)로 학습한 모델은 전화 통화 오디오와 같지 않다. 항상 목표 channel에서 평가한다.
- **짧은 utterance.** test audio가 3초 미만이면 EER가 급격히 나빠진다.
- **잡음 있는 enrollment.** 잡음 섞인 enrollment 하나가 anchor를 오염시킨다. 깨끗한 샘플 3개 이상을 평균낸다.
- **조건을 가로지르는 고정 threshold.** target domain의 held-out dev set에서 항상 threshold를 튜닝한다.
- **정규화되지 않은 embedding에 cosine 적용.** 먼저 L2-normalize한다. 그렇지 않으면 magnitude가 지배한다.

## 배포하기

`outputs/skill-speaker-verifier.md`로 저장한다. model, enrollment protocol, threshold-tuning plan, fraud safeguard를 고른다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행한다. synthetic "speaker"(서로 다른 tone profile)를 만들고, enroll한 뒤, 100-pair trial list에서 EER를 계산한다.
2. **중간.** 30개 VoxCeleb1 utterance(화자 5명 × 각 6개)에 SpeechBrain ECAPA를 사용한다. cosine과 PLDA의 EER를 계산한다.
3. **어려움.** `pyannote.audio`로 enroll → diarize → verify 전체 pipeline을 만든다. AMI dev set에서 DER를 평가한다.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-------------------|-----------|
| EER | 대표 metric | False Accept = False Reject가 되는 threshold. |
| Verification | 1:1 | "이 사람이 Alice인가?" |
| Identification | 1:N | "누가 말하고 있는가?" |
| Open-set | Unknown 가능 | Test set에 enrollment되지 않은 화자가 있을 수 있다. |
| Enrollment | 등록 | 화자의 reference embedding을 계산하는 것. |
| AAM-softmax | Loss | additive angular margin을 붙인 softmax. cluster separation을 강제한다. |
| PLDA | 고전 scoring | Probabilistic LDA. embedding 위의 likelihood-ratio scoring. |
| DER | Diarization metric | Diarization Error Rate. miss + false alarm + confusion. |

## 더 읽기

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — 고전적인 deep-embedding paper.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — 2020-2026년의 지배적 아키텍처.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — SV와 diarization을 위한 SSL backbone.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — production diarization + embedding stack.
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — 모델별 최신 EER 순위.
