# 음성 anti-spoofing과 오디오 watermarking — ASVspoof 5, AudioSeal, WaveVerify

> 음성 클로닝은 방어책보다 빠르게 배포되었다. 2026년 프로덕션 음성 시스템에는 두 가지가 필요하다. 실제 음성과 가짜 음성을 분류하는 감지기(AASIST, RawNet2), 그리고 압축과 편집을 견디는 watermark(AudioSeal)다. 둘 다 넣어 배포하거나, 음성 클로닝을 배포하지 말라.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 minutes

## 문제

관련된 세 가지 방어책:

1. **Anti-spoofing / deepfake detection.** 오디오 클립이 주어졌을 때 합성인가, 실제인가? ASVspoof 벤치마크(ASVspoof 2019 → 2021 → 5)가 표준이다.
2. **Audio watermarking.** 생성된 오디오 안에 사람이 들을 수 없는 신호를 삽입하고, 나중에 감지기가 추출할 수 있게 한다. AudioSeal(Meta)과 WavMark가 공개 선택지다.
3. **Authenticated provenance.** 오디오 파일 + 메타데이터의 암호학적 서명. C2PA / Content Authenticity Initiative.

감지는 협조하지 않는 공격자를 다룬다. Watermarking은 규정 준수를 다룬다. AI 생성 오디오는 그렇게 식별 가능해야 한다. 2026년에는 둘 다 필요하다.

## 개념

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — 2024-2025 벤치마크

이전 판본과 비교한 가장 큰 변화:

- **크라우드소싱 데이터**(스튜디오급 깨끗한 데이터가 아님) — 현실적인 조건.
- **약 2000명의 화자**(이전에는 약 100명).
- **32개 공격 알고리즘.** TTS + voice conversion + adversarial perturbation.
- **두 트랙.** Countermeasure(CM) 단독 감지. 생체 인식 시스템을 위한 Spoofing-robust ASV(SASV).

ASVspoof 5의 최신 성능은 약 7.23% EER이다. 더 오래된 ASVspoof 2019 LA에서는 0.42% EER이다. 실제 배포에서는 in-the-wild 클립에서 5-10% EER을 예상하라.

### AASIST와 RawNet2 — 감지 모델 계열

**AASIST**(2021, 2026년까지 업데이트). 스펙트럼 feature 위의 graph-attention. 현재 ASVspoof 5 countermeasure task의 SOTA다.

**RawNet2.** 원시 파형 위의 convolutional front-end + TDNN backbone. 더 단순한 기준선이지만 fine-tuning하면 여전히 경쟁력 있다.

**NeXt-TDNN + SSL features.** 2025년 변형: ECAPA 스타일 + WavLM features + focal loss. ASVspoof 2019 LA에서 0.42% EER을 달성한다.

### AudioSeal — 2024년 watermark 기본값

Meta의 **AudioSeal**(2024년 1월, v0.2는 2024년 12월). 핵심 설계:

- **국소적이다.** 16kHz 샘플 해상도(1/16000초)에서 프레임 단위로 watermark를 감지한다.
- **Generator + detector 공동 학습.** Generator는 들리지 않는 신호를 삽입하도록 배우고, detector는 augmentation을 거쳐 이를 찾도록 배운다.
- **견고하다.** MP3 / AAC 압축, EQ, ±10% speed-shift, +10dB SNR 잡음 혼합을 견딘다.
- **빠르다.** Detector는 실시간 485배 속도로 실행된다. WavMark보다 1000배 빠르다.
- **용량.** 16-bit payload(모델 ID, 생성 timestamp, 사용자 ID를 인코딩 가능)를 각 발화에 삽입할 수 있다.

### WavMark

AudioSeal 이전의 공개 기준선이다. Invertible neural network, 32 bits/sec. 문제:

- 동기화를 brute-force로 찾는 과정이 느리다.
- Gaussian noise나 MP3 압축으로 제거될 수 있다.
- 실시간 친화적이지 않다.

### WaveVerify (2025년 7월)

AudioSeal의 약점, 특히 시간 조작(반전, 속도)을 다룬다. FiLM 기반 generator + Mixture-of-Experts detector를 사용한다. 표준 공격에서는 AudioSeal과 경쟁 가능하고, 시간 편집을 처리한다.

### 공격자가 노리는 빈틈

AudioMarkBench에 따르면 "pitch shift에서 모든 watermark는 0.6 미만의 Bit Recovery Accuracy를 보이며, 거의 완전한 제거를 나타낸다." **Pitch-shift는 범용 공격이다.** 2026년의 어떤 watermark도 공격적인 pitch modification에 완전히 견고하지 않다. 그래서 watermarking과 함께 감지(AASIST)가 필요하다.

### C2PA / Content Authenticity Initiative

ML 기법이 아니라 manifest 형식이다. 오디오 파일은 생성 도구, 작성자, 날짜에 관한 암호학적으로 서명된 메타데이터를 담는다. Audobox / Seamless가 사용한다. 출처 증명에는 좋지만, 악의적 행위자가 재인코딩해서 메타데이터를 제거하면 아무것도 하지 못한다.

## 직접 만들기

### 1단계: 단순한 스펙트럼 특징 감지기(toy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

합성 음성은 종종 고주파 에너지가 비정상적으로 평평하다. 프로덕션 감지기는 이것이 아니라 AASIST를 쓴다. 하지만 직관은 유효하다.

### 2단계: AudioSeal 삽입 + 감지

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — watermark 존재 확률
# decoded_payload: 16 bits; 삽입한 payload와 비교
```

### 3단계: 평가 — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 4단계: 프로덕션 통합

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

모든 생성물은 (1) watermark, (2) 서명된 manifest, (3) 보존 정책을 준수하는 audit log를 함께 제공한다.

## 활용하기

| 사용 사례 | 방어책 |
|----------|---------|
| TTS / voice cloning 배포 | 모든 출력에 AudioSeal 삽입(타협 불가) |
| 생체 음성 잠금 해제 | AASIST + ECAPA ensemble; liveness challenge |
| 콜센터 사기 감지 | 수신 통화의 20% 샘플에 AASIST |
| 팟캐스트 진위성 | 업로드 시 C2PA signing, AI 생성이면 AudioSeal |
| 감지기 연구 / 학습 | ASVspoof 5 train/dev/eval sets |

## 함정

- **Watermark를 넣고 detector를 전혀 실행하지 않음.** 무의미하다. CI에 detector를 배포하라.
- **보정 없는 감지.** ASVspoof LA로 학습한 AASIST는 과적합된다. 실제 정확도는 떨어진다. 도메인에서 보정하라.
- **Pitch-shift 빈틈.** 공격적인 pitch shift는 대부분의 watermark를 제거한다. 감지 fallback을 둬라.
- **Metadata strip-and-rehost.** C2PA는 재인코딩으로 쉽게 우회된다. 항상 암호학적 방어 + 지각적 방어(watermark)를 함께 추가하라.
- **Liveness를 감지로 착각.** 사용자에게 임의 문구를 말하게 하라. replay attack은 막지만 실시간 클로닝은 막지 못한다.

## 내보내기

`outputs/skill-spoof-defender.md`로 저장하라. 음성 생성 배포를 위한 감지 모델, watermark, provenance manifest, 운영 playbook을 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. 합성 오디오에서 toy detector + toy watermark 삽입/감지를 수행한다.
2. **보통.** `audioseal`을 설치하고 TTS 출력에 16-bit payload를 삽입한 뒤 다시 디코딩하라. 오디오에 잡음을 넣고 Bit Recovery Accuracy를 측정하라.
3. **어려움.** ASVspoof 2019 LA에서 RawNet2 또는 AASIST를 fine-tune하라. EER을 측정하라. F5-TTS로 생성한 별도 hold-out 클립 세트에서 테스트하고 OOD 감지가 얼마나 나빠지는지 확인하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| ASVspoof | 벤치마크 | 격년 challenge. 2024 = ASVspoof 5. |
| CM (countermeasure) | 감지기 | 분류기: 실제 음성 vs 합성 / 변환 음성. |
| SASV | Speaker verif + CM | 통합 생체 인식 + spoof 감지. |
| AudioSeal | Meta watermark | 국소적, 16-bit payload, WavMark보다 485× 빠름. |
| Bit Recovery Accuracy | Watermark 생존성 | 공격 후 복구된 payload bit의 비율. |
| C2PA | Provenance manifest | 생성 / 저작권에 관한 암호학적 메타데이터. |
| AASIST | 감지기 계열 | Graph-attention 기반 anti-spoofing SOTA. |

## 더 읽을거리

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 현재 벤치마크.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — watermark 기본값.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 시간 공격을 위한 MoE detector.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — SOTA 감지 backbone.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 견고성 평가.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — provenance manifest 형식.
