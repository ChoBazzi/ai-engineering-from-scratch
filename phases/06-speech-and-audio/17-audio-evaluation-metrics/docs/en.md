# 오디오 평가 — WER, MOS, UTMOS, MMAU, FAD, 그리고 공개 리더보드

> 측정할 수 없는 것은 배포할 수 없다. 이 lesson은 모든 오디오 작업의 2026년 지표를 정리한다. ASR(WER, CER, RTFx), TTS(MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language(MMAU, LongAudioBench), 음악(FAD, CLAP), 화자(EER). 그리고 비교할 수 있는 leaderboard까지 다룬다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 minutes

## 문제

모든 오디오 작업에는 여러 지표가 있고, 각 지표는 서로 다른 축을 측정한다. 잘못된 지표를 쓰면 dashboard에서는 훌륭해 보이지만 프로덕션에서는 형편없는 모델을 배포하게 된다. 2026년의 정식 목록:

| 작업 | 주 지표 | 보조 지표 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| 음성 클로닝 | SECS (ECAPA cosine) | MOS · CER |
| 화자 검증 | EER | minDCF · FAR / FRR at operating point |
| 화자 분리 | DER | JER · speaker confusion |
| 오디오 분류 | top-1 · mAP | macro F1 · per-class recall |
| 음악 생성 | FAD | CLAP · listening panel MOS |
| 오디오 언어 모델 | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 개념

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 지표

**WER (Word Error Rate).** `(S + D + I) / N`. 점수 계산 전에 소문자화하고, 구두점을 제거하며, 숫자를 정규화한다. `jiwer`나 OpenAI의 `whisper_normalizer`를 사용하라. &lt; 5%는 낭독 음성에서 인간 수준이다.

**CER (Character Error Rate).** 같은 공식이지만 문자 단위다. 단어 분할이 모호한 성조 언어(중국어 표준어, 광둥어)에 사용한다.

**RTFx (inverse real-time factor).** wall-clock 1초당 처리한 오디오 초 수. 높을수록 좋다. Parakeet-TDT는 3380×에 도달한다. Whisper-large-v3는 약 30×다.

**First-token latency.** 오디오 입력부터 첫 전사 token까지의 wall-clock 시간. 스트리밍에서 중요하다. Deepgram Nova-3는 약 150ms다.

### TTS 지표

**MOS (Mean Opinion Score).** 사람이 매기는 1-5점 평가. 표준이지만 느리다. 모델당 100개 이상의 sample에 대해 sample마다 20명 이상의 청취자를 모아라.

**UTMOS (2022-2026).** 학습된 MOS 예측기. 표준 벤치마크에서 인간 MOS와 약 0.9 상관을 보인다. F5-TTS: UTMOS 3.95. ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** 음성 클로닝용. reference와 clone 출력 사이의 ECAPA embedding cosine. &gt; 0.75면 알아볼 수 있는 clone이다.

**WER-on-ASR-round-trip.** TTS 출력에 Whisper를 실행하고 입력 텍스트와 WER을 계산한다. 명료도 회귀를 잡아낸다. 2026년 SOTA는 &lt; 2% CER이다.

**TTFA (time-to-first-audio).** wall-clock 지연 시간. Kokoro-82M은 약 100ms, F5-TTS는 약 1초다.

### 음성 클로닝 전용

**SECS + MOS + CER**를 세트로 본다. SECS는 높지만 MOS가 낮은 클로닝은 음색은 맞지만 부자연스럽다는 뜻이다. 반대는 자연스럽지만 화자가 틀렸다는 뜻이다.

### 화자 검증

**EER (Equal Error Rate).** False Accept Rate가 False Reject Rate와 같아지는 임계값이다. VoxCeleb1-O의 ECAPA: 0.87%.

**minDCF (min Detection Cost).** 선택한 operating point(보통 FAR=0.01)에서의 가중 비용이다. EER보다 프로덕션 관련성이 높다.

### 화자 분리

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. 놓친 음성 + false-alarm 음성 + 화자 혼동을 각각 비율로 합친다. AMI meeting에서는 DER 약 10-20%가 현실적이다. pyannote 3.1 + Precision-2 commercial은 잘 녹음된 오디오에서 &lt;10% DER이다.

**JER (Jaccard Error Rate).** DER의 대안이며 짧은 segment bias에 더 견고하다.

### 오디오 분류

Multi-label: 모든 class에 대한 **mAP (mean Average Precision)**. AudioSet: BEATs-iter3가 0.548 mAP.

상호 배타적 multi-class: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1(Audio-MAE).

불균형 데이터: **macro F1** + **per-class recall**. class별로 보고하라. aggregate accuracy는 어떤 class가 실패하는지 숨긴다.

### 음악 생성

**FAD (Fréchet Audio Distance).** 실제 오디오와 생성 오디오의 VGGish embedding 분포 사이 거리. MusicCaps의 MusicGen-small: 4.5. MusicLM: 4.0. 낮을수록 좋다.

**CLAP Score.** CLAP embedding을 사용한 텍스트-오디오 정렬 점수. &gt; 0.3이면 합리적인 정렬이다.

**Listening panel MOS.** 소비자용 음악에서는 여전히 최종 판단이다. Suno v5는 TTS Arena에서 paired human preference 기준 ELO 1293이다.

### 오디오-언어 벤치마크

**MMAU (Massive Multi-Audio Understanding).** 1만 개 audio-QA 쌍.

**MMAU-Pro.** 어려운 1800개 항목, 네 범주: speech / sound / music / multi-audio. 4지선다 무작위 확률은 25%. Gemini 2.5 Pro 전체는 약 60%, multi-audio는 모든 모델에서 약 22%.

**LongAudioBench.** 의미 질의가 있는 여러 분 길이의 클립. Audio Flamingo Next가 Gemini 2.5 Pro를 앞선다.

**AudioCaps / Clotho.** 캡셔닝 벤치마크. SPICE, CIDEr, FENSE 지표.

### 스트리밍 speech-to-speech

**Latency P50 / P95 / P99.** 사용자 발화 종료부터 첫 가청 응답까지의 wall-clock 시간. Moshi: 200ms. GPT-4o Realtime: 300ms.

**WER / MOS**를 출력에 적용한다.

**Barge-in responsiveness.** 사용자가 끼어든 시점부터 assistant mute까지의 시간. 목표는 &lt; 150ms.

### 2026년 리더보드

| 리더보드 | 트랙 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | English + multilingual + long-form | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | English TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO from paired votes | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised audio | `hearbenchmark.com` |

## 직접 만들기

### 1단계: 정규화가 포함된 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 2단계: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 3단계: 음성 클로닝용 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 4단계: 음악 생성용 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 5단계: 화자 검증용 EER(Lesson 6과 같은 코드)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 활용하기

모든 배포에는 모델 업데이트마다 실행되는 고정 eval harness를 붙여라. 세 가지 기본 규칙:

1. **점수 계산 전에 정규화하라.** 소문자화, 구두점 제거, 숫자 확장. 정규화 규칙을 보고하라.
2. **평균이 아니라 분포를 보고하라.** 지연 시간은 P50/P95/P99. 분류는 per-class recall. MMAU는 범주별.
3. **하나의 정식 공개 벤치마크를 실행하라.** 프로덕션 데이터가 다르더라도 Open ASR / TTS Arena / MMAU에 보고하면 리뷰어가 같은 기준으로 비교할 수 있다.

## 함정

- **UTMOS 외삽.** VCTK 스타일의 깨끗한 음성에서 학습했다. 잡음/클론/감정 오디오 점수는 부정확하다.
- **MOS panel bias.** Amazon Mechanical Turk 작업자 20명은 대상 사용자 20명이 아니다. 중요도가 높다면 도메인 panel에 비용을 지불하라.
- **FAD는 reference set에 의존한다.** 모델 간 비교에는 같은 reference distribution을 사용하라.
- **Aggregate WER.** 전체 WER 5%가 억양 있는 음성의 WER 30%를 숨길 수 있다. demographic slice별로 보고하라.
- **공개 벤치마크 포화.** 대부분의 frontier model은 표준 벤치마크에서 천장에 가깝다. 실제 traffic을 반영하는 내부 hold-out set을 만들어라.

## 내보내기

`outputs/skill-audio-evaluator.md`로 저장하라. 모든 오디오 모델 release에 대해 지표, 벤치마크, 보고 형식을 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. toy input에서 WER / CER / EER / SECS / FAD-ish / MMAU-ish를 계산한다.
2. **보통.** TTS round-trip WER harness를 만들어라. Kokoro 또는 F5-TTS 출력을 Whisper에 통과시켜라. 50개 prompt에 대해 WER을 계산하고, WER &gt; 10%인 prompt를 표시하라.
3. **어려움.** Lesson 10의 LALM 선택지를 MMAU-Pro speech + multi-audio subset에서 채점하라(각 50개 항목). 범주별 accuracy를 보고하고 공개 수치와 비교하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| WER | ASR 점수 | 정규화 후 단어 수준 `(S+D+I)/N`. |
| CER | 문자 WER | 성조 언어나 문자 수준 시스템용. |
| MOS | 사람 의견 | 1-5점 평가. 20명 이상 청취자 × 100 samples. |
| UTMOS | ML MOS 예측기 | 학습된 모델. 인간 MOS와 약 0.9 상관. |
| SECS | Voice-clone similarity | reference와 clone 사이의 ECAPA cosine. |
| EER | Speaker verif score | FAR = FRR이 되는 임계값. |
| DER | Diarization score | (FA + Miss + Confusion) / total. |
| FAD | Music-gen quality | VGGish embedding 위의 Fréchet distance. |
| RTFx | 처리량 | wall-clock 1초당 오디오 초 수. |

## 더 읽을거리

- [jiwer](https://github.com/jitsi/jiwer) — 정규화 유틸리티가 있는 WER/CER 라이브러리.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — 학습된 MOS 예측기.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — 음악 생성 표준.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026년 live ranking.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 사람 투표 기반 TTS leaderboard.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — LALM reasoning leaderboard.
- [HEAR benchmark](https://hearbenchmark.com/) — audio SSL benchmark.
