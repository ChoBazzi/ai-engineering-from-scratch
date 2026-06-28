# 음성 인식(ASR) — CTC, RNN-T, Attention

> 음성 인식은 매 timestep마다 수행하는 오디오 분류를, 영어와 침묵을 아는 시퀀스 모델로 이어 붙인 것입니다. CTC, RNN-T, attention은 이를 수행하는 세 가지 방법입니다. 하나를 고르고 왜 그런지 이해하세요.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 minutes

## 문제

10초짜리 16 kHz 클립이 있습니다. 원하는 것은 문자열입니다. "turn on the kitchen lights". 어려움은 구조적입니다. 오디오 프레임은 문자와 일대일로 정렬되지 않습니다. "okay"라는 단어는 200 ms가 걸릴 수도 있고 1200 ms가 걸릴 수도 있습니다. 침묵은 발화를 끊어 줍니다. 어떤 음소는 다른 음소보다 깁니다. 출력 토큰 수는 미리 알 수 없습니다.

이 문제를 푸는 세 가지 정식화가 있습니다.

1. **CTC(Connectionist Temporal Classification).** 특별한 *blank*를 포함해 프레임별 토큰 확률을 냅니다. 디코딩 시 반복과 blank를 접습니다. Non-autoregressive이고 빠릅니다. wav2vec 2.0, MMS에서 사용합니다.
2. **RNN-T(Recurrent Neural Network Transducer).** Joint network가 encoder frame과 이전 토큰을 바탕으로 다음 토큰을 예측합니다. 스트리밍 가능합니다. Google의 온디바이스 ASR, NVIDIA Parakeet에서 사용합니다.
3. **Attention encoder-decoder.** Encoder가 오디오를 hidden state로 압축하고, decoder가 cross-attention으로 autoregressive하게 토큰을 생성합니다. Whisper, SeamlessM4T에서 사용합니다.

2026년 LibriSpeech test-clean의 SOTA WER는 1.4%(Parakeet-TDT-1.1B, NVIDIA)와 1.58%(Whisper-Large-v3-turbo)입니다. 차이는 작지만 배포상의 차이는 큽니다.

## 개념

![세 가지 ASR 정식화: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 직관.** Encoder가 `V+1`개 토큰(V개 문자 + blank)에 대한 `T`개의 프레임 수준 분포를 출력한다고 합시다. 길이 `U < T`인 target string `y`에 대해, 접었을 때 `y`가 되는 모든 프레임 정렬이 인정됩니다. CTC loss는 그런 모든 정렬에 대해 합산합니다. 추론은 프레임별 argmax, 반복 접기, blank 제거입니다.

장점: non-autoregressive, streamable, zero lookahead입니다. 단점: *조건부 독립 가정*입니다. 각 프레임 예측이 다른 프레임과 독립이라 내부 언어 모델이 없습니다. Beam search 또는 shallow fusion으로 외부 LM을 붙여 보완합니다.

**RNN-T 직관.** 토큰 이력을 임베딩하는 *predictor* 네트워크와 predictor state를 encoder frame과 결합해 `V+1`에 대한 joint distribution을 만드는 *joiner*를 추가합니다(`+1`은 null / no-emit). CTC가 무시한 조건부 의존성을 명시적으로 모델링합니다. 각 step이 과거 프레임과 과거 토큰에만 조건화되므로 스트리밍 가능합니다.

장점: streamable + internal LM입니다. 단점: 학습이 더 복잡하고 메모리를 많이 씁니다(3D loss lattice). RNN-T loss kernel은 그 자체로 하나의 라이브러리 범주입니다.

**Attention encoder-decoder.** Log-mel 프레임 위의 encoder(6-32개 transformer layer)와 decoder(6-32개 transformer layer)로 구성됩니다. Decoder는 encoder 출력에 cross-attention하여 autoregressive하게 토큰을 생성합니다. 정렬 제약이 없습니다. Attention은 오디오 어디든 볼 수 있습니다. Attention을 제한하지 않으면 streamable하지 않습니다(chunked Whisper-Streaming, 2024).

장점: offline ASR에서 최고 품질이고 표준 seq2seq 도구로 학습하기 쉽습니다. 단점: autoregressive latency가 출력 길이에 비례하며, 엔지니어링 없이는 스트리밍할 수 없습니다.

### WER: 하나의 숫자

**Word Error Rate** = `(S + D + I) / N`입니다. 여기서 S=substitutions, D=deletions, I=insertions, N=reference word count입니다. 단어 수준 Levenshtein edit distance와 같습니다. 낮을수록 좋습니다. WER가 20%를 넘으면 일반적으로 사용할 수 없고, 5% 미만이면 낭독 음성에서 human-parity입니다. 표준 benchmark의 2026년 수치는 다음과 같습니다.

| 모델 | LibriSpeech test-clean | LibriSpeech test-other | 크기 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

이들은 모두 encoder-decoder 또는 RNN-T 기반입니다. 순수 CTC 시스템(wav2vec 2.0)은 test-clean에서 약 1.8-2.1% 수준입니다.

## 직접 만들기

### 1단계: greedy CTC 디코딩

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

규칙은 두 가지입니다. 연속 반복을 접고 blank를 버립니다. 예: `a a _ _ a b b _ c` → `a a b c`.

### 2단계: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

프로덕션에서는 LM fusion이 포함된 prefix tree beam search를 사용합니다. 이것은 개념적 골격입니다.

### 3단계: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 4단계: Whisper로 추론하기

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026년 가장 강력한 범용 ASR을 위한 한 줄입니다. 24 GB GPU에서 약 20× realtime으로 실행됩니다.

### 5단계: Parakeet 또는 wav2vec 2.0으로 스트리밍하기

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

Streaming ASR에는 chunked encoder attention과 carryover state가 필요합니다. 이를 지원하는 라이브러리를 사용하세요(Parakeet용 NeMo, `chunk_length_s`를 사용하는 `transformers` pipeline).

## 활용하기

2026년 스택은 다음과 같습니다.

| 상황 | 선택 |
|-----------|------|
| 영어, offline, 최고 품질 | Whisper-large-v3-turbo |
| 다국어, robust | SeamlessM4T v2 |
| 스트리밍, 낮은 latency | Parakeet-TDT-1.1B 또는 Riva |
| Edge, mobile, <500 ms latency | quantized Whisper-Tiny 또는 Moonshine(2024) |
| Long-form | VAD 기반 chunking을 적용한 Whisper(WhisperX) |
| 도메인 특화(의료, 법률) | wav2vec 2.0 fine-tune + domain LM fusion |

## 2026년에도 배포되는 함정

- **VAD 없음.** 침묵 구간에서 Whisper를 실행하면 hallucination("Thanks for watching!")이 생깁니다. 항상 VAD로 gate하세요.
- **문자 vs 단어 vs subword WER.** 정규화(소문자화, 구두점 제거) *후* 단어 수준 WER를 보고하세요.
- **Language ID drift.** Whisper의 자동 LID는 noisy clip을 일본어나 웨일스어로 잘못 보낼 수 있습니다. 언어를 알고 있다면 `language="en"`을 강제하세요.
- **Chunking 없는 긴 클립.** Whisper는 30초 window를 갖습니다. 그보다 긴 경우 `chunk_length_s=30, stride=5`를 사용하세요.

## 결과물

`outputs/skill-asr-picker.md`로 저장합니다. 주어진 배포 목표에 맞춰 모델, decoding 전략, chunking, LM fusion을 고릅니다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하세요. 이 코드는 손으로 만든 CTC 출력을 greedy decode하고 reference와의 WER를 계산합니다.
2. **보통.** Step 2의 prefix-tree beam search를 올바르게 구현하세요(blank merge rule을 고려). 10개 예제 합성 데이터셋에서 greedy와 비교하세요.
3. **어려움.** [LibriSpeech test-clean](https://www.openslr.org/12)에 `whisper-large-v3-turbo`를 사용하세요. 처음 100개 utterance의 WER를 계산하세요. 공개 수치와 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| CTC | Blank-token loss | 모든 frame-to-token 정렬에 대한 marginal; non-AR. |
| RNN-T | Streaming loss | CTC + next-token predictor; 어순을 다룹니다. |
| Attention enc-dec | Whisper 스타일 | Encoder + cross-attending decoder; offline 품질이 가장 좋습니다. |
| WER | 보고하는 숫자 | 단어 수준의 `(S+D+I)/N`. |
| Blank | 비어 있음 | CTC에서 "이 프레임은 emission 없음"을 나타내는 특수 토큰. |
| LM fusion | 외부 언어 모델 | Beam search 중 weighted LM log-prob을 더합니다. |
| VAD | 침묵 gate | Voice activity detector; 비음성을 잘라냅니다. |

## 더 읽을거리

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — CTC 논문.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — RNN-T 논문.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — 2022년 표준 논문; 2024년에 v3-turbo 확장.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — 2026 Open ASR Leaderboard 선두.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 25개 이상 모델을 비교하는 실시간 benchmark.
