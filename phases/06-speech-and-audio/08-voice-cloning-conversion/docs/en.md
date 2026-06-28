# 음성 클로닝과 음성 변환

> Voice cloning은 누군가의 목소리로 텍스트를 읽는다. Voice conversion은 말한 내용을 보존하면서 당신의 목소리를 다른 사람의 목소리로 다시 쓴다. 둘 다 같은 분해에 달려 있다. speaker identity를 content에서 분리하는 것이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 minutes

## 문제

2026년에는 5초짜리 오디오 클립만으로도 consumer GPU에서 누구의 목소리든 고품질 clone을 만들 수 있다. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox는 모두 zero-shot 또는 few-shot cloning을 제공한다. 이 기술은 축복이자(accessibility TTS, dubbing, assistive voices) 무기다(scam call, 정치 deepfake, IP theft).

밀접하게 관련된 두 작업이 있다.

- **Voice cloning(TTS 쪽):** text + 5초 reference voice → 그 목소리의 audio.
- **Voice conversion(speech 쪽):** source audio(person A가 X를 말함) + person B의 reference voice → B가 X를 말하는 audio.

둘 다 waveform을 (content, speaker, prosody)로 factorize하고, 한 source의 content를 다른 source의 speaker와 재결합한다.

2026년 배포 시 반드시 따라야 하는 핵심 제약: **watermarking과 consent gate는 EU(AI Act, 2026년 8월 enforceable)와 California(AB 2905, 2025년 effective)에서 법적으로 요구된다**. pipeline은 들리지 않는 watermark를 내보내야 하며, 동의 없는 clone을 거부해야 한다.

## 개념

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning.** 수천 명의 화자로 학습된 모델에 5초 clip을 넘긴다. speaker encoder가 clip을 speaker embedding으로 mapping하고, TTS decoder가 그 embedding과 text를 조건으로 사용한다.

사용 모델: F5-TTS(2024), YourTTS(2022), XTTS v2(2024), OpenVoice v2(2024).

**Few-shot fine-tuning.** target voice를 5-30분 녹음한다. base model을 LoRA로 1시간 파인튜닝한다. 품질은 "괜찮음"에서 "구별 불가"로 뛰어오른다. Coqui와 ElevenLabs 모두 이 pattern을 지원하며, 커뮤니티는 F5-TTS와 함께 사용한다.

**Voice conversion(VC).** 두 계열이 있다.

- **Recognition-synthesis.** ASR 비슷한 모델로 content representation(예: soft phoneme posterior, PPG)을 추출한 뒤, target speaker embedding으로 다시 합성한다. 언어와 억양에 견고하다. KNN-VC(2023), Diff-HierVC(2023)가 사용한다.
- **Disentanglement.** bottleneck의 latent space에서 content, speaker, prosody를 분리하는 autoencoder를 학습한다. inference에서 speaker embedding을 바꾼다. 품질은 낮지만 빠르다. AutoVC(2019), VITS-VC 변형이 사용한다.

**Neural codec 기반 cloning(2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox는 audio를 SoundStream / EnCodec의 discrete token으로 취급하고, codec token 위에 큰 autoregressive 또는 flow-matching model을 학습한다. 짧은 prompt에서는 ElevenLabs에 비견되는 품질을 낸다.

### 덧붙임이 아닌 윤리

**Watermarking.** PerTh(Perth)와 SilentCipher(2024)는 오디오에 약 16-32 bit ID를 지각 불가능하게 embed한다. re-encoding, streaming, 일반적인 edit에도 살아남는다. production-ready open source다.

**Consent gates.** 모든 cloned output은 검증 가능한 consent record와 짝지어야 한다. "I, Rohit, on 2026-04-22, authorize this voice for X purpose." 같은 형태다. tamper-evident log에 저장한다.

**Detection.** AASIST, RawNet2, Wav2Vec2-AASIST가 detector로 배포된다. ASVspoof 2025 challenge는 ElevenLabs, VALL-E 2, Bark output을 상대로 한 state-of-the-art detector의 EER가 0.8-2.3%라고 발표했다.

### 숫자(2026)

| 모델 | Zero-shot 여부 | SECS(대상 유사도) | WER(명료도) | 파라미터 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70이면 대부분의 청자에게 일반적으로 target과 구별되지 않는다.

## 직접 만들기

### 1단계: recognition-synthesis로 분해하기(`main.py`의 code-only demo)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

개념은 단순하다. 구현의 대부분은 `tts_model`과 speaker encoder 안에 있다.

### 2단계: F5-TTS로 zero-shot clone하기

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript는 audio와 정확히 일치해야 한다. mismatch는 alignment를 깨뜨린다.

### 3단계: KNN-VC로 voice conversion하기

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC는 WavLM을 실행해 source와 target pool의 per-frame embedding을 추출한 다음, 각 source frame을 pool에서 가장 가까운 이웃으로 바꾼다. non-parametric이며 target speech 1분으로 동작한다.

### 4단계: watermark embed하기

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

payload는 약 32 bit이며, MP3 re-encode와 약한 noise 뒤에도 detect된다.

### 5단계: consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 사용하기

2026년의 stack:

| 상황 | 선택 |
|-----------|------|
| 5초 zero-shot clone, open-source | F5-TTS 또는 OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion(rewriting) | KNN-VC 또는 Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 또는 VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## 함정

- **Misaligned reference transcript.** F5-TTS와 유사 모델은 reference text가 punctuation까지 포함해 reference audio와 정확히 일치해야 한다.
- **Reverberant reference.** echo는 clone을 망친다. dry하고 close-mic으로 녹음한다.
- **Emotional mismatch.** "cheerful"한 training reference는 모든 것의 cheerful clone을 만든다. reference emotion을 target use와 맞춘다.
- **Language leakage.** English speaker를 clone한 뒤 French를 말하게 하면 accent가 그대로 남는 경우가 많다. cross-lingual model(XTTS, VALL-E X)을 사용한다.
- **Watermark 없음.** 2026년 8월부터 EU에서 법적으로 배포 불가다.

## 배포하기

`outputs/skill-voice-cloner.md`로 저장한다. consent gate + watermark + quality target을 갖춘 cloning 또는 conversion pipeline을 설계한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행한다. 두 "speaker" 사이의 cosine을 swap 전후로 계산해 speaker-embedding swap을 시연한다.
2. **중간.** OpenVoice v2로 자신의 목소리를 clone한다. reference와 clone 사이의 SECS를 측정한다. Whisper를 통해 CER를 측정한다.
3. **어려움.** 20개 clone에 SilentCipher watermark를 적용하고, 128 kbps MP3 encode+decode를 거친 뒤 payload를 detect한다. bit-accuracy를 보고한다.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-------------------|-----------|
| Zero-shot clone | 5초면 충분 | pretrained model + speaker embedding. 학습 없음. |
| PPG | Phonetic posteriorgram | language-agnostic content representation으로 쓰이는 per-frame ASR posterior. |
| KNN-VC | Nearest-neighbor conversion | 각 source frame을 가장 가까운 target-pool frame으로 바꾼다. |
| Neural codec TTS | VALL-E style | EnCodec/SoundStream token 위의 AR model. |
| Watermark | 들리지 않는 서명 | audio에 embed되어 re-encode 후에도 살아남는 bit. |
| SECS | Cloning fidelity | target과 clone speaker embedding 사이의 cosine. |
| AASIST | Deepfake detector | 합성 음성을 감지하는 anti-spoof model. |

## 더 읽기

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA zero-shot cloning.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — disentanglement-based voice conversion.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — detector와 synthesizer의 arms race, 2026년 업데이트.
