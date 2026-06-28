# 오디오-언어 모델: Whisper에서 Audio Flamingo 3까지의 흐름

> Whisper(Radford et al., 2022년 12월)는 speech recognition을 정리했다. 68만 시간의 weakly-supervised multilingual speech, 단순한 encoder-decoder transformer, 이후 모든 ASR release가 인용하게 만든 benchmark가 있었다. 그러나 recognition은 reasoning이 아니다. "이 녹음에 어떤 악기가 있나", "화자가 어떤 감정을 표현하나", "3분 지점에 무슨 일이 있었나"라고 묻는 것은 transcription이 아니라 audio understanding을 요구한다. Qwen-Audio, SALMONN, LTU, NVIDIA의 Audio Flamingo 3(AF3, 2025년 7월)는 이 stack을 점진적으로 만들었다. Whisper급 encoder를 유지하고, Q-former를 붙이고, audio-text instruction data로 학습하며, chain-of-thought reasoning을 더했다. 이 lesson은 그 arc를 따라간다.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 minutes

## 학습 목표

- waveform에서 log-Mel spectrogram을 계산한다: windowing, FFT, filter bank, log transform.
- encoder option을 비교한다: Whisper encoder, BEATs, AF-Whisper hybrid. 각각 언제 이기는지 판단한다.
- audio Q-former를 만든다: N개의 learnable query가 spectrogram patch에 cross-attend한다.
- cascaded(Whisper-then-LLM)와 end-to-end audio-LLM training을 설명한다. reasoning에는 end-to-end가 왜 더 잘 scale되는지 이해한다.

## 문제

Speech recognition은 Whisper가 해결했다. OCR-of-audio는 commodity다. 하지만 "commodity"는 transcription에서 멈춘다. 모델이 들은 것에 대해 reasoning할 수 없다면, 즉 timing, speaker, emotion, music structure, environmental sound를 다루지 못한다면 transcription만으로는 제품 feature를 만들 수 없다.

명백한 route는 세 가지다.

1. Cascade: Whisper가 transcribe하고, LLM이 transcript 위에서 reasoning한다. pure-speech scenario에서는 동작한다. music, environmental audio, multi-speaker overlap, emotion에서는 실패한다.

2. End-to-end audio-LLM: audio encoder가 audio token을 직접 LLM에 넣어 transcription을 건너뛴다. acoustic information(emotion, speaker, environment)을 보존한다. 새 training data가 필요하다.

3. Hybrid: transcribe도 reasoning도 할 수 있는 audio encoder + text decoder. Qwen-Audio와 Audio Flamingo가 이 route를 택한다.

## 개념

### Log-Mel 스펙트로그램: 입력 특징

모든 audio encoder는 같은 feature에서 시작한다. log-Mel spectrogram이다.

1. 16 kHz로 resample한다.
2. 25ms window, 10ms hop으로 short-time Fourier transform을 수행한다.
3. FFT 결과의 magnitude를 취한다.
4. Mel filter bank(보통 0-8000 Hz에 log-spaced된 80 filter)를 적용해 perceptual frequency로 warp한다.
5. dynamic range를 위해 log compress(log(1 + x))한다.

결과는 shape (T, 80)의 2D array이며, T는 time frame 수다. 100 Hz frame rate에서 30초 clip은 (3000, 80)이다.

### Whisper의 encoder

Whisper의 encoder는 log-Mel spectrogram을 time frame sequence로 처리하는 12-layer ViT-style transformer다. 출력은 time frame마다 하나의 hidden-state vector다.

ASR에서는 Whisper decoder가 encoder output에 조건화되어 text token을 생성하는 cross-attention transformer다. 표준 encoder-decoder다.

ALM(audio-LLM)에서는 encoder output을 다른 LLM의 input으로 쓰고 싶다. 패턴은 frozen Whisper encoder, trainable Q-former, frozen 또는 tuned LLM이다.

### BEATs와 audio-specific encoder

Whisper는 speech-dominant data로 학습되었다. music과 environmental audio에는 약하다.

BEATs(Chen et al., 2022)는 AudioSet으로 학습한 self-supervised transformer다. 같은 parameter count에서 music과 environmental sound를 Whisper보다 잘 포착한다.

AF-Whisper(Audio Flamingo 3의 hybrid): Whisper + BEATs feature를 concat하여 audio input으로 사용한다. Whisper는 linguistic signal을, BEATs는 acoustic signal을 담당한다.

### 오디오 Q-Former

BLIP-2의 visual Q-former와 같은 패턴이다. 고정 개수의 learnable query(대개 32 또는 64)가 audio encoder의 output frame 전체에 cross-attend한다. 이 query가 LLM이 소비하는 audio token이 된다.

Training alignment stage: Q-former만 학습하며, audio-text pair(AudioCaps, Clotho)에 contrastive + captioning loss를 사용한다. Instruction stage: end-to-end로 LLM을 unfreeze하고 instruction data로 학습한다.

### Arc — SALMONN, Qwen-Audio, AF3

SALMONN(Tang et al., 2023): Whisper + BEATs + Q-former + LLaMA. 진지한 reasoning 능력을 가진 첫 오픈 audio-LLM이다. MMAU benchmark composite는 약 0.55다.

Qwen-Audio(Chu et al., 2023): 유사한 architecture지만 더 풍부한 dataset으로 학습하고 multi-turn dialogue에 맞게 tuned했다. MMAU 약 0.60.

LTU — Listen, Think, Understand(Gong et al., 2023): 명시적 reasoning data, audio clip에 대한 chain-of-thought에 집중한다. 더 작지만 더 집중된 모델이다.

Audio Flamingo 3(Goel et al., 2025년 7월): 현재 오픈 SOTA. 8B LLM backbone(Qwen2 7B), Whisper-large encoder concat BEATs, 64-query Q-former, 100만+ audio-text instruction pair로 학습. MMAU 0.72이며 일부 sub-task에서는 proprietary frontier와 맞먹는다.

AF3는 audio용 on-demand chain-of-thought도 도입한다. 모델은 최종 답 전에 thinking token("먼저 악기를 식별하자: ...")을 선택적으로 emit할 수 있다. thinking을 켜면 complex reasoning task accuracy가 3-5 point 오른다.

### 캐스케이드와 end-to-end

Cascaded pipeline:

1. Whisper가 audio를 text로 transcribe한다.
2. LLM이 text 위에서 reasoning한다.

"이 podcast를 요약해 줘"에는 완벽히 동작한다. 다음에는 실패한다.
- "이 곡의 mood는 어떤가?" — mood는 말이 아니라 소리에 있다.
- "Alice와 Bob 중 누가 말하고 있나?" — speaker identification이 필요하다.
- "폭발음은 몇 초에 발생하나?" — text에서 temporal grounding이 사라진다.
- "이것은 real audio인가 generated audio인가?" — deepfake detection은 acoustic feature가 필요하다.

End-to-end는 acoustic signal을 보존한다. Qwen-Audio와 AF3는 music, environment, emotion을 native하게 처리한다.

### 2026 production recipe

새 audio-understanding product 기준:

- 목표가 transcription이고, music도 emotion inference도 없다면 cascaded.
- music, emotion, multi-speaker, complex audio reasoning이 있다면 AF3 / Qwen-Audio-family.

Cascaded는 더 싸고 단순하다. End-to-end는 더 능력이 좋다.

### MMAU — audio reasoning benchmark

MMAU(Massive Multimodal Audio Understanding)는 2024-2025 audio reasoning benchmark다.

- speech, music, environmental sound 전반의 10,000개 audio-text QA pair.
- classification, temporal reasoning, causal reasoning, open-ended QA를 포함한다.
- cascaded pipeline이 체계적으로 놓치는 것을 테스트한다.

오픈 SOTA(AF3)는 0.72, proprietary frontier는 약 0.78(Gemini 2.5 Pro, Claude Opus 4.7)이다. 격차가 VideoMME의 open-vs-closed delta보다 작아 audio-LLM이 성숙 중임을 보여 준다.

## 활용하기

`code/main.py`:

- stdlib으로 log-Mel spectrogram computation을 구현한다: windowing, naive DFT, Mel filter-bank.
- Audio Q-former skeleton: encoder output frame이 주어지면 Q, K, V, attention을 계산하고 N token을 emit한다.
- toy task에서 cascaded-vs-end-to-end를 비교한다.

## 산출물

이 lesson은 `outputs/skill-audio-llm-pipeline-picker.md`를 만든다. audio task(transcription, music tagging, emotion inference, multi-speaker diarization, environment classification)가 주어지면 cascaded, end-to-end AF3, hybrid 중 하나를 고른다.

## 연습 문제

1. 16kHz, 25ms window, 10ms hop, 80 Mel bin에서 30초 clip의 log-Mel spectrogram dimension을 계산하라. 48kHz에서는 어떻게 바뀌는가?

2. Whisper가 music에서 성능이 낮은 이유는 무엇인가? BEATs는 Whisper가 포착하지 못하는 어떤 audio feature를 포착하는가?

3. 64 query audio Q-former와 32 query audio Q-former를 비교하라. 어떤 task complexity에서 64가 비용을 상쇄하는가? 32는 어떤 경우 compute를 절약하는가?

4. AF3 Section 4의 on-demand thinking을 읽어라. chain-of-thought가 가장 도움이 되는 audio task 세 가지를 제안하라.

5. AF3 output을 사용해 최소 diarization pipeline을 구현하라. speaker change를 어떻게 signal할 것인가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Mel features" | Mel filter bank 이후 log-magnitude value로 이루어진 2D (time, frequency) array |
| Audio Q-former | "Audio Perceiver" | audio encoder output에서 LLM으로 들어가는 fixed-length query로 가는 cross-attention bottleneck |
| Cascaded | "ASR-then-LLM" | Whisper가 transcribe하고 text LLM이 reasoning하는 pipeline. acoustic information을 잃는다 |
| End-to-end | "Audio-LLM" | audio feature가 Q-former를 통해 LLM에 직접 들어간다. acoustic signal을 보존한다 |
| BEATs | "Audio AudioSet encoder" | AudioSet으로 학습한 SSL transformer. music + environmental sound에 강하다 |
| MMAU | "Audio reasoning bench" | speech, music, environment 전반의 10k QA pair. 2024 eval standard |
| On-demand thinking | "Audio CoT" | 모델이 최종 답 전에 reasoning token을 선택적으로 emit할 수 있으며 accuracy를 3-5 point 올린다 |

## 더 읽을거리

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
