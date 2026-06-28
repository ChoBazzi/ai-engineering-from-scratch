# 옴니 모델: Qwen2.5-Omni와 Thinker-Talker 분리

> 2024년 5월의 GPT-4o 제품 데모가 파괴적이었던 이유는 underlying model 때문이 아니라 product shape 때문이었다. 사용자가 말하고, 모델이 camera가 보는 것을 보고, 250ms 안에 다시 말해 주는 voice interface였다. 오픈 생태계는 2024년과 2025년 내내 그 product surface에 도달하기 위해 달렸다. Qwen2.5-Omni(2025년 3월)는 기준이 되는 오픈 설계다. Thinker(대형 text-generating transformer)와 Talker(병렬 speech-generating transformer)를 두고, streaming speech token으로 연결한다. Mini-Omni는 이를 단순화했고, Moshi는 latency를 맞췄으며, GLM-4-Voice는 중국어로 확장했다. 이 lesson은 Thinker-Talker architecture와 streaming real-time dialogue를 가능하게 하는 latency budget을 읽는다.

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 minutes

## 학습 목표

- inference pipeline을 Thinker(text reasoning)와 Talker(speech synthesis)로 나누고, parallel streaming이 왜 동작하는지 설명한다.
- 대화형 interaction의 time-to-first-audio-byte(TTFAB) budget을 component별로 계산한다.
- Thinker 안에서 vision, audio, text 전반에 time-aligned position encoding을 제공하는 TMRoPE를 설명한다.
- 세 가지 real-time conversational pattern을 말한다: half-duplex, turn-taking, full-duplex.

## 문제

real-time voice assistant는 많은 일을 빠르게 해야 한다.

1. 사용자의 말을 듣는다. real-time speech tokenization과 사용자가 말을 끝냈는지 알기 위한 voice activity detection(VAD)이 필요하다.
2. 선택적으로 본다. camera input이 2-4 FPS로 들어오고 audio와 함께 Thinker로 stream된다.
3. 생각한다. conversation history에 조건화된 response를 구성한다.
4. 말한다. audio token을 synthesize하고, waveform으로 decode하고, 사용자 speaker로 stream한다.

각 step은 latency를 더한다. 대화처럼 느끼려면 total round-trip이 500ms 미만이어야 한다. 그 아래에서는 사용자가 lag를 잘 느끼지 못한다. GPT-4o는 약 250ms를 주장한다. Moshi는 약 160ms다. Qwen2.5-Omni는 약 350-500ms다.

모든 component가 stream해야 한다. "모두 batch한 뒤 decode"하는 방식은 불가능하다.

## 개념

### Thinker와 Talker

Qwen2.5-Omni의 decomposition:

- Thinker: 7B-80B text-generating transformer. interleaved text + image + audio token을 소비한다. 무엇을 말할지 나타내는 text token을 출력한다.
- Talker: 더 작은 speech-generating transformer(200M-1B). Thinker의 text output token과 최근 speech-context token을 소비한다. discrete speech token(residual-VQ index)을 출력한다.
- Speech decoder: speech token을 실시간 audio sample로 바꾸는 streaming waveform decoder(SNAC, MoVQGAN 계열).

분리가 중요하다. Thinker는 좋은 reasoning을 위해 커야 한다. Talker는 text를 speech token으로 바꾸는 local job을 하므로 작아도 된다. 더 큰 Talker가 더 expressive한 것이 아니라 더 느릴 뿐이다.

둘을 병렬로 실행한다.

1. Thinker가 text token t_i를 emit한다.
2. Talker가 t_i를 streaming으로 소비하고 speech token s_i, s_{i+1}, ..., s_{i+k}를 emit한다.
3. Speech decoder가 도착하는 speech token을 소비하고 audio sample을 emit한다.
4. Thinker가 text token t_{i+3}에 도달할 때쯤 Talker는 이미 t_0..t_{i+2}에 대한 audio를 stream했다.

### TMRoPE — time-aligned multimodal position

Thinker는 image frame(예: 4 FPS로 도착), audio frame(초당 50 frame), conversation history의 text를 통합해야 한다. naive sequence order(모든 image, 그다음 모든 audio, 그다음 text)는 temporal alignment를 잃는다.

TMRoPE는 모든 token에 absolute timestamp를 부여한다. t=2.3s의 vision token. t=2.32s의 audio token. t=2.35s에 사용자가 말한 "stop" text token. RoPE가 timestamp로 attention을 rotate하므로 모델은 이들을 시간적으로 동시에 일어난 것으로 본다.

이것이 "그가 hello라고 말하면서 손을 흔들었다"가 동작하기 위한 infrastructure다. 모델은 같은 conceptual moment의 video frame과 audio를 본다.

### 스트리밍 음성 합성

Speech token은 반드시 stream되어야 한다. Mini-Omni(Xie & Wu, 2024)는 "language models can hear, talk while thinking in streaming"을 도입했다. Thinker output token과 Talker output token이 같은 sequence에서 interleave된다. Thinker가 다음 text token을 commit하는 즉시 Talker가 동작한다. batch boundary가 없다.

Moshi(Défossez et al., 2024년 10월)는 가장 빠른 오픈 구현이다. 단일 A100에서 160ms TTFAB. Architecture는 text token과 speech token을 alternating position에 emit하는 단일 7B transformer이며, thinking stream과 speaking stream을 분리하는 "inner monologue"를 갖는다. 이는 사실상 신중한 학습으로 Thinker + Talker를 하나의 모델에 fuse한 것이다.

### VAD와 turn-taking

Voice activity detection은 input side에서 실행된다. 두 가지 pattern:

- Half-duplex: 사용자가 말하면 모델은 듣고, 모델이 말하면 사용자는 듣는다. VAD silence detection(~200ms)으로 명확히 handoff한다.
- Full-duplex: 양쪽이 동시에 말할 수 있다. 모델은 backchannel("uh-huh")을 하거나 interrupt할 수 있다. 훨씬 어렵다. Moshi가 이를 지원한다.

Qwen2.5-Omni는 기본적으로 half-duplex를 지원하고, silence threshold를 통한 turn-taking을 사용한다. Full-duplex는 application-layer handling이 필요하다.

### Qwen3-Omni(2025년 11월)

후속 모델이다. Qwen3-80B Thinker, 더 큰 Talker, 개선된 TMRoPE-v2. Latency는 GPT-4o의 250ms에 가깝다. open weights다. OmniBench benchmark에서 Gemini 2.0 Live와 경쟁적이다.

### 프로덕션 지연 시간 예산

일반적인 streaming interaction:

- Mic -> audio tokens: 40-80ms.
- Prefill(prompt + history): 7B에서 100-200ms, 70B에서는 훨씬 더 큼.
- First Thinker text token: 40ms.
- Talker processes first text token: 20ms.
- First speech tokens commit: 40ms.
- Residual-VQ decode: 30ms.
- Speech waveform decode: 50-80ms.

총 TTFAB: 7B에서 320-510ms, 70B에서 600-900ms. Frontier quality는 보통 70B+를 의미한다. 이것이 frontier latency gap의 이유다.

### 토큰 속도 계산

50 Hz base speech token의 16kHz speech에서는 output 1초당 50 speech token이 필요하다. Talker는 따라잡기 위해 ≥50 tok/s를 emit해야 한다. H100에서 일반적인 LLM throughput이 30-80 tok/s라면 작은(200-300M) Talker는 충분히 빠르다. 7B Talker는 뒤처진다.

이것이 "그냥 main model을 쓰면 되지"가 아니라 small dedicated Talker model이 존재하는 이유다.

## 활용하기

`code/main.py`:

- mock token-emission rate로 Thinker-Talker pipeline을 시뮬레이션한다.
- configurable model size와 mic sample rate에 대해 TTFAB를 계산한다.
- VAD silence threshold를 사용한 half-duplex turn-taking을 시연한다.

## 산출물

이 lesson은 `outputs/skill-omni-streaming-budget.md`를 만든다. real-time voice product의 target TTFAB와 feature set(vision-in, bilingual, full-duplex)이 주어지면 Qwen2.5-Omni, Qwen3-Omni, Moshi, Mini-Omni 중 하나를 고르고 Thinker/Talker 크기를 정한다.

## 연습 문제

1. target TTFAB가 300ms다. 7B Thinker와 300M Talker에서 모든 component의 latency를 적어라.

2. Qwen2.5-Omni는 TMRoPE를 사용한다. 사용자가 t=1s에 말하기 시작하고 camera가 t=1.2s에 gesture를 포착하는 prompt에서 모델이 무엇을 보는지 설명하라.

3. Full-duplex support는 모델이 들으면서 audio를 emit해야 한다. 이를 가르치는 training data format을 제안하라.

4. Moshi paper Section 4를 읽어라. "inner monologue" separation과 이것이 Thinker-Talker split을 피하는 이유를 설명하라.

5. throughput budget을 계산하라. 16kHz speech에서 50 base-layer tokens/sec를 따라가려면 Talker는 얼마나 빠르게 token을 emit해야 하는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Thinker | "Reasoning brain" | 무엇을 말할지 생성하는 대형 text-generating transformer |
| Talker | "Speech-generating mouth" | Thinker의 text에서 discrete speech token을 생성하는 작은 transformer |
| TTFAB | "Latency budget" | Time-to-first-audio-byte: 사용자 speech end부터 첫 audio sample output까지의 시간 |
| TMRoPE | "Time-aligned RoPE" | vision, audio, text 전반에서 absolute timestamp를 사용하는 position encoding |
| Half-duplex | "Turn-taking" | 사용자와 모델이 번갈아 말한다. VAD silence가 user-done을 감지한다 |
| Full-duplex | "Simultaneous" | 모델이 동시에 말하고 들을 수 있으며 backchannel이 가능하다 |
| Inner monologue | "Moshi separation" | thinking-stream과 speaking-stream이 interleave되는 single-model design |

## 더 읽을거리

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
