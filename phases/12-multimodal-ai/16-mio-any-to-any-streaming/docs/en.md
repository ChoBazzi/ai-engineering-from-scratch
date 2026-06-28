# MIO와 Any-to-Any 스트리밍 멀티모달 모델

> GPT-4o는 대부분의 오픈 모델이 재현할 수 없던 제품을 내놓았다. 음성을 듣고, 영상을 보고, 실시간으로 말해 주는 에이전트다. 2024년 말 오픈 생태계의 답은 MIO(Wang et al., 2024년 9월)였다. MIO는 텍스트, 이미지, 음성, 음악을 토큰화하고, 인터리브된 시퀀스 위에서 하나의 causal transformer를 학습하며, 어떤 modality에서 어떤 modality로든 생성한다. AnyGPT(Zhan et al., 2024년 2월)는 개념 증명이었고, MIO는 그 스케일업이며, Unified-IO 2(Allen AI, 2023년 12월)는 vision + action grounding을 갖춘 사촌 격이다. 이 lesson은 any-to-any 패턴, 즉 네 개의 tokenizer, 하나의 transformer, 스트리밍 친화적 decode를 읽는다.

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 minutes

## 학습 목표

- 텍스트, 이미지, 음성, 음악 토큰을 충돌 없이 담는 공유 vocabulary를 설계한다.
- SEED-Tokenizer(이미지)와 SpeechTokenizer residual-VQ(음성)를 압축 + 재구성 trade-off 관점에서 비교한다.
- any-to-any 생성을 단계적으로 만드는 네 단계 curriculum을 설명한다.
- 세 가지 오픈 any-to-any recipe와 핵심 trade-off를 말한다: MIO, AnyGPT, Unified-IO 2.

## 문제

통합 멀티모달 모델은 말하기는 쉽지만 대규모로 만들기는 어렵다. 2024년까지 대부분의 "any-to-any" 시스템은 pipeline이었다. vision model → text representation → speech model → audio. 각 hop은 정보를 잃고, latency를 더하며, 학습을 복잡하게 만든다. GPT-4o의 데모 영상은 1초 미만 응답이 가능한 단일 모델 대안을 보여 주었고, 오픈 시스템은 몇 달 뒤처져 있었다.

엔지니어링 과제:

- 모든 modality마다 tokenizer가 있어야 하고, 재구성에 충분할 만큼 정보를 보존하며 압축해야 하며, transformer가 소비할 수 있는 속도로 토큰을 만들어야 한다.
- 하나의 vocabulary가 텍스트(32k+), 이미지(16k+), 음성(4k+), 음악(8k+)을 위한 공간을 배정해야 한다. 최소 4만 개 이상의 entry가 필요하다.
- 학습 데이터는 모든 input-output pair(text→image, image→speech, speech→image 등)를 포함해야 하거나, 모델이 조합할 수 있어야 한다.
- 추론은 대화형 latency(<500ms time-to-first-audio-byte)를 만족할 만큼 빠르게 output token을 스트리밍해야 한다.

## 개념

### 네 modality를 위한 네 tokenizer

MIO의 tokenizer stack:

- 텍스트: 표준 BPE, vocab ~32000.
- 이미지: SEED-Tokenizer(2023) — discrete codebook 4096개를 가진 quantized VAE, 이미지당 32x32 token.
- 음성: SpeechTokenizer residual-VQ(2023) — 16kHz waveform을 8개의 계층적 codebook으로 인코딩한다. 첫 level은 거친 content이고, 이후 level은 prosody와 speaker identity를 더한다.
- 음악: 유사한 residual-VQ(Meta의 MusicGen / Encodec 계열), 4-8개 codebook.

각 modality는 정수 token을 만든다. 이 token들은 공유 vocabulary 안에서 서로 겹치지 않는 ID range를 받는다.

```text
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

총 vocabulary는 약 48k다. input embedding과 output projection이 이 전체 범위를 포괄한다.

### 스트리밍 디코딩

음성 생성은 residual-VQ를 사용한다. transformer는 base(layer 0) speech token을 예측하고, 병렬 decode되는 residual quantizer가 이후 layer를 예측한다. 각 layer 0 token은 16kHz audio에서 대략 50ms에 해당한다.

스트리밍 패턴:

1. 사용자가 마이크에 말하면 real-time audio tokenizer가 50ms마다 speech token을 내보낸다.
2. MIO는 도착하는 즉시 token을 소비한다(prompt prefill + incremental forward).
3. 생성된 output token이 스트리밍되고, 병렬 speech decoder가 약 50-150ms latency로 이를 audio sample로 변환한다.
4. Time-to-first-audio-byte: MIO paper에서는 약 300-500ms로, GPT-4o의 약 250ms에 접근한다.

Mini-Omni(arXiv:2408.16725), GLM-4-Voice(arXiv:2412.02612), Moshi(arXiv:2410.00037)는 보완적인 streaming speech-LLM 설계다. 특히 Moshi는 단일 GPU에서 160ms round-trip을 달성한다.

### 네 단계 curriculum

MIO의 학습 curriculum:

1. Stage 1 — alignment. 대규모 modality-pair corpora: text-image, text-speech, text-music. 각 pair는 자체 token vocabulary segment를 사용한다. 공유 vocabulary를 학습한다.
2. Stage 2 — interleaved. 여러 modality가 섞인 문서(이미지 + video가 있는 blog, transcript가 있는 podcast 등). cross-modality context를 학습한다.
3. Stage 3 — speech-enhanced. 텍스트 능력을 잃지 않으면서 음성 품질을 끌어올리기 위한 추가 audio data.
4. Stage 4 — SFT. modality 전반의 instruction tuning: VQA, captioning, narration, speech-to-speech dialogue.

특정 stage를 빼면 특정 능력이 약해진다. stage 2를 건너뛰면 cross-modality context를 잃고, stage 3을 건너뛰면 음성이 나빠진다.

### 시각 사고 사슬

MIO는 chain-of-visual-thought를 도입한다. 모델이 reasoning step으로 중간 image token을 내보내는 방식이다. "고양이가 나무를 오르고 있나?"라는 질문에서 모델은 다음을 수행한다.

1. 장면을 렌더링하는 `<image>` token을 내보낸다(input image 또는 sketch에서).
2. sketch를 분석하는 텍스트를 내보낸다.
3. 최종 답을 내보낸다.

렌더링된 중간 이미지는 scratchpad 역할을 한다. spatial reasoning task의 benchmark가 개선된다. 이 아이디어는 텍스트 reasoning의 chain-of-thought와 닮았다.

### Any-to-any 경쟁 모델

- AnyGPT(arXiv:2402.12226): 4개 modality(텍스트, 이미지, 음성, 음악), 유사한 설계.
- Unified-IO 2(arXiv:2312.17172): vision action output, depth, normal을 추가한다. task 다양성은 더 높고 scale은 더 작다.
- NExT-GPT(arXiv:2309.05519): LLM + modality별 diffusion decoder. 단일 모델 접근은 아니다.
- CoDi(arXiv:2305.11846): composable diffusion. 공유 latent를 통한 any-to-any.

MIO는 pure-token any-to-any에 가장 가깝다. AnyGPT는 그 개념적 선조다.

### 지연 시간 예산

대화형 제품에서는 모든 component의 latency가 중요하다.

- Mic to audio tokens: ~50ms.
- Prefill(audio tokens + history): 8B model에서 ~100ms.
- First output token: ~50ms.
- Parallel residual-VQ + speech decoder: ~100-150ms.

총 time-to-first-audio-byte는 최소 약 300ms다. GPT-4o는 약 250ms를 주장한다. Moshi는 160ms를 주장한다. MIO/AnyGPT는 공개 benchmark 기준 400-600ms 범위다.

### Any-to-any가 여전히 어려운 이유

2026년에도 오픈 any-to-any 모델은 두 축에서 closed 모델을 뒤따른다.

- 음성 품질. residual-VQ tokenizer는 lossy하다. 대화 음성은 ElevenLabs급 voice와 비교하면 로봇처럼 들린다.
- Cross-modality reasoning. 모델에게 "보이는 것에 대해 노래해 줘"라고 요청하면 순수 vision task보다 여전히 실패가 잦다.

이는 열린 연구 문제다. Qwen3-Omni(Lesson 12.20)는 2025년 기준 가장 발전한 오픈 시도다.

## 활용하기

`code/main.py`:

- 네 modality vocabulary allocation을 정의하고 출력한다.
- 멀티모달 input(text, image, audio-clip, music) 목록을 tokenizer router로 보낸다.
- text-to-speech 응답의 streaming decode를 latency counting과 함께 시뮬레이션한다.
- encoder, prefill, decoder latency가 주어졌을 때 예상 time-to-first-audio-byte를 계산한다.

## 산출물

이 lesson은 `outputs/skill-any-to-any-pipeline-auditor.md`를 만든다. 대화형 제품 spec(modalities in, modalities out, latency target)이 주어지면 MIO-family 설계 선택을 audit하고 latency budget을 계산한다.

## 연습 문제

1. 제품이 speech input을 받고 speech output을 반환한다. end-to-end latency budget target은 무엇인가? 시간을 쓰는 component를 나열하라.

2. SpeechTokenizer residual-VQ는 8개 codebook을 사용한다. residual level을 순차가 아니라 병렬로 decode해야 하는 이유와 이 방식이 가져오는 latency 절감을 제안하라.

3. vocabulary가 32k text + 4k image + 4k speech를 갖고 있다. 8k music과 약 10개의 separator를 추가하라. hidden dim 4096에서 embedding matrix parameter cost는 얼마인가?

4. Chain-of-visual-thought는 중간 이미지를 내보낸다. 어떤 종류의 질문이 이득을 보는가? 추가 token 때문에 어떤 종류는 손해를 보는가?

5. Moshi(arXiv:2410.00037)를 읽어라. "inner monologue" 기법을 설명하고 MIO의 chain-of-visual-thought와 비교하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Any-to-any | "Multimodal in/out" | 텍스트, 이미지, 음성, 음악을 어떤 방향으로든 받아들이고 내보내는 단일 모델 |
| Residual-VQ | "Speech tokenizer stack" | 각 layer가 정보를 추가하는 multi-codebook tokenization. base layer는 content이고 이후 layer는 prosody다 |
| SEED-Tokenizer | "Image codes" | MIO가 사용하는 4096-entry codebook 기반 discrete image tokenizer |
| Chain-of-visual-thought | "Visual scratchpad" | 최종 답 전에 reasoning step으로 중간 이미지를 생성하는 방식 |
| Time-to-first-audio-byte | "TTFAB" | 사용자 음성부터 첫 audio output까지의 latency. 대화처럼 느끼려면 <500ms |
| Four-stage curriculum | "Training recipe" | Alignment -> interleaved -> speech-enhanced -> SFT 순서의 학습 절차 |

## 더 읽을거리

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
