# 오픈 가중치 VLM 레시피: 실제로 중요한 것

> 2024-2026년 open-weight VLM literature는 ablation table의 숲이다. Apple의 MM1은 image encoder, connector, data mix의 13가지 조합을 실험했다. Allen AI의 Molmo는 자세한 human caption이 GPT-4V distillation보다 낫다는 것을 입증했다. Cambrian-1은 encoder 20개 이상을 비교했다. Idefics2는 다섯 축 design space를 정식화했다. Prismatic VLMs는 통제된 benchmark에서 27개 training recipe를 비교했다. 그 모든 소음 속에서도 여러 논문에 걸쳐 유지되는 작은 결과 집합이 있다. image encoder는 connector architecture보다 중요하고, data mixture는 둘 모두보다 중요하며, 자세한 human caption은 distilled synthetic data보다 낫다. 이 lesson은 당신이 직접 읽지 않아도 되도록 그 table들을 읽어 준다.

**Type:** Learn
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 minutes

## 학습 목표

- VLM design space의 다섯 축인 image encoder, connector, LLM, data mix, resolution schedule을 설명한다.
- MM1 / Idefics2 / Cambrian-1 ablation table을 읽고 어떤 knob이 특정 benchmark를 움직이는지 예측한다.
- compute budget과 task mix가 주어졌을 때 새 VLM의 recipe(encoder, connector, data, resolution)를 고른다.
- 같은 token count에서 자세한 human caption이 GPT-4V distillation을 이기는 이유를 설명한다.

## 문제

수백 개의 open-weight VLM이 존재한다. "좋은" model과 "state-of-the-art" model 사이의 차이 대부분은 architecture가 아니다. data, resolution schedule, encoder choice다. model이 기대보다 못할 때 어떤 knob을 먼저 돌릴지 아는 것은 5백만 GPU-hour짜리 실수를 막는다.

2023년 wave(LLaVA-1.5, InstructBLIP, MiniGPT-4)는 caption-pair pretraining + LLaVA-Instruct-150k 위에서 동작했다. 좋은 baseline이었다. MMMU 약 35% 부근에서 한계에 도달했다.

2024년 wave(MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs)는 exhaustive ablation을 수행했다. 결과는 놀랍고 실용적이었다.

## 개념

### 다섯 축 design space

Idefics2(Laurençon et al., 2024)는 축을 다음처럼 명명했다.

1. Image encoder. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. encoder는 patch size, resolution, pretraining objective가 다르다.
2. Connector. MLP(2-4 layer), Q-Former(32 query + cross-attn), Perceiver Resampler(64 query), C-Abstractor(convolutional + bilinear pooling).
3. Language model. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. LLM size는 지배적인 parameter cost다.
4. Training data. Caption pair(CC3M, LAION), interleaved(OBELICS, MMC4), instruction(LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Resolution schedule. Fixed 224/336/448, AnyRes, native dynamic. Training 중 ramp하거나 constant로 둔다.

모든 production VLM은 각 축에서 선택을 한다. MMMU score variance의 대부분은 축 1, 4, 5로 설명된다. 어떤 connector를 골랐는지는 주된 원인이 아니다.

### 축 1: encoder > connector

MM1 Section 3.2는 CLIP ViT-L/14에서 SigLIP SO400m/14로 바꾸면 MMMU가 3점 이상 오른다는 것을 보였다. connector를 MLP에서 Perceiver Resampler로 바꾸는 것은 1점 미만을 더했다. Idefics2도 반복 검증했다. 같은 token count에서 SigLIP > CLIP이고, Q-Former ≈ MLP ≈ Perceiver다.

Cambrian-1의 "Cambrian Vision Encoders Match-Up"(Tong et al., 2024)은 vision-centric benchmark(CV-Bench)에서 encoder 20개 이상을 실행했다. leaderboard 상위권은 DINOv2와 SigLIP의 혼합이다. CLIP은 중간권이고, ImageBind와 ViT-MAE는 낮다. CLIP ViT-L에서 DINOv2 ViT-g/14로 가는 gap은 CV-Bench에서 약 5-7점이다.

2026년 open VLM의 기본 encoder는 semantic + dense feature를 위한 SigLIP 2 SO400m/14이며, segmentation/grounding이 필요하면 DINOv2 ViT-g/14 feature와 concatenate하기도 한다(Cambrian의 "Spatial Vision Aggregator"가 이렇게 한다).

### 축 2: connector design은 대체로 무승부

MM1, Idefics2, Prismatic, MM-Interleaved는 같은 결론에 도달했다. visual-token count를 고정하면 connector architecture는 거의 중요하지 않다. mean-pooled patch 위의 2-layer MLP는 같은 token budget에서 32-query Q-Former와 1점 이내 성능을 낸다.

중요한 것은 token count다. 더 많은 visual token은 더 많은 LLM compute와 더 나은 성능을 뜻하지만 어느 지점 이후 수익이 줄어든다. 이미지당 64 token은 OCR에 너무 적다. 대부분 open VLM의 sweet spot은 576-1024 token이다. 2048+는 document와 chart에서만 도움이 된다.

Q-Former vs MLP는 quality 문제가 아니라 cost 문제다. Q-Former는 image resolution과 관계없이 token을 32-64개로 제한한다. MLP는 모든 patch token을 방출한다. high-res input에서는 Q-Former가 LLM context를 절약한다. low-res에서는 차이가 noise다.

### 축 3: LLM size가 ceiling을 정한다

LLM을 7B에서 13B로 두 배 키우면 모든 VLM 논문에서 MMMU가 안정적으로 2-4점 오른다. 70B에서는 대부분 benchmark가 포화된다. VLM의 multimodal reasoning ceiling은 LLM의 text reasoning ceiling이다. visual encoder는 feed할 수 있을 뿐, 대신 추론하지는 못한다.

이것이 Qwen2.5-VL-72B와 Claude Opus 4.7이 MMMU-Pro와 ScreenSpot-Pro를 압도하는 이유다. language brain이 거대하다. 7B VLM은 영리한 connector design으로 70B VLM을 대체할 수 없다.

### 축 4: data, 자세한 human caption이 distillation을 이긴다

Molmo + PixMo(Deitke et al., 2024)는 모두가 읽어야 할 2024년 결과다. Allen AI는 human annotator가 이미지를 1-3분짜리 dense speech-to-text pass로 묘사하게 하여, 712K장의 densely-captioned image를 만들었다. training data에는 GPT-4V distillation이 전혀 없었다.

Molmo-72B는 11개 benchmark 중 11개에서 Llama-3.2-90B-Vision을 이겼다. delta는 architecture가 아니다. caption quality다. 자세한 human caption은 짧은 web caption보다 이미지당 5-10배 더 많은 정보를 담고, GPT-4V distillation이 hallucinate하는 지점에서도 factually grounded 상태를 유지한다.

ShareGPT4V(Chen et al., 2023)와 Cauldron(Idefics2)도 human + GPT-4V caption을 섞어 같은 playbook을 따랐다. 추세는 명확하다. 2026 frontier에서는 caption density > caption quantity > distillation convenience다.

### 축 5: resolution과 schedule

Idefics2의 ablation에서는 384 -> 448이 1-2점을 더했다. image splitting(AnyRes)을 통해 448 -> 980으로 가면 OCR benchmark에서 3-5점을 더 얻었다. flat resolution training은 medium accuracy에서 plateau가 생긴다. resolution ramping(224에서 시작해 448 또는 native로 끝남)은 더 빠르게 학습하고 더 높은 곳에서 끝난다.

Cambrian-1은 resolution vs token trade-off를 실행했다. fixed compute에서 낮은 resolution에 더 많은 token을 쓸 수도 있고, 높은 resolution에 더 적은 token을 쓸 수도 있다. OCR에서는 higher resolution이 이긴다. general scene understanding에서는 lower-res-more-tokens가 이긴다.

2026년 production recipe는 Stage 1을 384 fixed로 학습하고, Stage 2를 OCR-heavy task에 대해 최대 1280까지 dynamic resolution으로 학습하는 것이다.

### Prismatic의 통제 비교

Prismatic VLMs(Karamcheti et al., 2024)는 모든 축을 통제한 논문이다. 같은 13B LLM, 같은 instruction data, 같은 evaluation에서 한 번에 하나의 축만 바꾼다. 결과:

- 이미지당 visual-token count가 variance의 약 60%를 설명한다.
- Encoder choice가 약 20%를 설명한다.
- Connector architecture가 약 5%를 설명한다.
- 나머지(data mix, scheduler, LR)가 약 15%다.

이것은 거친 분해지만, literature에서 "무엇을 먼저 ablate해야 하는가"에 대한 가장 깨끗한 답이다.

### 2026년용 picker

증거를 종합하면, 2026년 새 project를 위한 기본 open-VLM recipe는 다음과 같다.

- Encoder: native resolution의 SigLIP 2 SO400m/14 with NaFlex. segmentation/grounding이 필요하면 dense feature를 위해 DINOv2 ViT-g/14와 concatenate한다.
- Connector: patch token 위의 2-layer MLP. token-constrained 상황이 아니면 Q-Former를 건너뛴다.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2. cost에는 7B, quality에는 70B를 쓰며 target latency로 고른다.
- Data: PixMo + ShareGPT4V + Cauldron에 task-specific instruction data를 보강한다.
- Resolution: dynamic(long side당 min 256, max 1280 pixel).
- Schedule: Stage 1 alignment(projector-only), Stage 2 full fine-tune, Stage 3 task-specific fine-tune.

이 기본값 하나하나는 lesson 끝에 인용된 논문들의 measured ablation으로 거슬러 올라간다.

## 활용하기

`code/main.py`는 ablation table parser와 recipe picker다. MM1과 Idefics2 ablation table을 축약해 encode하고 다음 query를 할 수 있게 한다.

- "budget X와 task Y가 주어졌을 때 어떤 recipe가 이기는가?"
- "7B Llama에서 SigLIP을 CLIP으로 바꾸면 예상 MMMU delta는 얼마인가?"
- "80% confidence answer를 얻으려면 어떤 axis를 먼저 ablate해야 하는가?"

출력은 예상 benchmark delta와 "ablate first" recommendation을 포함한 ranked recipe list다.

## 산출물

이 lesson은 `outputs/skill-vlm-recipe-picker.md`를 만든다. target task mix, compute budget, latency target이 주어지면 각 선택을 정당화하는 ablation citation과 함께 full recipe(encoder, connector, LLM, data mix, resolution schedule)를 출력한다. 새 VLM project가 시작될 때마다 engineer들이 Idefics2 ablation table을 다시 발명하는 일을 막는다.

## 연습 문제

1. MM1 Section 3.2를 읽어라. 50M image budget의 fixed 2B LLM에서는 어떤 encoder가 이기는가? 13B LLM에서도 답이 뒤집히는가? 왜인가?

2. Cambrian-1은 DINOv2 + SigLIP concatenation이 vision-centric benchmark에서는 각각 단독보다 뛰어나지만 MMMU에서는 signal을 더하지 않는다고 찾았다. 어떤 benchmark가 이득을 보고 어떤 benchmark가 flat할지 예측하라.

3. target이 2B LLM 위의 mobile UI agent다. encoder, connector, resolution, data mix를 고르라. 각 선택을 specific ablation table로 정당화하라.

4. Molmo는 4B와 72B model을 제공한다. 4B는 closed 7B VLM과 경쟁력 있고, 72B는 11/11 benchmark에서 Llama-3.2-90B-Vision을 이긴다. 이것은 LLM-size plateau hypothesis에 대해 무엇을 말해 주는가?

5. 7B VLM에서 data-mix quality를 encoder quality와 분리하기 위한 ablation table을 설계하라. 최소 몇 번의 training run이 필요한가? 네 가지 axis setting을 제안하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Ablation | "Turning one knob" | 다른 모든 것을 고정하고 정확히 하나의 design-space axis만 다르게 한 여러 training run |
| Connector | "Bridge" / "projector" | vision encoder output을 LLM token space로 mapping하는 trainable module(MLP, Q-Former, Perceiver) |
| Detailed human caption | "Dense caption" | web alt text보다 풍부한 여러 문장짜리 human-written description(보통 80-300 token) |
| Distillation | "GPT-4V captions" | 더 강한 proprietary VLM이 생성한 training data. 편리하지만 inherited hallucination에 취약하다 |
| AnyRes / dynamic res | "High-res path" | tiling 또는 M-RoPE로 encoder native resolution보다 큰 이미지를 넣는 strategy |
| Resolution ramp | "Curriculum" | 낮은 resolution에서 시작해 증가시키는 training schedule로 alignment learning을 빠르게 한다 |
| Vision-centric bench | "CV-Bench / BLINK" | language-heavy reasoning보다 fine-grained visual perception을 압박하는 evaluation |
| PixMo | "Molmo's data" | Allen AI의 712K densely-captioned image dataset. human speech를 dense caption으로 전사했다 |

## 더 읽을거리

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
