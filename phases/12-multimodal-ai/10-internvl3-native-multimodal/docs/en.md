# InternVL3: 네이티브 멀티모달 사전학습

> InternVL3 이전의 모든 open VLM은 같은 세 단계 recipe를 따랐다. trillions of text tokens로 학습된 text LLM을 가져오고, vision encoder를 붙인 뒤, 그 접합부를 fine-tune한다. 이것은 동작하지만 alignment debt가 있다. text LLM은 전체 pretraining budget을 순수 text에 썼고 visual token을 native하게 이해하지 못한다. post-hoc으로 vision을 더하면 LLM은 text를 잊지 않으면서 visual input을 text reasoning에 연결하는 법을 다시 배워야 한다. InternVL3(Zhu et al., 2025년 4월)는 post-hoc 접근을 거부한다. 하나의 pretraining run에서 step one부터 text와 multimodal을 interleave한다. 그 결과 open 78B params에서 MMMU-Pro 기준 Gemini 2.5 Pro와 맞먹는다. 이 lesson은 native pretraining이 필요한 이유와 그것을 선택할 때 무엇이 바뀌는지 읽는다.

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 minutes

## 학습 목표

- post-hoc VLM training이 alignment debt를 쌓는 이유를 세 가지 measurable symptom(catastrophic forgetting, answer drift, visual-text inconsistency)을 들어 설명한다.
- InternVL3의 native pretraining corpus mix와 text : interleaved : caption 비율이 중요한 이유를 설명한다.
- V2PE(variable visual position encoding)를 Qwen2-VL의 M-RoPE와 비교한다.
- Visual Resolution Router(ViR)와 Decoupled Vision-Language(DvD) deployment optimization을 설명한다.

## 문제

Post-hoc VLM training이 기본값이다. LLaVA, BLIP-2, Qwen-VL, Idefics는 모두 이미 pretraining된 LLM(Llama, Vicuna, Qwen, Mistral)을 가져와 vision을 더한다. training stage는 보통 다음과 같다.

1. Frozen LLM + frozen vision encoder + trainable projector를 caption pair로 학습해 embedding을 align한다.
2. LLM을 unfreeze하고 instruction data(LLaVA-Instruct, ShareGPT4V)로 학습한다.
3. optional task-specific fine-tune을 한다.

Alignment debt의 세 가지 symptom이 나타난다.

- Catastrophic forgetting. post-hoc VLM이 text-only skill을 잊는다. GSM8K score가 5-10점 떨어진다. Hellaswag score가 떨어진다. pure-text agent가 regress한다.
- Answer drift. 같은 visual question의 작은 phrasing 차이가 다른 answer를 만든다. vision encoder는 LLM 자신의 token보다 약한 binding으로 LLM에 연결된다.
- Visual-text inconsistency. VLM이 이미지를 올바르게 설명한 뒤, 자기 설명과 모순되는 질문 답변을 할 수 있다. visual token은 text처럼 LLM 내부 consistency check에 참여하지 못한다.

이 symptom들은 잘 문서화되어 있다. MM1.5 Section 4가 이를 정량화한다. LLaVA-OneVision의 ablation도 이를 암시한다. Native pretraining이 답이다.

## 개념

### 네이티브 멀티모달 사전학습

InternVL3는 step one부터 native multimodal인 corpus로 scratch training한다. mix는 다음과 같다.

- 40% text-only data(FineWeb, Proof-Pile-2 등)
- 35% interleaved image-text data(OBELICS, MMC4-style)
- 20% paired image-caption data
- 5% video-text data

Vision token, text token, cross-modal interaction은 모두 첫 gradient step부터 같은 loss에 참여한다. alignment pretraining도, projector freezing stage도, 회복해야 할 catastrophic forgetting도 없다.

Base model은 single stage로 학습된다. instruction tuning은 뒤따르지만, base model은 이미 visual token을 first-class citizen으로 이해한다.

### V2PE(가변 시각 위치 인코딩)

Qwen2-VL은 fixed axis allocation을 가진 M-RoPE를 사용한다. InternVL3는 V2PE를 도입한다. modality type(text, image, video)에 따라 position encoding이 달라지고 learnable scaling을 갖는다. 실제로는 다음과 같다.

- Text token은 1D position(text index)을 받는다.
- Image patch는 2D position(row, col)을 받는다.
- Video frame은 3D position(time, row, col)을 받는다.

세 modality는 같은 RoPE frequency base를 공유하지만, band별 hidden-dim allocation은 fixed split이 아니라 learned parameter다. pretraining 중 temporal frequency resolution과 spatial frequency resolution의 trade-off를 조정할 자유가 생긴다.

V2PE의 ablation claim은 같은 compute에서 M-RoPE 대비 video benchmark 1-2점 향상이다. 혁명은 아니지만 더 깔끔하다.

### 시각 해상도 라우터(ViR)

Deployment optimization이다. 모든 image가 full-resolution encoding을 필요로 하지는 않는다. detail이 낮고 object 하나만 있는 photo를 1280px native로 encode하면 token을 낭비한다. ViR은 encoding 전에 질문에 답하는 데 필요한 최소 resolution을 예측하는 작은 classifier다.

routing은 세 tier를 가진다. low-res(256 token), medium(576), high(2048+). production traffic의 60% query에서는 low 또는 medium이 충분하다. net effect는 같은 quality에서 2-3x throughput이다.

### 분리형 비전-언어 배포(DvD)

큰 VLM을 serve할 때 vision encoder는 image당 한 번 실행되지만 LLM은 output token마다 autoregressive하게 실행된다. 두 component는 bottleneck이 다르다(vision = conv + attention의 GPU memory bandwidth, LLM = KV cache). DvD는 둘을 separate GPU에 나누고 streaming으로 넘긴다.

8B + 400M encoder model에서 DvD는 co-located 대비 per-node throughput을 대략 두 배로 늘린다.

### 단일 단계와 다단계 품질

InternVL3의 핵심 benchmark claim은 78B params에서 Gemini 2.5 Pro의 MMMU-Pro와 맞먹는다는 것이다. 38B에서는 GPT-4o와 맞먹고, 8B에서는 open-8B leaderboard를 이끈다. 모두 single-stage pretrain + instruction-tune recipe다.

Alignment-debt hypothesis는 측정 가능하다. InternVL3-8B는 vision-benchmark gain 단위당 Qwen2.5-VL-7B보다 text-benchmark point(MMLU, GSM8K)를 덜 잃는다. training이 둘로 나뉘지 않고 하나였기 때문에 model은 더 generalist다.

### InternVL3.5와 InternVL-U

InternVL3.5(2025년 8월)는 recipe를 scale한다. 같은 native-pretrain approach, 더 많은 data, 더 많은 params다. MMMU improvement는 incremental하다.

InternVL-U(2026)는 unified generation을 더한다. 같은 backbone 위에 MMDiT head를 얹어 image output을 만든다. "U"는 "Understanding + generation"을 뜻하며, Transfusion-style unified model(Lesson 12.13)을 좇는다. 같은 native-pretrain backbone이 understanding head와 generation head를 모두 지원한다.

### Native pretraining의 trade-off

Native pretraining은 공짜가 아니다.

- Compute. 새 VLM을 scratch로 학습하는 비용은 text LLM을 학습하는 것과 같다. millions of GPU-hours다. post-hoc adaptation은 기존 LLM weight를 재사용해 비용 대부분을 절약한다.
- Data. scale 있는 interleaved image-text corpus는 드물다. OBELICS는 141M document이고 MMC4는 571M이다. text만으로는 15T token이 존재한다. multimodal pretraining data scarcity는 hard constraint다.
- Base-LLM reuse. Native pretraining은 나중에 새 LLM을 끼워 넣는 선택지를 포기한다. post-hoc은 adapter만 재학습해 Llama-3.1을 Llama-4로 바꿀 수 있다.

InternVL3의 bet는 alignment debt가 reuse loss보다 더 나쁘다는 것이다. benchmark는 이 주장을 뒷받침한다. production cost는 future lab이 싸게 복제하는 것을 막는다. post-hoc VLM은 대부분 project에서 여전히 더 싸기 때문에 계속 존재할 것이다.

## 활용하기

`code/main.py`는 training-corpus mixer와 ViR router simulator다. 이 코드는 다음을 수행한다.

- target corpus mix(%text, %interleaved, %caption, %video)를 받아 modality별 예상 step 수를 계산한다.
- query batch(distribution: 50% low-detail, 30% medium, 20% high-detail)에서 ViR routing을 simulate하고 average token count를 보고한다.
- encoder vs LLM FLOPs가 주어졌을 때 DvD throughput estimate를 보고한다.
- params, compute, data, 예상 alignment-debt symptom 측면에서 post-hoc vs native pretraining을 나란히 출력한다.

## 산출물

이 lesson은 `outputs/skill-native-vs-posthoc-auditor.md`를 만든다. proposed VLM training plan이 주어지면 native로 갈지 post-hoc으로 갈지 audit하고, alignment-debt risk를 flag하며, corpus mix를 권장한다. 새 open-VLM project의 규모를 잡고 training strategy를 선택해야 할 때 사용하라.

## 연습 문제

1. InternVL3-8B(native pretrain)와 LLaVA-OneVision-7B(post-hoc)의 compute delta를 추정하라. GPU-hour ratio는 대략 얼마인가? gap은 무엇으로 설명되는가?

2. InternVL3는 40% text / 35% interleaved / 20% caption / 5% video를 보고한다. target task가 video-heavy라면 새 ratio를 제안하고, base model이 여전히 상당한 text와 caption data를 필요로 하는 이유를 주장하라.

3. forgetting에 대한 MM1.5 Section 4를 읽어라. post-hoc training에서 가장 큰 regression을 보인 정확한 benchmark 이름을 말하라. regression cost는 얼마였는가?

4. ViR은 traffic의 60%를 low-resolution encoding으로 route한다. 어떤 query를 잘못 route하는가(high-res가 필요한데 low-res로 보내는 경우)? 세 가지 router-failure mode를 제안하라.

5. DvD는 vision과 LLM을 separate GPU로 나눈다. 어떤 traffic pattern에서 DvD가 throughput을 돕기는커녕 해치는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Native multimodal pretraining | "From scratch together" | text + image + video token이 나중에 붙는 것이 아니라 step 1부터 loss에 참여한다 |
| Alignment debt | "Post-hoc penalty" | frozen LLM에 vision을 붙이면서 생기는 text skill과 answer consistency의 측정 가능한 regression |
| V2PE | "Variable visual pos encoding" | modality별 learnable position encoding allocation. InternVL3의 M-RoPE successor |
| ViR | "Resolution router" | encoding 전에 query별로 필요한 최소 resolution을 골라 inference token을 절약하는 작은 classifier |
| DvD | "Decoupled deployment" | vision encoder는 한 GPU, LLM은 다른 GPU에 두고 stream handoff한다. 큰 VLM에서 throughput을 두 배로 늘린다 |
| InternVL-U | "Unified understanding + generation" | native-pretrain backbone에 image-generation head를 더한 2026 follow-up |
| Interleaved corpus | "OBELICS / MMC4" | text와 image가 자연스러운 reading order로 들어 있는 document. native pretraining의 원재료 |

## 더 읽을거리

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
