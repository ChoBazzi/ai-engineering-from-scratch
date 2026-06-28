# LLaVA-OneVision: 하나의 모델로 단일 이미지, 다중 이미지, 비디오 처리

> LLaVA-OneVision(Li et al., 2024년 8월) 이전 open-VLM 세계에는 별도의 계보가 있었다. single image에는 LLaVA-1.5, multi-image에는 Mantis와 VILA 같은 model, video에는 Video-LLaVA와 Video-LLaMA가 있었다. 각각은 자기 benchmark에서 이기고 다른 영역에서는 실패했다. LLaVA-OneVision은 하나의 curriculum이 세 scenario 모두를 지배하는 단일 model을 학습시킬 수 있으며, emergent task-transfer effect(single-image skill이 video로 export되고, multi-image reasoning이 single-image로 export됨)가 specialist들의 합보다 낫다고 주장했다. recipe는 놀랄 만큼 단순하다. scenario 전반에서 일정하게 유지되는 visual-token budget과, single-image에서 OneVision(multi-image), video로 이동하는 explicit curriculum이다. 이 lesson은 budget, curriculum, emergent behavior를 읽는다.

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 minutes

## 학습 목표

- single-image, multi-image, video input 전반에서 일정하게 유지되는 visual-token budget을 설계한다.
- catastrophic forgetting 없이 single-image skill을 video로 transfer하는 training curriculum 순서를 정한다.
- curriculum이 제대로 구성되면 같은 parameter count에서 단일 model이 specialist를 이기는 이유를 설명한다.
- LLaVA-OneVision이 보고한 세 가지 emergent capability인 multi-camera reasoning, set-of-mark prompting, iPhone-screenshot agent를 말한다.

## 문제

Image, multi-image, video는 각각 model에 다른 부담을 준다.

Single-image는 OCR과 fine detail을 잡기 위해 high-resolution token(AnyRes, 약 2880 visual token)을 원한다. sample당 budget은 image 하나, 2880 token이다.

Multi-image는 image 간 reasoning이 context에 들어가도록 여러 image를 moderate resolution(각각 약 576 token)으로 원한다. sample당 budget은 4-8 image, 각 576 token, 총 2300-4600 token이다.

Video는 temporal dynamics를 잡기 위해 낮은 resolution의 frame을 많이 원한다(spatial pooling 이후 frame당 약 196 token). sample당 budget은 8-32 frame, 각 196 token, 총 1600-6200 token이다.

별도의 model을 학습하면 budget 하나를 고르면 된다. 하나의 model을 학습하려면 context를 터뜨리지 않으면서 scenario 전반에서 합리적으로 scale되는 budget이 필요하다.

OneVision 이전의 기본 답은 "하나의 scenario를 학습하고 나머지는 무시하라"였다. Video-LLaVA는 image model에 추가 training stage로 video를 retrofit했다. LLaVA-NeXT는 tiling으로 multi-image support를 더했다. 셋 모두를 깔끔하게 처리하지는 못했다.

## 개념

### OneVision 토큰 예산

LLaVA-OneVision은 sample당 대략 3000-4000 token의 unified visual-token budget을 고르고, scenario별로 다르게 할당한다.

- Single image: AnyRes-9(3x3 tile + thumbnail), 각 tile은 384에서 729 patch, aggressive bilinear pooling 2x2 -> tile당 182. 총 9 * 182 + 182 = 1820 token. 또는 AnyRes-4 at 729-per-tile = 2916 + 729.
- Multi-image: 각 이미지는 moderate resolution(384, tiling 없음), pooling 없는 729 token. budget 6 image -> 4374 token.
- Video: 384 resolution의 32 frame에 aggressive 3x3 bilinear pool -> frame당 81 token. 총 32 * 81 = 2592 token.

이 할당은 전체 token을 대략 일정하게 유지한다. LLM은 context를 터뜨리는 batch를 보지 않는다. encoder는 scenario별로 다른 geometry를 만들지만 LLM은 같은 budget을 소비한다.

### 세 단계 curriculum

LLaVA-OneVision은 세 단계로 학습한다.

1. Single-image SFT(stage SI). 모든 data는 single-image-plus-text다. high-resolution AnyRes input으로 학습한다. perception, OCR, fine-grained understanding을 가르친다. LLaVA-NeXT data와 OneVision-specific single-image data를 사용한다.
2. OneVision SFT(stage OV). single-image + multi-image + video(uniformly sampled frame)를 섞는다. unified token budget으로 학습한다. heterogeneous batch shape를 처리하도록 model을 가르친다. weight reset은 없고 stage SI에서 이어 간다.
3. Task transfer(stage TT). target task mix로 이어 학습한다. 보통 product에 따라 multi-image나 video 비중을 높인다. deployment용 optional fine-tune이다.

중요한 점은 curriculum order가 중요하다는 것이다. 같은 data를 써도 video-first나 multi-image-first는 single-image-first보다 image performance가 나쁘다. 논문은 이것을 명시적으로 ablate한다.

### Curriculum이 작동하는 이유

Single-image training은 perceptual base를 만든다. Patch token은 fine-grained visual feature를 담고, LLM은 그것을 text와 통합하는 법을 배운다. Multi-image와 video는 structural challenge(어떤 image가 무엇인지, 무엇이 먼저 일어났는지)를 도입하며, 강한 perceptual base 없이 배우기 어렵다.

모든 scenario를 처음부터 함께 학습하면 model은 perception을 underfit하고(batch당 single-image data가 제한됨), structure를 overfit한다(multi-image / video data가 많음). 결과는 cross-image reasoning pattern은 따르지만 시각적으로 얕은 model이다.

Curriculum ordering은 stage SI에서 perception strength를 얻고, stage OV에서 compositional/temporal reasoning을 얻으며, 둘 중 어느 것도 잃지 않게 한다.

### 시나리오 간 창발 능력

LLaVA-OneVision 논문은 세 가지 emergent capability를 보고한다.

1. Multi-camera reasoning. multi-image + video를 별도로 학습했지만, inference 때 multi-camera driving scene에 대해 reasoning하도록 요청한다. model은 training에서 정확히 그 format을 본 적이 없어도 view들을 올바르게 통합한다.
2. Set-of-mark prompting. 사용자가 이미지의 object에 번호 marker를 붙이면 model은 "mark 3이 mark 7에 비해 무엇을 하는가"를 reasoning한다. mark나 annotation으로 학습하지 않았고, spatial grounding + multi-image reference의 조합에서 배웠다.
3. iPhone-screenshot agent. 사용자가 iPhone screen screenshot을 제공하고 다음 click 계획을 묻는다. UI screenshot, user workflow video, multi-image before/after pair로 학습했다. agent use case로 일반화된다.

이것들은 trained task가 아니다. curriculum의 compositional structure에서 emergent하게 나타난다.

### 시각 토큰 풀링

Token budget에는 pooling이 필요하다. OneVision은 2D patch grid에 bilinear interpolation을 적용한다. 24x24 = 576 patch가 12x12 = 144(2x factor) 또는 8x8 = 64(3x factor)가 된다. locality를 보존하기 위해 pooling은 token space가 아니라 patch-grid space에서 수행된다.

scenario별 pooling factor 선택 자체가 hyperparameter다. pooling을 덜 하면 token이 많아지고 representation이 풍부해진다. pooling을 더 하면 token이 적어지고 더 많은 frame/image가 들어간다.

### LLaVA-OneVision-1.5

2025년 follow-up(LLaVA-OneVision-1.5, arXiv 2509.23661)은 training data, model weight, code가 "fully open"이다. 일부 benchmark에서 proprietary gap을 맞추고 recipe를 민주화했다. 같은 curriculum, 더 많은 data, 더 나은 base LLM이다. architecture change는 없다.

### Qwen2.5-VL과의 대비

Qwen2.5-VL(Lesson 12.09)은 다른 선택을 한다. fixed pooling 대신 M-RoPE와 dynamic FPS를 사용한다. budget은 input에 따라 scale된다. 1분 video는 5초 video보다 더 많은 token을 쓴다. LLaVA-OneVision은 budget을 고정하고 pooling을 scale한다. 둘 다 작동한다. configurability와 predictability를 맞바꾸는 선택이다.

## 활용하기

`code/main.py`는 OneVision-style VLM을 위한 curriculum and budget planner다. sample당 token budget과 target scenario mix(예: 40% single-image, 30% multi-image, 30% video)가 주어지면 다음을 수행한다.

- scenario별 resolution, pooling factor, frame 수를 할당한다.
- 모든 scenario가 shared budget 안에 들어가는지 확인한다.
- 예상 token count, LLM FLOPs, under-tokenized scenario를 보고한다.
- stage-by-stage training schedule을 출력한다.

OneVision fine-tune을 계획하거나 VLM deployment의 request당 cost를 sanity-check할 때 사용하라.

## 산출물

이 lesson은 `outputs/skill-onevision-budget-planner.md`를 만든다. target task distribution과 per-sample budget이 주어지면 AnyRes factor, per-frame pooling, video frame count, curriculum stage weight를 출력한다. unified-scenario VLM을 training 또는 fine-tuning할 때마다 이 skill을 사용하라.

## 연습 문제

1. product가 80% single-image, 10% multi-image(2-4 image), 10% video(8-16 frame)를 지원한다. token budget을 설계하라. heavy multi-image를 하지 않아 절약한 extra budget을 어디에 넣겠는가?

2. LLaVA-OneVision Section 4.3(emergent capabilities)을 읽어라. paper가 보고하지 않았지만 curriculum이 unlock할 가능성이 높은 네 번째 emergent skill을 제안하라.

3. curriculum order를 바꿔 multi-image를 먼저 학습하고, 그다음 single-image, video를 학습한다고 하자. 어떤 benchmark가 저하될지, 왜 그런지 예측하라.

4. paper는 sample당 8 frame만으로 video benchmark를 학습했다고 보고한다. 이것이 inference에서 30초 video로 일반화되는가? 무엇이 먼저 깨지는가? token budget인가 temporal reasoning인가?

5. 24x24 patch를 12x12로 bilinear pooling하는 것은 각 dimension에서 4x reduction이다. stdlib Python으로 pooling을 구현하고 각 2x2 block의 mean이 bilinear output과 일치하는지 검증하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| OneVision scenario | "Single-image, multi-image, or video" | unified VLM이 처리하는 세 input shape 중 하나. budget은 전체에서 일정하게 유지된다 |
| Token budget | "How many tokens per sample" | training / inference sample당 LLM이 보는 total visual token. 보통 3000-4000 |
| Curriculum | "Training order" | emergent transfer를 위해 선택한 stage ordering(single-image -> multi-image -> video) |
| Bilinear pooling | "Token shrink" | token 수를 줄이면서 locality를 보존하기 위해 patch grid(2D)에 bilinear interpolation을 적용하는 것 |
| Emergent skill | "Not trained, still works" | matching training data 없이 inference에서 나타나는 capability. curriculum composition 때문에 생긴다 |
| AnyRes-k | "k-tile setup" | fixed resolution의 k sub-tile과 thumbnail 하나. 일반적인 k는 {4, 9} |
| Task transfer | "Cross-scenario generalization" | shared backbone을 통해 single-image에서 배운 skill이 video에 적용되는 것, 또는 그 반대 |

## 더 읽을거리

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
