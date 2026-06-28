# Vision Transformer와 패치 토큰 기본형

> 멀티모달을 다루기 전에, 이미지는 transformer가 먹을 수 있는 token sequence가 되어야 합니다. 2020년 ViT paper는 16x16 pixel patch, linear projection, position embedding으로 이 문제에 답했습니다. 5년 뒤인 2026년의 모든 frontier model(2576px native의 Claude Opus 4.7, Gemini 3.1 Pro, Qwen3.5-Omni)도 여전히 이 방식으로 시작합니다. Encoder는 ViT에서 DINOv2, SigLIP 2로 바뀌었고, register token이 추가되었으며, positional scheme은 2D-RoPE가 되었지만 primitive는 유지되었습니다. 이 lesson은 patch-token pipeline을 처음부터 끝까지 읽고 stdlib Python으로 구현해, Phase 12의 나머지 과정에서 "visual token"에 대한 구체적인 mental model을 갖게 합니다.

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 minutes

## 학습 목표

- HxWx3 이미지를 올바른 positional encoding을 가진 patch token sequence로 변환합니다.
- 주어진 (patch size, resolution, hidden dim, depth)의 ViT에 대해 sequence length, parameter count, FLOPs를 계산합니다.
- ViT를 2020년 연구에서 2026년 production으로 끌어올린 세 가지 upgrade를 설명합니다: self-supervised pretraining(DINO / MAE), register token, native-resolution packing.
- Downstream task에 맞춰 CLS pooling, mean pooling, register token 중 무엇을 쓸지 선택합니다.

## 문제

Transformer는 vector sequence에서 동작합니다. Text는 이미 sequence입니다(byte 또는 token). 이미지는 세 개의 color channel을 가진 2D pixel grid이며 sequence가 아닙니다. 모든 pixel을 펼치면 224x224 RGB 이미지는 150,528개의 token이 되고, 이 길이에서 self-attention은 시작부터 불가능합니다(sequence length에 대해 quadratic).

2020년 이전 접근은 앞단에 CNN feature extractor를 붙였습니다. ResNet이 2048-dim vector의 7x7 feature map을 만들고, 이 49개 token을 transformer에 넣는 식입니다. 이 방식은 작동하지만 CNN의 bias(translation equivariance, local receptive field)를 물려받고, scale을 먹고 자라는 transformer의 장점을 잃습니다.

Dosovitskiy et al. (2020)은 직설적으로 물었습니다. CNN을 건너뛰면 어떨까? 이미지를 고정 크기 patch(예: 16x16 pixel)로 나누고, 각 patch를 vector로 linearly project하고, positional embedding을 더한 뒤, vanilla transformer에 sequence로 넣는 것입니다. 당시에는 convolution 없는 vision이라는 이단적 발상이었습니다. 하지만 충분한 data(JFT-300M, 이후 LAION)가 있으면 ImageNet에서 ResNet을 이겼고 계속 개선되었습니다.

2026년에 ViT primitive는 의심할 여지가 없는 foundation입니다. 모든 open-weights VLM의 vision tower는 어떤 후손(DINOv2, SigLIP 2, CLIP, EVA, InternViT)입니다. 질문은 더 이상 "patch를 쓸 것인가?"가 아니라 "어떤 patch size, 어떤 resolution schedule, 어떤 pretraining objective, 어떤 positional encoding을 쓸 것인가?"입니다.

## 개념

### Token으로서의 patch

Shape가 `(H, W, 3)`인 이미지 `x`와 patch size `P`가 있을 때, 이미지를 겹치지 않는 `(H/P) x (W/P)` grid patch로 자릅니다. 각 patch는 pixel의 `P x P x 3` cube입니다. 각 cube를 `3 P^2` vector로 flatten합니다. Shape가 `(3 P^2, D)`인 shared linear projection `W_E`를 적용해 각 patch를 model hidden dimension `D`로 mapping합니다.

ViT-B/16 canonical config의 경우:
- Resolution 224, patch size 16 -> grid 14x14 -> 196 patch token.
- 각 patch는 `16 x 16 x 3 = 768` pixel value이며, `D = 768`로 project됩니다.
- Learnable `[CLS]` token을 더하면 sequence length는 197입니다.

Patch projection은 kernel size `P`, stride `P`, output channel `D`인 2D convolution과 수학적으로 동일합니다. Production code는 실제로 이렇게 구현합니다: `nn.Conv2d(3, D, kernel_size=P, stride=P)`. "Linear projection"이라는 설명은 개념적 framing이고, kernel framing은 효율적인 구현입니다.

### 위치 임베딩

Patch에는 내재된 순서가 없습니다. Transformer는 patch를 bag처럼 봅니다. 초기 ViT는 learnable 1D positional embedding(위치마다 하나의 768-dim vector, 총 197개)을 더했습니다. 작동은 하지만 model을 training resolution에 묶어 둡니다. Inference에서 grid를 바꾸면 position table을 interpolate해야 합니다.

최신 vision backbone은 2D-RoPE(Qwen2-VL의 M-RoPE, SigLIP 2의 기본값) 또는 factorized 2D position을 사용합니다. 2D-RoPE는 patch의 (row, column) index에 따라 query와 key vector를 회전하므로, model은 rotation angle에서 relative 2D position을 추론합니다. Position table이 없습니다. Inference에서 arbitrary grid size를 처리합니다.

### CLS 토큰, pooled output, register token

Image-level representation은 무엇일까요? 세 가지 선택지가 공존합니다.

1. `[CLS]` token. Learnable vector를 patch sequence 앞에 붙입니다. 모든 transformer block 뒤 CLS token의 hidden state가 image representation입니다. BERT에서 물려받았습니다. Original ViT, CLIP이 사용합니다.
2. Mean pool. Patch token의 output hidden state를 평균냅니다. SigLIP, DINOv2, 대부분의 최신 VLM이 사용합니다.
3. Register token. Darcet et al. (2023)은 명시적 sink token 없이 학습한 ViT가 self-attention을 장악하는 high-norm "artifact" patch를 만든다는 사실을 관찰했습니다. 4-16개의 learnable register token을 추가하면 이 load를 흡수하고 dense-prediction quality(segmentation, depth)를 개선합니다. DINOv2와 SigLIP 2는 모두 register를 포함해 배포됩니다.

선택은 downstream task에서 중요합니다. CLS는 classification에 충분합니다. Patch token을 LLM에 넣는 VLM에서는 pooling을 완전히 건너뜁니다. 모든 patch가 LLM input token이 됩니다. Register는 handoff 전에 버립니다. 이는 content가 아니라 scaffolding입니다.

### 사전학습: 지도, 대조, 마스크, self-distilled

2020년 ViT는 JFT-300M의 supervised classification으로 pretrained되었습니다. 곧 다음 방식들이 대체했습니다.

- CLIP (2021): 400M pair에 대한 contrastive image-text. Lesson 12.02.
- MAE (2021, He et al.): patch의 75%를 mask하고 pixel을 reconstruct합니다. Pure image에서 작동하는 self-supervised 방식입니다.
- DINO (2021) / DINOv2 (2023): label도 caption도 없는 student-teacher self-distillation. 2023년 DINOv2 ViT-g/14는 가장 강한 purely-visual backbone이며 "dense features" use case의 기본값입니다.
- SigLIP / SigLIP 2 (2023, 2025): sigmoid loss와 native aspect ratio를 위한 NaFlex를 적용한 CLIP. 2026년 open VLM(Qwen, Idefics2, LLaVA-OneVision)의 지배적인 vision tower입니다.

Pretraining 선택은 backbone이 잘하는 일을 결정합니다. CLIP/SigLIP은 text와의 semantic matching, DINOv2는 dense visual feature, MAE는 downstream finetuning의 시작점에 적합합니다.

### 스케일링 법칙

ViT scaling(Zhai et al. 2022)은 ViT quality가 model size, data size, compute에서 예측 가능한 law를 따른다는 점을 확립했습니다. Fixed compute에서:
- 더 큰 model + 더 많은 data -> 더 좋은 quality.
- Patch size는 sequence length와 fidelity 사이의 lever입니다. Patch 14(DINOv2/SigLIP SO400m에서 일반적)는 patch 16보다 이미지당 token이 많습니다. OCR과 dense task에 좋지만 속도에는 나쁩니다.
- Resolution은 또 다른 큰 lever입니다. 224에서 384, 512로 올리면 거의 항상 좋아지며 FLOPs는 quadratic cost로 증가합니다.

ViT-g/14(1B params, patch 14, resolution 224 -> 256 tokens)와 SigLIP SO400m/14(400M params, patch 14)는 2026년 open VLM의 두 workhorse encoder입니다.

### ViT의 parameter count

전체 계산은 `code/main.py`에 있습니다. 224에서 ViT-B/16의 경우:

```text
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

Checkpoint를 load하기 전에 모든 ViT를 이런 식으로 대략 계산하세요. Backbone size는 모든 downstream VLM에서 VRAM floor를 결정합니다.

### 2026년 production config

2026년에 대부분의 open VLM이 탑재하는 encoder는 native resolution(NaFlex)의 SigLIP 2 SO400m/14입니다. 특징은 다음과 같습니다.
- 400M parameters.
- Patch size 14, default resolution 384 -> image당 729 patch token.
- Image-level task에는 mean pool, VQA에서는 729개 patch가 모두 LLM으로 흐릅니다.
- 4 register token, LLM handoff 전에 폐기.
- Native aspect ratio를 위한 image-level scaling이 적용된 2D-RoPE.

이 config의 모든 결정은 읽을 수 있는 paper로 거슬러 올라갑니다.

```figure
image-patch-tokens
```

## 사용하기

`code/main.py`는 patch tokenizer이자 geometry calculator입니다. (image H, W, patch P, hidden D, depth L)을 받아 다음을 보고합니다.

- Patching 이후 grid shape와 sequence length.
- Synthetic 8x8 pixel toy image의 token sequence(flatten + project 경로를 따라갑니다).
- Patch embed, position embed, transformer block, head로 나눈 parameter count.
- Target resolution에서 forward pass당 FLOPs.
- ViT-B/16 @ 224, ViT-L/14 @ 336, DINOv2 ViT-g/14 @ 224, SigLIP SO400m/14 @ 384의 comparison table.

실행해 보세요. Parameter count를 published number와 맞춰 보세요. Patch size와 resolution을 바꿔 보며 token-count cost를 체감하세요.

## 결과물

이 lesson은 `outputs/skill-patch-geometry-reader.md`를 생성합니다. ViT config(patch size, resolution, hidden dim, depth)가 주어지면, 근거와 함께 token-count, parameter-count, VRAM estimate를 생성합니다. VLM의 vision backbone을 고를 때마다 이 skill을 사용하세요. "token이 폭발해서 LLM context가 가득 찼다"는 놀람을 막아 줍니다.

## 연습문제

1. Native 1280x720 input과 patch size 14에서 Qwen2.5-VL의 patch-token sequence length를 계산하세요. CLS-only representation과 어떻게 비교되나요?

2. 1080p frame(1920x1080)은 patch 14에서 몇 개의 token을 만들까요? 5분 video를 30 FPS로 처리하면 총 visual token은 몇 개인가요? Pooling, frame sampling, token merging 중 어떤 비용 절감이 가장 큰가요?

3. Pure Python으로 patch token에 대한 mean pooling을 구현하세요. DINOv2 output의 196개 token에 대한 mean-pool이 pooled embedding을 요청했을 때 model의 `forward` 반환값과 일치하는지 확인하세요.

4. "Vision Transformers Need Registers"(arXiv:2309.16588)의 Section 3을 읽으세요. Register가 어떤 artifact를 흡수하며 downstream dense prediction에서 왜 중요한지 두 문장으로 설명하세요.

5. `code/main.py`를 수정해 patch-n'-pack을 지원하세요. 서로 다른 resolution의 image list가 주어지면 하나의 packed sequence와 block-diagonal attention mask를 생성하세요. Lesson 12.06에 도달하면 그 내용과 대조해 검증하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | 입력 이미지의 고정 크기 non-overlapping 영역이며 하나의 token이 됩니다 |
| Patch embedding | "Linear projection" | Flatten된 patch pixel을 D-dim vector로 mapping하는 shared learned matrix(또는 stride=P인 Conv2d) |
| CLS token | "Class token" | 전체 이미지를 대표하는 final hidden state를 갖는 prepended learnable vector이며, 2026년에는 선택 사항입니다 |
| Register token | "Sink token" | Pretraining 중 ViT가 만드는 high-norm attention artifact를 흡수하는 extra learnable token |
| Position embedding | "Positional info" | Sequence가 order-aware해지도록 만드는 per-position vector 또는 rotation이며, 최신 기본값은 2D-RoPE입니다 |
| Grid | "Patch grid" | 주어진 resolution과 patch size에서 patch의 (H/P) x (W/P) 2D array |
| NaFlex | "Native flexible resolution" | Retraining 없이 여러 aspect ratio와 resolution을 하나의 model이 처리하는 SigLIP 2 기능 |
| Backbone | "Vision tower" | Patch-token output을 VLM의 LLM에 공급하는 pretrained image encoder |
| Pooling | "Image-level summary" | Patch token을 하나의 vector로 바꾸는 전략: CLS, mean, attention pool, register-based |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14는 이미지당 더 많은 token을 만들고 OCR fidelity가 좋지만 느립니다. Patch 16은 고전적 기본값입니다 |

## 더 읽을거리

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — original ViT.
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE, self-supervised pretraining.
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — scale에서의 self-distillation, label 없음.
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token과 artifact analysis.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026년 기본 vision tower.
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — empirical scaling laws.
