# 임의 해상도 비전: Patch-n'-Pack과 NaFlex

> 실제 이미지는 224x224 정사각형이 아니다. 영수증은 9:16이고, 차트는 16:9이며, 의료 스캔은 4096x4096일 수 있고, 모바일 스크린샷은 9:19.5다. 2024년 이전 VLM의 답, 즉 모든 것을 고정 정사각형으로 리사이즈하는 방식은 OCR, 문서 이해, 고해상도 장면 파싱을 가능하게 하는 신호를 버렸다. NaViT(Google, 2023)는 가변 해상도 패치를 block-diagonal masking으로 단일 transformer 배치에 packed할 수 있음을 보였다. Qwen2-VL의 M-RoPE(2024)는 absolute positional table을 완전히 없앴다. LLaVA-NeXT의 AnyRes는 고해상도 이미지를 base + sub-images로 타일링했다. SigLIP 2의 NaFlex 변형(2025)은 이제 모든 aspect ratio를 하나의 checkpoint로 처리하려는 open VLM의 기본 encoder다. 이 lesson은 patch-n'-pack을 처음부터 끝까지 구현한다.

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 minutes

## 학습 목표

- 가변 해상도 이미지 배치의 patch를 하나의 sequence로 pack하고 block-diagonal attention mask를 만든다.
- 주어진 task에 대해 AnyRes tiling(LLaVA-NeXT), NaFlex(SigLIP 2), M-RoPE(Qwen2-VL) 중 하나를 선택한다.
- OCR, chart, photography에서 resize 없이 token budget을 계산한다.
- square-resize의 세 가지 failure mode인 찌그러진 텍스트, 잘린 content, padding token 낭비를 설명한다.

## 문제

Transformer는 sequence를 기대한다. 배치는 같은 길이 sequence의 stack이다. 이미지가 224x224라면 매번 196개의 patch token을 얻고, padding도 필요 없으며, 일이 끝난다. 224로 학습하고 224로 추론하면 resolution을 다시 생각할 필요가 없다.

현실은 협조적이지 않다. 문서는 세로형(8.5x11 inch, 대략 2:3)이다. 차트 스크린샷은 가로형(16:9)이다. 영수증은 길고 좁다(1:3). 의료 영상은 2048x2048 이상으로 제공된다. 모바일 기기 스크린샷은 1170x2532(0.46:1)이다.

2024년 이전의 세 가지 선택지와 각각이 실패하는 이유:

1. 고정 정사각형(224x224 또는 336x336)으로 resize한다. 찌그러짐이 텍스트와 얼굴을 왜곡한다. downscale이 차트 label과 OCR content를 파괴한다. LLaVA-1.5까지의 표준 관행이었다.
2. 고정 aspect ratio로 crop한다. 이미지 대부분을 버리게 되고, crop 위치를 고르는 것 자체가 별도의 vision 문제다.
3. 긴 변에 맞춰 padding한다. 왜곡은 해결하지만 세로형 이미지에서는 token의 50% 이상을 padding에 낭비한다. 그 모든 pad token에 대해 quadratic attention 비용을 낸다.

2024-2025년의 답은 transformer가 이미지의 native resolution patch를 먹게 하고, 낭비 compute 없이 heterogeneous batch를 하나의 sequence에 pack하는 방법을 찾는 것이다.

## 개념

### NaViT와 patch-n'-pack

NaViT(Dehghani et al., 2023)는 이것이 scale에서 동작함을 보인 논문이다. 아이디어는 기계적이다.

1. 배치의 각 이미지에 대해 선택한 patch size(예: 14)에서 native patch grid를 계산한다.
2. 각 이미지의 patch를 자기만의 variable-length sequence로 flatten한다.
3. 모든 이미지의 patch를 배치용 하나의 긴 sequence로 concatenate한다.
4. 이미지 A의 patch가 이미지 A 안에서만 attend하도록 block-diagonal attention mask를 만든다.
5. patch별 position information(2D RoPE 또는 fractional position embedding)을 유지한다.

336x336(576 token), 224x224(256 token), 448x336(768 token) 이미지 세 장의 배치는 1600x1600 block-diagonal mask를 갖는 하나의 1600-token sequence가 된다. padding이 없다. 낭비 compute가 없다. transformer가 임의 aspect ratio를 처리한다.

NaViT는 training 중 fractional patch dropping도 도입했다. 배치 전체에서 patch의 50%를 무작위로 drop하는 방식이며, regularization과 training speedup을 동시에 제공한다. SigLIP 2가 이것을 이어받았다.

### AnyRes(LLaVA-NeXT)

LLaVA-NeXT의 AnyRes는 실용적인 대안이다. 고해상도 이미지와 고정 encoder(CLIP 또는 SigLIP at 336)가 주어지면 이미지를 tile한다.

1. 미리 정의된 set, 예를 들어 (1x1), (1x2), (2x1), (1x3), (3x1), (2x2) 등에서 이미지 aspect ratio에 가장 잘 맞는 grid layout을 고른다.
2. 전체 이미지를 grid로 tile한다. 각 tile은 336x336 crop이 된다.
3. thumbnail도 만든다. 전체 이미지를 336x336으로 resize한 global-context token이다.
4. 모든 tile을 frozen 336-encoder에 통과시킨다. tile token + thumbnail token을 concatenate한다.

2x2 grid와 thumbnail을 쓰는 672x672 이미지는 4 * 576 + 576 = 2880 visual token이 된다. 비싸지만 효과적이다. LLM은 local detail과 global context를 모두 본다.

AnyRes는 encoder가 frozen이고 한 resolution만 지원할 때 선택하는 경로다. 큰 이미지에서는 token count가 폭발한다. 4x4 grid의 1344x1344 이미지는 9216 + 576, 즉 약 9800 token이 되어 8k LLM context 대부분을 채운다.

### M-RoPE(Qwen2-VL)

Qwen2-VL은 Multimodal Rotary Position Embedding을 도입했다. NaViT의 fractional position이나 AnyRes의 tile-and-thumbnail 대신, 각 patch가 3D position(temporal, height, width)을 가진다. query/key rotation이 임의 H, W, temporal length를 처리한다.

M-RoPE는 retraining 없이 native dynamic resolution을 제공한다. 추론 때 임의 HxW 이미지를 넣으면 patch embedder가 H/14 x W/14 token을 만들고, 각 token은 (t=0, r=row, c=col) position을 받으며, RoPE가 올바른 frequency로 attention을 rotate한다. 끝이다. Qwen2.5-VL과 Qwen3-VL도 이를 이어간다. InternVL3의 V2PE는 modality별 variable encoding을 더한 같은 아이디어다.

AnyRes와 달리 M-RoPE는 native resolution에서 O(H x W / P^2) token이다. tile overhead가 곱해지지 않는다. NaViT와 달리 여전히 forward 하나에 image 하나를 기대한다. resolution이 다른 이미지를 batching하려면 그 위에 patch-n'-pack이 여전히 필요하다.

### NaFlex(SigLIP 2)

NaFlex는 SigLIP 2 checkpoint의 native-flex mode다. 하나의 model이 추론 시 여러 sequence length(256, 729, 1024 token)를 처리한다. 내부적으로 training 중 NaViT-style patch-n'-pack과 patch별 absolute fractional position을 사용한다. 핵심 장점은 하나의 checkpoint로 추론 시 task에 따라 token budget을 고를 수 있다는 점이다.

Semantic task(classification, retrieval)에는 256 token을 쓴다. OCR이나 chart understanding에는 1024 token을 쓴다. retraining은 필요 없다.

### 패킹 마스크

Block-diagonal mask는 대부분의 구현이 넘어지는 지점이다. 길이 `N_total`의 packed sequence가 길이 `n_i`인 이미지 `i=0..B-1`를 담고 있을 때, shape `(N_total, N_total)`인 mask `M`은 두 index가 같은 image block에 있으면 1이고, 아니면 0이다. cumulative length list로 만들 수 있다.

```text
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

PyTorch에서는 `torch.block_diag`나 explicit gather로 한 줄에 만들 수 있다. FlashAttention의 variable-length path(`cu_seqlens`)는 mask를 완전히 건너뛰고 cumulative-length tensor를 직접 사용해 sequence 내부에서만 attend한다. 일반적인 batch에서는 dense mask보다 약 10x 빠르다.

### 토큰 예산

Task별로 strategy를 고른다.

- OCR / documents: 1024-4096 token. SigLIP 2 NaFlex at 1024 또는 AnyRes 3x3 + thumbnail.
- Charts and UI: native 384-448에서 729-1024 token. max pixels cap이 있는 Qwen2.5-VL dynamic resolution.
- Natural photos: 256-576 token이면 충분하다. downstream LLM은 충분히 본다. content density가 높은 곳에 token 비용을 지불한다.
- Video: spatial pooling 이후 frame당 64-128 token, 2-8 FPS. Lesson 12.17에서 다룬다.

2026년 production rule은 per-task max-pixels cap을 고르고, 그 cap까지 native aspect ratio로 encode하며, batch를 pack하고 padding을 생략하는 것이다. Qwen2.5-VL은 바로 이 knob을 위해 `min_pixels`와 `max_pixels`를 노출한다.

## 활용하기

`code/main.py`는 integer pixel coordinate를 가진 heterogeneous image batch에 대해 patch-n'-pack을 구현한다. 이 코드는 다음을 수행한다.

- (H, W) image size list를 받는다.
- patch size 14에서 각 이미지의 patch sequence length를 계산한다.
- 이를 전체 길이 `sum(n_i)`인 하나의 sequence로 pack한다.
- block-diagonal attention mask를 만든다(dense, 명확성을 위해).
- packed cost를 square-resize 및 AnyRes tiling과 비교한다.
- mixed batch(receipt, chart, screenshot, photo)에 대한 token budget table을 출력한다.

실행해 보라. 출력되는 숫자들이 2026년 모든 open VLM이 patch-n'-pack을 쓰는 이유다.

## 산출물

이 lesson은 `outputs/skill-resolution-budget-planner.md`를 만든다. mixed-aspect-ratio workload(OCR, chart, photo, video frame)와 total-token budget이 주어지면 올바른 strategy(NaFlex, AnyRes, M-RoPE, fixed-square)를 고르고 per-request configuration을 출력한다. product용 VLM 크기를 산정할 때 이 skill을 사용하라. latency budget을 죽이는 조용한 10x token 폭증을 막아 준다.

## 연습 문제

1. 영수증이 600x1500(1:2.5)이다. patch size 14에서 native-resolution token은 몇 개인가? 336으로 square-resize한 뒤에는 몇 개인가? 실제로 어느 쪽이 OCR accuracy를 더 잃는가?

2. 길이 256, 576, 729, 1024인 이미지 네 장의 batch에 대한 block-diagonal mask를 만들어라. attention matrix가 2585x2585이고 정확히 `256^2 + 576^2 + 729^2 + 1024^2`개의 non-zero entry를 갖는지 검증하라.

3. patch 14에서 1792x896 이미지에 대해 (a) 336으로 square-resize 후 encode, (b) AnyRes 2x1 + thumbnail, (c) native M-RoPE를 비교하라. 어느 방식이 token을 가장 적게 쓰는가? 어느 방식이 detail을 가장 많이 보존하는가?

4. Fractional patch dropping을 구현하라. packed sequence가 주어졌을 때 token의 50%를 균일 무작위로 drop하고, block-diagonal mask를 그에 맞게 갱신하라. mask sparsity 변화를 측정하라.

5. Qwen2-VL 논문(arXiv:2409.12191)의 Section 3.2를 읽어라. `min_pixels`와 `max_pixels`가 무엇을 제어하는지, 그리고 두 bound가 모두 중요한 이유를 두 문장으로 설명하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | 서로 다른 이미지의 variable-length patch sequence를 하나의 batch dimension으로 concatenate하는 것 |
| Block-diagonal mask | "Packing mask" | pack 안의 이웃이 아니라 각 이미지의 patch가 자기 자신에게만 attend하도록 제한하는 attention mask |
| AnyRes | "LLaVA-NeXT tiling" | 고해상도 이미지를 fixed-size tile grid와 global thumbnail로 나누고, 모든 tile을 fixed encoder로 encode하는 방식 |
| NaFlex | "SigLIP 2 native-flex" | retraining 없이 추론 시 256/729/1024-token budget을 처리하는 단일 SigLIP 2 checkpoint |
| M-RoPE | "Multimodal RoPE" | position table 없이 임의 H, W, T를 처리하는 3D rotary position encoding(time, row, column) |
| cu_seqlens | "FlashAttention packing" | dense block-diagonal mask 대신 FlashAttention varlen path가 사용하는 cumulative-length tensor |
| min_pixels / max_pixels | "Resolution bounds" | 매우 작거나 큰 input에서 token count를 제한하는 Qwen2.5-VL per-request knob |
| Visual token budget | "How many tokens per image" | 이미지당 방출되는 patch token의 대략적 개수이며 LLM prompt budget과 attention cost를 결정한다 |

## 더 읽을거리

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
