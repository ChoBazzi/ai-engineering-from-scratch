# Vision Transformer (ViT)

> 이미지는 패치의 격자입니다. 문장은 토큰의 격자입니다. 같은 트랜스포머가 둘 다 처리합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 minutes

## 문제

2020년 전까지 컴퓨터 비전은 곧 컨볼루션을 뜻했습니다. ImageNet, COCO, 탐지 벤치마크의 모든 SOTA는 CNN 백본을 사용했습니다. 트랜스포머는 언어용이었습니다.

Dosovitskiy et al. (2020)의 "An Image is Worth 16x16 Words"는 컨볼루션을 완전히 제거할 수 있음을 보였습니다. 이미지를 고정 크기 패치로 자르고, 각 패치를 임베딩으로 선형 투영한 뒤, 그 시퀀스를 바닐라 트랜스포머 인코더에 넣습니다. 충분한 규모(ImageNet-21k 사전학습 또는 그 이상)에서는 ViT가 ResNet 기반 모델과 맞먹거나 더 뛰어납니다.

ViT는 2026년 더 넓은 패턴의 출발점이었습니다. 하나의 아키텍처, 여러 모달리티입니다. Whisper는 오디오를 토큰화합니다. ViT는 이미지를 토큰화합니다. 로보틱스에는 액션 토큰이 있습니다. 비디오에는 픽셀 토큰이 있습니다. 트랜스포머는 신경 쓰지 않습니다. 시퀀스를 넣으면 학습합니다.

2026년 기준 ViT와 그 후속 모델(DeiT, Swin, DINOv2, ViT-22B, SAM 3)은 비전의 대부분을 차지합니다. CNN은 여전히 엣지 기기와 지연 시간에 민감한 작업에서 이깁니다. 그 외에는 거의 모든 스택 어딘가에 ViT가 있습니다.

## 개념

![Image → patches → tokens → transformer](../assets/vit.svg)

### 1단계 — patchify

`H × W × C` 이미지를 평평한 패치의 `N × (P·P·C)` 시퀀스로 나눕니다. 일반적인 설정은 `224 × 224` 이미지, `16 × 16` 패치이며, 그 결과 768개 값으로 이루어진 패치 196개가 나옵니다.

```text
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

패치 크기는 조절 손잡이입니다. 작은 패치는 토큰이 더 많고 해상도가 좋지만 어텐션 비용이 제곱으로 증가합니다. 큰 패치는 더 거칠지만 더 저렴합니다.

### 2단계 — linear embedding

하나의 학습된 행렬이 각 평평한 패치를 `d_model`로 투영합니다. 이는 커널 크기 `P`, 스트라이드 `P`인 컨볼루션과 같습니다. PyTorch에서는 말 그대로 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`이며, 두 줄로 구현됩니다.

### 3단계 — `[CLS]` token을 앞에 붙이고 positional embeddings 더하기

- 학습 가능한 `[CLS]` 토큰을 앞에 붙입니다. 그 최종 은닉 상태가 분류에 쓰이는 이미지 표현입니다.
- 학습 가능한 위치 임베딩(ViT 원본) 또는 사인파 2D 위치 임베딩(이후 변형)을 더합니다.
- 2024년 이후에는 RoPE가 2D 위치로 확장되었고, 때로는 명시적 임베딩 없이 사용됩니다.

### 4단계 — 표준 transformer encoder

`LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 블록을 L개 쌓습니다. BERT와 동일합니다. 비전 전용 레이어가 없습니다. 이것이 논문의 교육적 핵심입니다.

### 5단계 — head

분류에서는 `[CLS]` 은닉 상태 → 선형층 → softmax를 사용합니다. DINOv2나 SAM에서는 `[CLS]`를 버리고 패치 임베딩을 직접 사용합니다.

### 중요했던 변형

| 모델 | 연도 | 변화 |
|-------|------|--------|
| ViT | 2020 | 원본입니다. 고정 패치 크기와 전체 전역 어텐션을 사용합니다. |
| DeiT | 2021 | 증류를 사용해 ImageNet-1k만으로도 학습할 수 있습니다. |
| Swin | 2021 | shifted window를 쓰는 계층형 구조입니다. 준제곱 비용을 고정했습니다. |
| DINOv2 | 2023 | 자기지도 방식입니다(레이블 없음). 가장 좋은 범용 비전 특징입니다. |
| ViT-22B | 2023 | 22B 파라미터입니다. 스케일링 법칙이 적용됩니다. |
| SigLIP | 2023 | ViT + 언어 쌍, 시그모이드 대조 손실입니다. |
| SAM 3 | 2025 | 무엇이든 분할합니다. ViT-Large + 프롬프트 가능한 마스크 디코더입니다. |

### 시간이 걸린 이유

ViT에는 CNN의 귀납적 편향(이동 불변성, 지역성)이 없기 때문에 CNN에 맞먹으려면 *아주 많은* 데이터가 필요합니다. 1억 장이 넘는 레이블 이미지나 강력한 자기지도 사전학습이 없으면, 동일한 compute에서는 CNN이 여전히 이깁니다. 2021년 DeiT가 증류 기법으로 이를 완화했고, 2023년 DINOv2가 자기지도 학습으로 이 문제를 사실상 해결했습니다.

## 직접 만들기

`code/main.py`를 보세요. 순수 stdlib로 patchify + 선형 임베딩 + sanity check를 구현합니다. 학습은 하지 않습니다. 현실적인 규모의 ViT에는 PyTorch와 여러 시간의 GPU 시간이 필요합니다.

### 1단계: 가짜 이미지

`(R, G, B)` 튜플의 행 리스트로 된 24 × 24 RGB 이미지입니다. 6×6 패치를 사용해 패치 16개를 만들고, 각 패치는 108차원 임베딩 벡터가 됩니다.

### 2단계: patchify

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

래스터 순서, 즉 격자를 행 우선으로 훑는 순서입니다. 모든 ViT가 이 순서를 사용합니다.

### 3단계: linear embed

각 평평한 패치에 무작위 `(patch_flat_size, d_model)` 행렬을 곱합니다. `[CLS]`를 앞에 붙인 뒤 출력 형태가 `(N_patches + 1, d_model)`인지 확인합니다.

### 4단계: 현실적인 ViT의 파라미터 수 세기

ViT-Base의 파라미터 수를 출력합니다. 설정은 12 layers, 12 heads, d=768, patch=16입니다. ResNet-50(~25M)과 비교하세요. ViT-Base는 약 86M입니다. ViT-Large는 약 307M, ViT-Huge는 약 632M입니다.

## 사용하기

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 임베딩은 2026년 이미지 특징의 기본값입니다.** 백본을 고정하고 작은 head만 학습하세요. 분류, 검색, 탐지, 캡셔닝에 모두 동작합니다. Meta의 DINOv2 체크포인트는 텍스트가 아닌 모든 비전 작업에서 CLIP을 능가합니다.

**패치 크기 선택.** 작은 모델은 16×16(ViT-B/16)을 사용합니다. 조밀한 예측(분할)에는 8×8 또는 14×14(SAM, DINOv2)를 사용합니다. 아주 큰 모델은 14×14를 사용합니다.

## 실전 적용

`outputs/skill-vit-configurator.md`를 보세요. 이 스킬은 데이터셋 크기, 해상도, compute 예산을 바탕으로 새 비전 작업에 맞는 ViT 변형과 패치 크기를 고릅니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 패치 수가 `(H/P) * (W/P)`와 같고, 평평한 패치 차원이 `P*P*C`와 같은지 확인하세요.
2. **보통.** 2D 사인파 위치 임베딩을 구현하세요. 각 패치의 `row`와 `col`에 대해 독립적인 사인파 코드를 만들고 이어 붙입니다. 작은 PyTorch ViT에 넣고 CIFAR-10에서 학습 가능한 위치 임베딩과 정확도를 비교하세요.
3. **어려움.** 3-layer ViT(PyTorch)를 만들고, 4×4 패치로 MNIST 이미지 1,000장에 학습하세요. 테스트 정확도를 측정하세요. 이제 같은 1,000장에 DINOv2 방식의 사전학습을 추가하세요. 단순화해서, 마스크된 패치로부터 패치 임베딩을 예측하도록 인코더만 학습합니다. 정확도가 오르나요?

## 핵심 용어

| 용어 | 사람들이 보통 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Patch | "비전 트랜스포머 토큰" | 이미지의 `P × P × C` 영역에 있는 픽셀값을 평평하게 만든 벡터입니다. |
| Patchify | "자르고 평평하게 만들기" | 이미지를 겹치지 않는 패치로 자르고, 각각을 벡터로 펼칩니다. |
| `[CLS]` token | "이미지 요약" | 앞에 붙는 학습 가능한 토큰입니다. 최종 임베딩이 이미지 표현입니다. |
| Inductive bias | "모델이 가정하는 것" | ViT는 CNN보다 사전 가정이 적습니다. 그 차이를 메우려면 더 많은 데이터가 필요합니다. |
| DINOv2 | "자기지도 ViT" | 이미지 증강 + momentum teacher로 레이블 없이 학습합니다. 2026년 최고의 범용 이미지 특징입니다. |
| SigLIP | "CLIP의 후속" | 시그모이드 대조 손실로 학습한 ViT + 텍스트 인코더입니다. 동일 compute에서 CLIP보다 좋습니다. |
| Swin | "윈도우 ViT" | 지역 어텐션 + shifted window를 쓰는 계층형 ViT입니다. 준제곱 비용입니다. |
| Register tokens | "2023년 트릭" | attention sink를 흡수하는 몇 개의 추가 학습 가능 토큰입니다. DINOv2 특징을 개선합니다. |

## 더 읽을거리

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 논문입니다.
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT입니다.
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin입니다.
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2입니다.
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — DINOv2를 위한 register-token 수정입니다.
