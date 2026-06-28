---
name: skill-image-tensor-inspector
description: 이미지 형태의 텐서나 배열을 검사하고 dtype, layout, range, raw/normalized/standardized 여부를 보고합니다
version: 1.0.0
phase: 4
lesson: 1
tags: [computer-vision, debugging, preprocessing, tensors]
---

# 이미지 텐서 검사기

비전 파이프라인의 어느 지점에서든 이미지 형태의 배열을 들고 있고, 그 배열이 정확히 어떤 상태인지 알아야 할 때 쓰는 진단 skill입니다.

## 언제 사용할까

- 사전 학습 모델이 엉뚱한 예측을 반환하고 preprocessing이 의심될 때.
- OpenCV와 torchvision 사이에서 파이프라인을 옮기는데 channel order가 불분명할 때.
- 여러 framework의 layer를 쌓는 중 batch axis가 계속 잘못된 위치에 나타날 때.
- loss가 `log(num_classes)`에 멈춰 있는 training loop를 디버깅할 때.

## 입력

- `x`: 임의의 2-D, 3-D, 4-D array-like(NumPy, PyTorch, JAX).
- 선택적 `expected`: 비교 검사할 invariant dict. 예: `{"layout": "CHW", "range": "standardized"}`.

## 절차

1. **backend 확인** - `x`가 NumPy, Torch, JAX 중 무엇인지 감지합니다. 원본을 바꾸지 않고 검사를 위해 NumPy로 변환합니다.

2. **rank 분류**:
   - rank 2 -> single-channel image (H, W).
   - rank 3 -> 마지막 축이 1, 3, 4 중 하나이고 다른 두 축보다 엄격히 작으면 `HWC`, 아니면 `CHW`.
   - rank 4 -> axis 1이 {1, 3, 4}에 있고 **동시에** axis 2나 axis 3 중 하나가 16보다 크면 `NCHW`를 우선합니다. 아니면 `NHWC`를 우선합니다. axis-1만 검사하면 `(3, 4, 224, 3)` 같은 작은 이미지 NHWC batch를 잘못 분류합니다.
   - `(1, 3, 3, 3)` 같은 모호한 경우는 추측하지 말고 항상 `ambiguous`로 표시합니다. 호출자가 `expected`를 제공하도록 요구합니다.

3. **dtype과 range 분류**:
   - [0, 255] 범위의 `uint8` -> `raw`.
   - min >= 0이고 max <= 1.01인 `float*` -> `normalized`.
   - min < 0이고 |mean| < 0.5이며 0.5 <= std <= 1.5인 `float*` -> `standardized`.
   - 그 밖의 경우 -> `unusual`, histogram을 출력합니다.

4. **채널별 통계** - 채널별 mean과 std를 보고합니다. 배열이 standardized처럼 보이면 ImageNet mean/std와 비교하고 match confidence를 표시합니다.

5. **보고서**는 다음 정확한 block으로 출력합니다.

```text
[inspector]
  backend:   numpy | torch | jax
  rank:      2 | 3 | 4
  layout:    HW | HWC | CHW | NHWC | NCHW
  dtype:     <dtype>
  shape:     <shape>
  range:     raw | normalized | standardized | unusual
  min/max:   <min> / <max>
  per-channel mean: [ ... ]
  per-channel std:  [ ... ]
  likely source:    camera | PIL | OpenCV | torchvision | random init
  likely target:    display | training | inference
```

6. `likely target`을 기준으로 **다음 조치**를 추천합니다.
   - `display`: HWC로 transpose하고, clip한 뒤, uint8로 변환합니다.
   - `training`: dataset stats로 standardize하고, CHW로 transpose한 뒤, batch axis를 추가합니다.
   - `inference`: model card의 정확한 invariant와 맞춥니다.

## 규칙

- 입력을 절대 변경하지 않습니다. 진단만 출력합니다.
- `expected`가 제공되면 모든 mismatch를 `[expected X got Y]`로 표시합니다.
- layout이나 channel order가 모호하면 silent-failure 위험을 명시합니다.
- 옵션 목록이 아니라 한 번에 하나의 조치만 추천합니다.
