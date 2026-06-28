---
name: skill-segmentation-mask-inspector
description: 클래스 분포, 예측 마스크 통계, 과소 예측되거나 경계가 흐려질 가능성이 가장 높은 클래스를 보고합니다
version: 1.0.0
phase: 4
lesson: 7
tags: [computer-vision, segmentation, debugging, evaluation]
---

# Segmentation Mask Inspector

"loss가 내려갔다"와 "마스크가 실제로 맞아 보인다" 사이의 차이를 진단합니다.

## 사용할 때

- 훈련 직후 mIoU는 괜찮아 보이지만 시각 검사는 그렇지 않다고 말할 때.
- 배포 전: 예측의 클래스 균형을 ground truth와 비교해 확인할 때.
- 큰 객체의 클래스별 IoU는 높지만 작은 객체에서는 낮을 때.
- 픽셀 수가 작아 IoU에는 잘 드러나지 않는 경계 artifact를 디버깅할 때.

## 입력

- `preds`: 예측 클래스 ID의 (N, H, W) 텐서.
- `targets`: ground-truth 클래스 ID의 (N, H, W) 텐서.
- `num_classes`: 정수.
- 선택적 `class_names`: C개 문자열의 리스트.

## 단계

1. **클래스 픽셀 히스토그램.** `preds`와 `targets`에서 클래스별 픽셀 비율을 계산합니다. `|pred% - gt%| / max(gt%, 1e-6) > 0.30`인 클래스(상대 편차 30% 초과)를 표시합니다. ground truth에 없는 클래스(`gt% == 0`)는 예측 비율이 `0.3`을 넘으면 바로 표시합니다.

2. **클래스별 IoU**와 **클래스별 boundary F1**. Boundary F1은 각 마스크를 3픽셀 dilate하고 교집합을 구해 점수화합니다. IoU > 0.7이지만 boundary F1 < 0.5인 클래스는 에지가 흐려지고 있습니다.

3. **작은 객체 재현율.** 모든 ground-truth connected component를 크기 bucket(tiny < 100 px, small < 1000 px, medium < 10000 px, large >= 10000 px)으로 나눕니다. 클래스별, bucket별 recall을 보고합니다. 큰 객체 recall은 0.9보다 높지만 작은 객체 recall이 0.3보다 낮으면 해상도 / receptive-field 문제를 뜻합니다.

4. **혼동 쌍.** 각 클래스에 대해 가장 자주 혼동되는 클래스(그 클래스의 ground-truth 마스크 안에서 가장 흔한 잘못된 예측 클래스)를 찾습니다. 상위 3개 쌍을 보고합니다.

5. **포화 검사(`preds`만으로는 불가능하며 `probs` 또는 `logits` 필요).** 호출자가 raw per-pixel 확률 분포 `probs: (N, C, H, W)`를 전달하면, 클래스별로 `probs.max(dim=1) > 0.99`인 픽셀 비율을 계산합니다. 높은 포화도(한 클래스 픽셀의 >0.9)는 과신을 시사합니다. label smoothing 또는 calibration 후보입니다. argmax된 `preds`만 있으면 이 단계를 건너뛰고 보고서에 명시합니다.

## 보고서 형식

```text
[mask-inspector]
  classes: C

[class distribution]
  name       gt %    pred %   delta
  ...

[metrics]
  class       IoU     bF1    recall_tiny  recall_small  recall_medium  recall_large
  ...

[confusion pairs]
  class A confused with class B: <N> pixels (most common)
  class B confused with class A: <N> pixels
  ...

[verdict]
  most impactful issue: <one sentence>
```

## 규칙

- 가장 빈번한 클래스가 먼저 오도록 클래스 행을 gt pixel share 내림차순으로 정렬합니다.
- IoU < 0.4 또는 boundary F1 < 0.3인 클래스를 `critical`로 표시합니다.
- small-object recall이 지배적인 실패라면 더 높은 해상도 훈련, 마지막 인코더 단계의 더 작은 stride, 또는 feature-pyramid decoder를 권장합니다.
- boundary F1이 지배적인 실패라면 boundary-aware loss(Lovasz 또는 BoundaryLoss), horizontal flip을 쓰는 TTA, stride-less decoder를 권장합니다.
- 클래스 인덱스만 유일한 식별자로 출력하지 마세요. `class_names`가 제공되면 모든 행에서 사용하세요.
