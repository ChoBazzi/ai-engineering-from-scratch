---
name: skill-anchor-designer
description: ground-truth box dataset이 주어지면 (w, h)에 k-means를 실행하고 FPN level별 anchor set과 coverage statistics를 반환한다
version: 1.0.0
phase: 4
lesson: 6
tags: [computer-vision, detection, anchors, kmeans]
---

# Anchor Designer

anchor는 anchor-based detector에서 가장 dataset-specific한 hyperparameter입니다. default COCO anchor는 cell-culture image, satellite tile, small-object surveillance에서 성능이 떨어집니다. 이 skill은 target data에 실제로 맞는 anchor를 도출합니다.

## 사용할 때

- 새 dataset에서 첫 training run을 시작하기 전에.
- 그 밖에는 건강한 모델인데 very small 또는 very large object의 recall이 약할 때.
- major dataset expansion 이후 box size distribution이 바뀌었을 수 있을 때.

## 입력

- `boxes`: `(cx, cy, w, h)` 또는 `(x1, y1, x2, y2)` 형식의 shape (N, 4) numpy array. 최소 1000개 positive box 권장.
- `num_anchors_per_level`: 보통 3.
- `num_fpn_levels`: 보통 3(P3, P4, P5) 또는 4.
- `input_size`: training-resolution HxW.
- Optional `strides`: level별 stride. 생략하면 `[8, 16, 32, 64]`의 처음 `num_fpn_levels` entries를 사용합니다. detector의 FPN stride가 다르면 더 길거나 짧은 array를 명시적으로 넘기세요.

## 단계

1. **box를 정규화합니다.** `input_size`에서 pixel unit의 `(w, h)` pair로 변환합니다. w 또는 h가 2 pixel 미만인 것은 버립니다.

2. **k-means를 실행합니다.** `k = num_anchors_per_level * num_fpn_levels`로 `(w, h)` pair에 실행합니다. Euclidean distance가 아니라 `1 - IoU(box, cluster)`를 distance function으로 쓰세요. `(w, h)`에 Euclidean을 쓰면 가늘고 긴 box와 square box가 함께 collapse됩니다. 모든 box는 동일하게 기여합니다(unweighted). class-imbalanced dataset에서 larger-box recall을 원한다면 weight vector를 넘기는 대신 input array에서 rare-class box를 반복하세요.

3. **cluster를 area 오름차순으로 정렬합니다.** `num_fpn_levels` group으로 나누되, group마다 `num_anchors_per_level`개를 둡니다. 가장 작은 area는 highest-resolution level(smallest stride)로 갑니다.

4. **level별 coverage statistics를 계산합니다.**
   - 각 ground-truth box와 해당 level의 best anchor 사이 `median IoU`.
   - `recall@IoU=0.5` — best anchor의 IoU가 0.5 이상인 box의 비율.
   - `area coverage` — box area가 해당 level의 `[anchor_min_area / 4, anchor_max_area * 4]` 안에 드는 비율.

5. **level별 anchor를 보고하고** `recall@IoU=0.5 < 0.9`인 level을 flag하세요. 그 level의 anchor는 data와 잘 맞지 않으므로 retune하거나 level당 anchor 수를 늘려야 합니다.

## 보고서 형식

```text
[anchor-designer]
  total boxes:         <N>
  clusters:            <k>
  distance metric:     1 - IoU

[level P3  stride=8]
  anchors (w, h):      [(A, B), (C, D), (E, F)]
  median IoU:          <X>
  recall@IoU=0.5:      <X>
  coverage:            <X>
  flag:                ok | retune

[level P4  stride=16]
  ...

[summary]
  overall recall@IoU=0.5: <X>
  smallest anchor:        <w x h>
  largest anchor:         <w x h>
  recommendation:         <one sentence if any level flagged>
```

## 규칙

- 항상 IoU-based distance를 사용하세요. Euclidean k-means는 보기에는 그럴듯하지만 경험적으로 더 나쁜 anchor를 만듭니다.
- cluster를 area 기준으로 정렬한 뒤 ascending order로 level에 할당하세요.
- `num_anchors_per_level = 1`이면 k-means를 완전히 건너뛰세요. 대신 box를 area quantile로 `num_fpn_levels`개 bin으로 나누고(예: 3 levels이면 terciles), 각 level의 anchor를 bin별 median (w, h)로 설정하세요. 작은 dataset에서는 `k = num_fpn_levels`로 k-means를 돌리는 것보다 이 방식이 더 robust합니다.
- negative anchor dimension을 절대 출력하지 마세요. 1로 clamp하세요.
- dataset에 box가 200개 미만이면 anchor search가 unreliable하다고 warning하고, default COCO anchor와 더 많은 training data 사용을 권장하세요.
