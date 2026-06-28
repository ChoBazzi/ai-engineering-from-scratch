---
name: skill-heatmap-to-coords
description: 모든 production pose model이 쓰는 sub-pixel heatmap-to-coordinate routine 작성
version: 1.0.0
phase: 4
lesson: 21
tags: [keypoint, pose, subpixel, inference]
---

# Heatmap to Coords

raw keypoint heatmap을 sub-pixel 정밀도의 coordinate로 바꾼다. 모든 pose pipeline에서 가장 저렴한 accuracy upgrade다.

## 사용할 때

- heatmap-based keypoint model을 배포할 때.
- pose metric을 benchmark할 때. OKS는 sub-pixel accuracy에 매우 민감하다.
- pose code를 한 framework에서 다른 framework로 porting할 때.

## 입력

- `heatmaps`: `(N, K, H, W)` tensor, model이 낸 keypoint별 heatmap.
- `confidence_threshold`: peak가 이 값보다 낮은 keypoint를 버린다.

## 단계

1. **Argmax** 각 heatmap에서 integer peak location을 찾는다.
2. **First-difference offset** — 이웃 pixel에서 sub-pixel offset을 추정한다. `0.25` coefficient는 `sigma >= 1`인 Gaussian heatmap에 맞춰 calibrate한 heuristic이다. 원칙적인 sub-pixel recovery가 필요하면 full quadratic fit(DARK) 또는 Gaussian fit을 사용한다.

```text
dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
```

DARK / quadratic variant는 local quadratic으로 근사한다.

```text
dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
        / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)
```

quadratic fit은 peak가 뚜렷한 heatmap에서 더 정확하다. sign-based offset은 heatmap이 noisy할 때 더 안전한 default다.

3. **Offset 추가** integer peak에 offset을 더한다.
4. **Confidence** — keypoint별 peak value를 반환한다. client는 이를 low-confidence prediction mask에 사용한다.
5. **Boundary case** — peak가 한 axis의 첫 pixel이나 마지막 pixel에 놓이면 이웃 중 하나가 clamp된다. offset은 0으로 collapse되며, 이것이 가장 안전한 fallback이다.

## 출력 template

```python
import torch

def heatmap_to_coords_subpixel(heatmaps, threshold=0.2):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    ys_int = ys.long()
    xs_int = xs.long()

    x_minus = (xs_int - 1).clamp(min=0)
    x_plus = (xs_int + 1).clamp(max=W - 1)
    y_minus = (ys_int - 1).clamp(min=0)
    y_plus = (ys_int + 1).clamp(max=H - 1)

    batch_idx = torch.arange(N).view(-1, 1).expand(-1, K)
    kp_idx = torch.arange(K).view(1, -1).expand(N, -1)

    dx_raw = (heatmaps[batch_idx, kp_idx, ys_int, x_plus]
              - heatmaps[batch_idx, kp_idx, ys_int, x_minus])
    dy_raw = (heatmaps[batch_idx, kp_idx, y_plus, xs_int]
              - heatmaps[batch_idx, kp_idx, y_minus, xs_int])
    dx = 0.25 * torch.sign(dx_raw)
    dy = 0.25 * torch.sign(dy_raw)

    at_left = xs_int == 0
    at_right = xs_int == (W - 1)
    at_top = ys_int == 0
    at_bottom = ys_int == (H - 1)
    dx = torch.where(at_left | at_right, torch.zeros_like(dx), dx)
    dy = torch.where(at_top | at_bottom, torch.zeros_like(dy), dy)

    refined_x = xs + dx
    refined_y = ys + dy
    coords = torch.stack([refined_x, refined_y], dim=-1)
    mask = conf >= threshold
    return coords, conf, mask
```

## 보고서

```text
[subpixel decode]
  keypoints:   K
  threshold:   <float>
  valid_rate:  fraction of keypoints above threshold
```

## 규칙

- neighbour index는 항상 valid range로 clamp한다. edge 바깥 keypoint는 zero-difference offset을 가지지만 crash하지 않는다.
- client가 low-confidence point를 mask할 수 있도록 coordinate와 함께 confidence를 반환한다.
- Sub-pixel refinement는 peak 주변 heatmap이 smooth할 때만 도움이 된다. training이 sigma >= 1인 Gaussian target을 사용했는지 확인한다.
- heatmap resolution이 매우 작다면(< 48x48), coordinate 추출 전에 heatmap을 full image size로 upsample하는 것을 고려한다. sub-pixel offset은 stride에 따라 scale된다.
