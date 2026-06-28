---
name: skill-mot-evaluator
description: ground-truth track에 대해 MOTA / IDF1 / HOTA를 계산하는 완전한 evaluation harness를 작성한다
version: 1.0.0
phase: 4
lesson: 27
tags: [mot, evaluation, tracking, metrics]
---

# MOT Evaluator

문헌과 공정하게 비교할 수 있도록 tracker 출력을 표준 MOTA/IDF1/HOTA pipeline으로 감싼다.

## 언제 사용하는가

- 새 tracker를 MOT17 / MOT20 / DanceTrack / SportsMOT에서 benchmark할 때.
- 자체 footage에서 ByteTrack, BoT-SORT, SAM 2를 비교할 때.
- 논문이나 PR description에 넣을 재현 가능한 숫자를 만들 때.

## 입력

- `predictions`: 프레임별 `(track_id, x, y, w, h, confidence)` tuple list.
- `ground_truth`: 프레임별 `(gt_id, x, y, w, h)` tuple list.
- `iou_threshold`: MOTA에서는 보통 0.5. HOTA는 sweep을 사용한다.
- `evaluator`: `py-motmetrics`(MOTA, IDF1) 또는 `TrackEval`(HOTA).

## 출력 형식 계약

`py-motmetrics`와 `TrackEval`은 모두 특정 on-disk format을 기대한다.

```text
# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

Frame은 1-indexed이고, box는 (x1, y1, x2, y2)가 아니라 (x, y, w, h)다. 대부분의 integration bug는 변환에서 나온다.

## 단계

1. Tracker 출력을 MOT Challenge text format으로 변환한다.
2. 두 파일 모두에 `py-motmetrics.io.loadtxt`를 실행한다.
3. `mm.metrics.create().compute()`로 MOTA + IDF1을 계산한다.
4. HOTA는 같은 파일과 `Metrics: HOTA`로 `TrackEval`을 호출한다.
5. Dashboard용으로 결과를 JSON으로 저장한다.

## 구현 스케치

```python
import motmetrics as mm

def evaluate_mota_idf1(pred_path, gt_path):
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=0.5)
    metrics = mm.metrics.create().compute(
        acc, metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches"]
    )
    return metrics


def write_mot_txt(predictions, path):
    with open(path, "w") as f:
        for frame_idx, detections in enumerate(predictions, start=1):
            for tid, x, y, w, h, conf in detections:
                f.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
```

## 보고서

```text
[mot evaluation]
  frames:     <int>
  gt tracks:  <int>
  pred tracks: <int>

[metrics]
  MOTA:       <float>
  MOTP:       <float>
  IDF1:       <float>
  IDP/IDR:    <float/float>
  ID switches: <int>
  HOTA:       <float>  (from TrackEval)
```

## 규칙

- 출력 text file에서는 항상 1-indexed frame을 사용한다. MOT tooling이 이를 기대한다.
- 쓰기 전에 (x1, y1, x2, y2)를 (x, y, w, h)로 변환한다.
- 현대적인 비교에서 MOTA만 단독으로 보고하지 않는다. IDF1과 HOTA를 포함한다.
- MOT17의 private detection과 public detection에 주의한다. 이들은 별도로 평가되며 섞으면 score가 부풀려진다.
- Sequence별 score를 log한다. Aggregate는 어려운 단일 sequence의 실패를 숨긴다.
