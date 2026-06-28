---
name: skill-point-cloud-loader
description: 올바른 정규화, 중심 맞추기, 점 샘플링을 갖춘 .ply / .pcd / .xyz 파일용 PyTorch Dataset 작성
version: 1.0.0
phase: 4
lesson: 13
tags: [3d-vision, point-cloud, data-loading, pytorch]
---

# 포인트 클라우드 로더

3D 스캔 파일 폴더를 바로 학습할 수 있는 PyTorch `Dataset`으로 바꿉니다.

## 사용 시점

- 새 포인트 클라우드 분류 / 세그멘테이션 프로젝트를 시작할 때.
- `.ply`, `.pcd`, `.xyz` 형식을 오갈 때.
- 모델이 오류 없이 학습되지만 수렴이 나쁠 때 디버깅하기 위해. 이 경우 데이터 로더 정규화가 잘못된 경우가 많습니다.

## 입력

- `data_root`: 포인트 클라우드 파일 폴더와 선택적 레이블 CSV.
- `file_format`: ply | pcd | xyz | npy.
- `num_points`: 고정 샘플링 크기. 보통 1024 또는 2048.
- `augmentation`: none | rotate | jitter | mixup.

## 정규화 정책

모든 프로덕션 포인트 클라우드 파이프라인은 다음 순서로 적용합니다:

1. 클라우드를 **중심 맞추기**: centroid를 뺍니다.
2. 단위 구로 **스케일링**: 중심으로부터의 최대 거리로 나눕니다.
3. `num_points`개 점을 **샘플링**합니다. 클라우드에 더 많은 점이 있으면 형태를 충실히 보존하기 위해 **farthest point sampling**(FPS)을 사용하거나 속도를 위해 random sampling을 사용합니다. 더 적으면 점을 반복합니다.
4. 점 순서를 **셔플**합니다. 어차피 모델에는 순서가 중요하지 않아야 하지만, 셔플은 우발적인 순서 의존성을 끊습니다.

## 출력 템플릿

```python
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def _read_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    # Fallback: minimal ascii-ply reader
    ...

def _fps(points, k):
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    seed = np.random.randint(len(points))
    idx[0] = seed
    for i in range(1, k):
        dist = np.minimum(dist, ((points - points[idx[i-1]]) ** 2).sum(axis=1))
        idx[i] = int(np.argmax(dist))
    return idx

def normalise(points):
    centre = points.mean(axis=0)
    points = points - centre
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / max(scale, 1e-8)

class PointCloudDataset(Dataset):
    def __init__(self, files, labels, num_points=1024, augment=False):
        self.files = files
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pts = _read_ply(self.files[i])
        pts = normalise(pts)
        if len(pts) >= self.num_points:
            idx = _fps(pts, self.num_points)
            pts = pts[idx]
        else:
            reps = int(np.ceil(self.num_points / len(pts)))
            pts = np.tile(pts, (reps, 1))[:self.num_points]
        # Shuffle point order to break any accidental dependencies (especially
        # important when tiling repeats points in deterministic order).
        np.random.shuffle(pts)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
            pts = pts @ R
            pts = pts + np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        return torch.from_numpy(pts).transpose(0, 1), int(self.labels[i])
```

## 보고서

```text
[dataset]
  files:          <N>
  format:         <ply|pcd|xyz|npy>
  points_per_sample: <int>
  normalise:      centre + unit sphere
  sampling:       FPS | random
  augmentation:   <list>
```

## 규칙

- 항상 스케일링 전에 중심을 맞춥니다. 순서를 바꾸면 "unit sphere"의 의미가 달라집니다.
- 형태 작업에는 random sampling보다 FPS를 선호합니다. 모든 점이 어차피 중요한 세그멘테이션에서는 random도 괜찮습니다.
- 평가 중에는 절대 augmentation을 적용하지 않습니다. 학습 중에만 적용합니다.
- 포인트 클라우드 파일이 색상이나 법선 같은 추가 채널을 포함한다면, Dataset이 xyz만이 아니라 `(3 + C, num_points)` 텐서를 반환하도록 확장하세요.
