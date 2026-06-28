---
name: skill-depth-to-pointcloud
description: 올바른 intrinsics handling과 .ply export로 depth maps에서 point clouds를 만듭니다
version: 1.0.0
phase: 4
lesson: 26
tags: [depth, point-cloud, 3d, intrinsics]
---

# Depth를 Point Cloud로 변환

depth map과 colour image를 textured point cloud로 바꿉니다. visualisation이나 추가 3D 작업을 위해 export할 수 있습니다.

## 언제 사용할까

- depth predictions를 실제 3D scene처럼 시각화할 때.
- single image에서 sparse 3D reconstruction을 bootstrapping할 때.
- SfM이 실패할 때 3DGS training input을 만들 때.
- predicted depth를 LiDAR ground truth와 비교할 때.

## 입력

- `depth`: output에서 원하는 단위의 depths를 담은 `(H, W)` numpy array(metres 권장).
- `rgb`: colours를 담은 `(H, W, 3)` numpy array(uint8 또는 float32 [0, 1]).
- `intrinsics`: pixel units의 `(fx, fy, cx, cy)`.
- 선택적 `depth_scale`: predicted depth units를 metres로 변환하는 multiplier.

## 파이프라인

1. **Validate** - 포함하려는 모든 곳에서 depth는 positive and finite여야 합니다. invalid pixels를 mask out합니다.
2. **Lift** - pixel별로 `X = (u - cx) * d / fx`, `Y = (v - cy) * d / fy`, `Z = d`.
3. **Pair** with RGB - 각 3D point는 대응 pixel의 `(r, g, b)` triple을 받습니다.
4. **Export** - PLY(portable), `.xyz`(lightweight), `.pcd`(Open3D-native), `.las`/`.laz`(geospatial).

## 구현 템플릿

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics, depth_scale=1.0, min_depth=0.1, max_depth=100.0):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def write_ply(path, points, colors=None, valid_mask=None):
    p = points.reshape(-1, 3)
    if valid_mask is not None:
        p = p[valid_mask.flatten()]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {p.shape[0]}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.uint8)
        if valid_mask is not None:
            c = c[valid_mask.flatten()]
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is not None:
            for pt, col in zip(p, c):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
        else:
            for pt in p:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
```

## 보고서

```text
[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised
```

## 규칙

- invalid depth(zero, NaN, inf, saturated)는 항상 mask합니다. 포함하면 origin 주변에 garbage points cloud가 생깁니다.
- relative-depth model의 prediction이면 metric으로 export하지 마세요. convention을 알리기 위해 output filename에 `relative_` prefix를 붙입니다.
- camera coordinate convention을 일관되게 유지합니다(OpenCV: +X right, +Y down, +Z forward). downstream tool이 OpenGL(+Y up)을 기대하면 sign을 바꿉니다.
- dense scenes(> 1M points)에는 subsample parameter를 제공합니다. PLY files > 500 MB는 어디서든 로드하기 번거롭습니다.
- "reasonable" output을 만들려고 depth를 조용히 clip하지 마세요. 무엇이 버려졌는지 사용자가 알 수 있도록 threshold를 경고와 함께 명시적으로 clip합니다.
