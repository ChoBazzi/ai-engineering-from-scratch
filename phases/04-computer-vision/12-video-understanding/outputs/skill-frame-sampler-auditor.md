---
name: skill-frame-sampler-auditor
description: Off-by-one, short-clip handling, crop consistency를 기준으로 video pipeline의 frame sampler를 감사한다
version: 1.0.0
phase: 4
lesson: 12
tags: [computer-vision, video, sampling, debugging]
---

# Frame Sampler Auditor

Frame sampling은 video pipeline이 깨지는 지점이다. 여기서 생긴 bug는 모든 downstream metric으로 전파된다.

## 사용 시점

- 새 video data loader를 작성할 때.
- 논문의 숫자를 재현하는데 training accuracy가 보고값보다 낮을 때.
- Eval accuracy가 run마다 불안정한 video model을 debugging할 때.

## 입력

- `sampler_code`: (num_frames_total, T)를 받아 T indices를 반환하는 Python function.
- `T`: target clip length.
- Optional test cases: exercise할 `num_frames_total` 값. 예: `[3, T-1, T, T+1, 30, 300, 3000]`.

## Checks

### 1. Short clip handling
`num_frames_total < T`를 넣는다. 반환되는 모든 index는 `[0, num_frames_total - 1]` 안에 있어야 한다. 표준 padding policy는 남은 position에 마지막 frame을 반복하는 것이다.

### 2. Boundary indices
`num_frames_total == T`를 넣는다. 반환 index는 정확히 `[0, 1, ..., T-1]`이어야 한다.

### 3. Uniform distribution
`num_frames_total == 10 * T`를 넣는다. 반환 index는 monotonically increasing이어야 하고 대략 균등한 간격이어야 한다.

### 4. Dense window bounds
Dense sampling의 경우 `num_frames_total == 3 * T`를 넣는다. 반환 index는 contiguous window를 형성해야 하며 clip 끝을 절대 넘지 않아야 한다.

### 5. Determinism
같은 input과 deterministic sampler의 경우 같은 RNG로 sampler를 두 번 호출한다. Index가 일치해야 한다.

### 6. Crop consistency
Pipeline이 frame마다 spatial crop도 반환한다면, 같은 seed로 같은 clip에 sampler를 두 번 실행하고 모든 frame이 같은 crop box(같은 `(x, y, w, h)`)를 사용하는지 확인한다. 한 clip 안에서 frame마다 crop이 다르면 temporal coherence가 깨지며, 이는 고전적인 silent bug다. 허용되는 variation: *per clip*으로 적용되고 clip 안에서는 일관된 augmentation.

## Report

```text
[sampler audit]
  name: <function name>
  T:    <int>

[short-clip handling]
  passed | failed (<details>)

[boundary]
  passed | failed

[uniform spacing]
  passed | failed (<stddev of gaps>)

[dense window]
  passed | failed (<details>)

[determinism]
  passed | failed

[crop consistency]
  passed | failed (<per-frame crop varies: yes/no>)

[verdict]
  ok | fix required
```

## 규칙

- Short-clip handling이 out-of-range index를 반환하면 sampler를 "ok"로 표시하지 않는다.
- Dense sampler는 `num_frames_total - 1`을 넘는 window를 절대 반환하면 안 된다.
- Sampler가 stochastic(dense)이면 명시적으로 seeded RNG가 있을 때만 determinism을 test한다.
- Canonical policy를 제안하되 조용히 고치지는 않는다. 마지막 frame으로 pad, window를 end에 clamp, half-open interval을 round한다.
