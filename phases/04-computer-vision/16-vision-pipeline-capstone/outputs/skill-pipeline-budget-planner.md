---
name: skill-pipeline-budget-planner
description: target latency와 throughput이 주어지면 모든 pipeline stage에 time budget을 배정하고 어떤 stage가 먼저 budget을 놓칠지 표시합니다
version: 1.0.0
phase: 4
lesson: 16
tags: [vision, pipeline, performance, deployment]
---

# Pipeline Budget Planner

latency/throughput target을 stage별 budget으로 바꿔 모든 team member가 어떤 숫자를 목표로 engineering하는지 알게 합니다.

## 사용할 때

- 새 vision service를 만들기 전에 각 stage의 expectation을 정할 때.
- 첫 benchmark 후 어떤 stage가 budget에서 가장 먼지 볼 때.
- SLA가 바뀌어 budget을 다시 협상해야 할 때.

## 입력

- `p95_latency_target_ms`: request별 budget.
- `target_qps`: replica별 throughput.
- `stages`: `{ name: str, current_ms: float }`의 list.

## 배정 규칙

현재 측정값이 없을 때 일곱 standard stage에 대한 기본 배정:

| Stage | Share |
|-------|-------|
| decode + preprocess | 15% |
| detector forward | 55% |
| postprocess detections (NMS, clamp) | 5% |
| crop + resize for classifier | 5% |
| classifier forward | 15% |
| schema validation | <1% |
| response serialisation | 4% |

GPU-bound pipeline(cloud)에서는 detector share가 70%까지 올라가는 경우가 많습니다. CPU에서는 preprocessing과 classifier batching이 더 많이 먹습니다.

## 보고서

```text
[budget plan]
  p95 target:  <ms>
  throughput:  <qps per replica>

| stage               | target_ms | current_ms | headroom | gate |
|---------------------|-----------|------------|----------|------|
| decode+preprocess   | ...       | ...        | ...      | ok|X |
| detector            | ...       | ...        | ...      | ok|X |
| ...                 | ...       | ...        | ...      |      |

[bottleneck]
  stage:  <name>
  miss:   <ms over budget>
  lever:  <specific action>

[levers]
  decode+preprocess:   Pillow-SIMD, libjpeg-turbo, decode on GPU via NVJPEG
  detector:            smaller backbone, lower input resolution, INT8, TensorRT
  postprocess:         GPU-side NMS (torchvision.ops), fused masks
  crop+resize:         GPU crop with grid_sample, batched interpolate
  classifier:          smaller backbone, INT8, warm cache, batch
  schema:              skip validation in hot path, validate at boundaries only
  response:            orjson, stream protobuf
```

## 규칙

- production path에서 schema validation을 제거하라고 추천하지 마세요. 대신 boundary로 옮기라고 제안하세요.
- preprocessing이 budget을 놓치면 model을 바꾸기 전에 항상 Pillow-SIMD 또는 NVJPEG를 먼저 시도하세요.
- detector miss가 target의 30%보다 크면 현재 모델을 최적화하지 말고 model을 바꾸세요.
- current_ms > 1.1 * target_ms이면 gate를 `X`로 표시하고, budget의 10% 이내이면 `ok`로 표시하세요.
