---
name: skill-latency-profiler
description: warmup, synchronisation, percentiles, memory tracking을 포함한 완전한 latency-benchmarking script를 작성합니다
version: 1.0.0
phase: 4
lesson: 15
tags: [edge, deployment, profiling, benchmarking]
---

# Latency Profiler

어떤 PyTorch 모델이든 규율 있는 latency benchmark를 만듭니다. downstream의 누구라도 실제로 신뢰할 수 있는 report를 냅니다.

## 사용할 때

- 배포할 backbone을 고르기 전에 여러 candidate backbone을 비교할 때.
- quantisation 또는 pruning 전후.
- runtime 변경(eager vs ONNX vs TensorRT) 후.
- deployment-readiness report를 생성할 때.

## 입력

- `model`: PyTorch `nn.Module`.
- `input_shape`: `(1, 3, 224, 224)` 같은 tuple.
- `device`: `cpu` | `cuda` | `mps`.
- `warmup`: 기본값 10.
- `iters`: 기본값 100.

## 확인 항목

### 1. Warmup
시간을 재지 않고 모델을 `warmup`번 실행합니다. 첫 forward의 JIT compilation과 cold cache effect를 잡아냅니다.

### 2. Synchronisation
`cuda`에서는 각 timed forward pass 전후에 `torch.cuda.synchronize()`를 호출합니다.
`mps`에서는 `torch.mps.synchronize()`를 호출합니다.

### 3. Timer
벽시계 측정에는 `time.perf_counter()`를 사용합니다. millisecond로 변환합니다.

### 4. Percentiles
전체 timing list를 정렬합니다. `p50, p90, p95, p99, mean, std`를 보고합니다.

### 5. Memory
`cuda`에서는 실행 후 `torch.cuda.max_memory_allocated()`를 호출하고 baseline을 뺍니다.
`cpu`에서는 전후에 `tracemalloc` 또는 `psutil.Process().memory_info().rss`를 사용합니다.

### 6. Batch-size sweep
선택적으로 `batch_size in [1, 4, 16, 32]`에 대해 benchmark를 반복해 throughput vs latency tradeoff를 드러냅니다.

## 출력 템플릿

```python
import time
import torch
import psutil, os

def profile(model, input_shape, device="cpu", warmup=10, iters=100):
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss / 1e6

    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    mean = sum(times) / len(times)
    std  = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def pct(p):
        idx = max(0, min(len(times) - 1, int(len(times) * p) - 1))
        return times[idx]

    report = {
        "p50_ms":  pct(0.50),
        "p90_ms":  pct(0.90),
        "p95_ms":  pct(0.95),
        "p99_ms":  pct(0.99),
        "mean_ms": mean,
        "std_ms":  std,
        "rss_mb":  proc.memory_info().rss / 1e6 - baseline_rss,
    }
    if device == "cuda":
        report["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return report
```

## 규칙

- 항상 warmup을 실행하세요. 첫 forward timing은 절대 믿지 마세요.
- mean이 아니라 percentile입니다. outlier 하나가 mean을 두 배로 만들 수 있지만 p50은 거의 움직이지 않을 수 있습니다.
- production과 같은 input_shape를 사용하세요. 224x224의 latency는 384x384의 latency가 아닙니다.
- CUDA에서는 `torch.cuda.synchronize()`를 절대 빼지 마세요. 없으면 숫자는 의미가 없습니다.
- 숫자와 함께 torch version, CUDA version, device name을 기록하세요. 그렇지 않으면 비교할 수 없게 됩니다.
