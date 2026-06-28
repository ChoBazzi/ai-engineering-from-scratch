# GPU 설정과 Cloud

> 학습을 위해서는 CPU training도 괜찮다. 실제 training에는 GPU가 필요하다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 minutes

## 학습 목표

- `nvidia-smi`와 PyTorch의 CUDA API를 사용해 local GPU availability를 확인한다
- 무료 cloud 기반 experiment를 위해 T4 GPU가 있는 Google Colab을 구성한다
- CPU와 GPU의 matrix multiplication을 benchmark하고 speedup을 측정한다
- fp16 rule of thumb을 사용해 VRAM에 들어갈 수 있는 가장 큰 model을 추정한다

## 문제

phases 1-3의 대부분 lesson은 CPU에서 잘 실행된다. 하지만 CNN, transformer, LLM(phases 4+)을 training하기 시작하면 GPU acceleration이 필요하다. CPU에서 8시간 걸리는 training run이 GPU에서는 10분이면 끝난다.

선택지는 세 가지다: local GPU, cloud GPU, Google Colab(free).

## 개념

```
선택지:

1. Local NVIDIA GPU
   Cost: $0 (이미 가지고 있는 경우)
   Setup: CUDA + cuDNN 설치
   Best for: 정기적인 사용, 큰 dataset

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: 빠른 experiment, 집에 GPU가 없는 경우

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: 본격적인 training, 큰 model
```

## 직접 만들기

### 선택지 1: Local NVIDIA GPU

GPU가 있는지 확인한다:

```bash
nvidia-smi
```

CUDA가 포함된 PyTorch를 설치한다:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 선택지 2: Google Colab

1. [colab.research.google.com](https://colab.research.google.com)으로 이동한다
2. Runtime > Change runtime type > T4 GPU를 선택한다
3. 확인을 위해 `!nvidia-smi`를 실행한다

이 course의 notebook을 Colab에 직접 upload한다.

### 선택지 3: Cloud GPU

Lambda Labs, RunPod, Vast.ai를 사용할 때:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### GPU가 없나요? 괜찮습니다.

대부분의 lesson은 CPU에서 동작한다. GPU가 필요한 lesson은 그렇게 명시하고 Colab link를 포함한다.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 직접 만들기: GPU vs CPU benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 연습 문제

1. 위 benchmark를 실행하고 CPU와 GPU 시간을 비교한다
2. GPU가 없다면 Google Colab에서 실행하고 비교한다
3. GPU memory가 얼마나 있는지 확인하고 들어갈 수 있는 가장 큰 model을 추정한다(rule of thumb: fp16에서는 parameter당 2 bytes)

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| CUDA | "GPU programming" | GPU에서 code를 실행할 수 있게 해 주는 NVIDIA의 parallel computing platform |
| VRAM | "GPU memory" | system RAM과 분리된 GPU의 Video RAM. model size를 제한한다. |
| fp16 | "Half precision" | 16-bit floating point. accuracy loss를 최소화하면서 fp32의 절반 memory를 사용한다 |
| Tensor Core | "빠른 matrix hardware" | matrix multiplication을 위한 특수 GPU core. 일반 core보다 4-8배 빠르다 |
