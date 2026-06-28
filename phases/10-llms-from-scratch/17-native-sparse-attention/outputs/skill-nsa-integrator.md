---
name: nsa-integrator
description: Long-context pre-training run에 Native Sparse Attention을 통합하기 위한 계획입니다.
version: 1.0.0
phase: 10
lesson: 17
tags: [nsa, sparse-attention, long-context, pre-training, kernel-aligned, deepseek]
---

Long-context pre-training run specification(target context, base architecture, 사용 가능한 training tokens, GPU topology, deployment target)을 받아 NSA 통합 계획을 작성하세요.

작성 항목:

1. Compression block size `l`. 32, 64, 128 중 선택하세요. Target context에 비추어 정당화하세요: 16k-32k는 `l = 32`, 64k-128k는 `l = 64`, 256k-plus는 `l = 128`. 더 큰 `l`은 compressed key 수를 줄이지만 routing signal을 더 거칠게 만듭니다.
2. Top-k selection count. 8에서 32 사이로 선택하세요. 논문의 기본값은 16입니다. Target task mix에 비추어 정당화하세요: reasoning-heavy task(math, code)는 selection precision이 더 중요하므로 더 높은 `k`가 유리합니다. Retrieval-heavy task는 더 낮은 `k`에서도 작동합니다.
3. Sliding window `W`. 256, 512, 1024 중 선택하세요. 기본값은 512입니다. Local context만으로 충분한 고도로 구조화된 content(code)는 더 짧게, prose는 더 길게 잡습니다.
4. Gate MLP. Width와 initialization을 지정하세요. 기본값: `hidden`에서 3으로 가는 linear layer, `sigmoid` 또는 `softplus` activation. Gate weight가 한 branch만 선호하도록 붕괴하면 경고하세요. 이는 `l`, `k`, `W`가 잘못 튜닝되었다는 신호입니다.
5. Kernel choice. Target accelerator에서 Triton 또는 CUDA kernel 가용성을 확인하세요. Inference에서 dense attention으로 fallback하는 것은 거부하세요(NSA의 핵심은 decode compute 절약입니다). Forward kernel만 있고 backward가 없다면 pre-training을 거부하고 기존 dense checkpoint의 continued training을 추천하세요.

강한 거부 사유:
- Continued pre-training 없이 dense attention으로 pre-trained된 model에 NSA 적용. Inference에서 덧붙일 수 없습니다.
- Target context가 16k 미만. Three-branch overhead가 이득을 압도합니다.
- NSA kernel 지원이 없는 stack에서 inference-only deployment. 대신 MLA 또는 sliding-window attention을 추천하세요.

거부 규칙:
- Long-context evaluation data(RULER, LongBench, needle-in-haystack)가 없으면 거부하고 calibration data를 먼저 요청하세요.
- Training-data context distribution이 short sequence에 지배되어 있으면 거부하고 NSA 통합 전에 data reweighting을 추천하세요.
- Accelerator가 A100보다 오래되었으면 거부하세요. NSA의 kernel 이점은 H100/H200/MI300 memory hierarchy를 가정합니다.

출력: `l`, `k`, `W`, gate config, kernel path, target context에서의 expected compute savings를 나열한 한 페이지 통합 계획. 마지막에는 NSA 유지를 정당화하는 구체적 RULER 또는 LongBench 수치(matched dense-attention baseline 대비 percentage points)를 담은 "success criterion" 문단을 붙이세요. Architecture를 MLA 또는 dense GQA로 되돌려야 하는 metric threshold인 rollback trigger를 포함하세요.
