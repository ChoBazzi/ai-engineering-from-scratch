---
name: hybrid-picker
description: 주어진 workload에 대해 pure Transformer, Jamba-style hybrid, pure SSM 중 하나를 선택합니다.
version: 1.0.0
phase: 10
lesson: 21
tags: [jamba, mamba, ssm, hybrid, long-context, memory-budget, architecture]
---

workload specification(context length profile p50/p99, task mix, GPU당 memory budget, target throughput, quality-vs-speed priority)이 주어지면 pure Transformer(+MoE +MLA), Jamba-style hybrid, pure Mamba model 중 하나를 추천하세요.

다음을 산출하세요:

1. Context-length bucket. Short(16k 미만), medium(16k-64k), long(64k-256k), ultra-long(256k-plus). 첫 결정을 이끕니다.
2. Architecture recommendation. pure Transformer, 1:7 hybrid, 1:3 hybrid, 1:15 hybrid, pure Mamba 중 하나를 고르세요. context bucket과 task의 in-context-recall 요구를 함께 사용해 정당화하세요.
3. Memory budget check. 목표 context에서 KV cache + SSM state를 계산하세요. weights와 activation memory(일반적으로 weights와 KV cache 위에 10-20 GB)를 고려한 뒤 target accelerator에 들어가는지 확인하세요.
4. Quality tradeoff disclosure. 선택한 sparsity level의 quality cost를 문서화하세요. 1:7보다 낮은 hybrid ratio는 in-context retrieval에서 측정 가능한 만큼 저하되고, pure Mamba는 일부 state-tracking task에서 실패합니다.
5. Inference stack compatibility. 선택한 architecture가 target stack(vLLM, TensorRT-LLM, SGLang, llama.cpp)에서 지원되는지 확인하세요. hybrid는 pure Transformer보다 tooling coverage가 얇습니다.

강한 거부 조건:
- context가 16k 미만인데 Jamba-style hybrid를 쓰는 것. architectural overhead가 정당화되지 않습니다.
- reasoning-heavy 또는 multi-document cross-reference task에 pure Mamba를 쓰는 것. state-tracking limit가 문제가 됩니다.
- sub-1:15 hybrid ratio. 이보다 낮으면 in-context recall이 신뢰하기 어렵습니다.
- 지정된 accelerator에서 계산된 memory budget에 들어가지 않는 모든 recommendation.

거부 규칙:
- workload가 진짜로 short context와 long context가 섞여 있다면 hybrid recommendation을 거부하고 pure Transformer(가능하면 MLA 포함)를 추천하세요. hybrid는 long-context workload에서 특히 빛납니다.
- accelerator가 consumer-grade(24GB 이하)이면 hybrid-size model을 거부하고 distilled small hybrid 또는 quantized pure Transformer를 추천하세요.
- workload가 latency-sensitive batch-1 generation이고 모델이 새로워 existing deployment path가 없다면 거부하고 더 단순한 경로로 speculative decoding(Phase 10 · 15)을 갖춘 well-supported pure Transformer를 추천하세요.

출력: context bucket, architecture choice, 목표 context에서의 KV cache, quality tradeoff disclosure, inference stack compatibility를 나열한 한 페이지 recommendation. 마지막에는 첫 10k production request에서 추천을 확인할 구체적 long-context evaluation(RULER, LongBench, needle-in-haystack)을 명명하는 "what to monitor" 문단을 붙이세요.
