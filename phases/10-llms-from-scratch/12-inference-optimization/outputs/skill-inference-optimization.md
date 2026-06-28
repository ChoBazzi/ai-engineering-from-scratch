---
name: skill-inference-optimization
description: LLM inference serving의 throughput, latency, cost를 진단하고 최적화합니다.
version: 1.0.0
phase: 10
lesson: 12
tags: [inference, kv-cache, batching, speculative-decoding, vllm, optimization]
---

# LLM Inference Optimization Pattern

두 단계가 있습니다. prefill은 compute-bound이고 parallel합니다. decode는 memory-bound이고 sequential합니다. 모든 optimization은 둘 중 하나 또는 둘 다를 겨냥합니다.

```text
Request -> Prefill (process prompt) -> Decode (generate tokens) -> Response
              |                            |
         Compute-bound               Memory-bound
         Optimize: fusion,           Optimize: batching,
         prefix caching              quantization, speculation
```

## 결정 framework

### Step 1: bottleneck 식별

workload의 ops:byte ratio를 측정하세요.

| ops:byte | Bound | What to optimize |
|----------|-------|-----------------|
| < 50 | Memory | KV cache를 quantize하고 batch size를 늘리세요 |
| 50-200 | Transitional | 둘 다 중요하므로 batching부터 시작하세요 |
| > 200 | Compute | Kernel fusion, tensor parallelism, FP8 |

### Step 2: engine 선택

- **Default**: vLLM(가장 넓은 model support, PagedAttention, OpenAI-compatible API)
- **Multi-turn / structured output**: SGLang(RadixAttention prefix caching, constrained decoding)
- **Max NVIDIA throughput**: TensorRT-LLM(kernel fusion, H100의 FP8)

### Step 3: 순서대로 optimization 적용

1. **KV cache** -- 항상 켜세요. downside가 없습니다.
2. **Continuous batching** -- 항상 켜세요. downside가 없습니다(vLLM/SGLang은 기본적으로 수행).
3. **Prefix caching** -- 공유 system prompt가 있으면 켜세요(대부분 chatbot에 있음).
4. **Quantization** -- KV cache INT8/FP8은 quality loss를 작게 유지하면서 memory를 2-4배 줄입니다.
5. **Speculative decoding** -- throughput보다 latency가 더 중요할 때 추가하세요.
6. **Tensor parallelism** -- model이 한 GPU에 들어가지 않을 때 GPU에 나누세요.

## KV cache memory 공식

```text
per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_param
total = per_token * sequence_length * num_concurrent_users
```

일반 model의 quick reference(BF16):

| Model | Per token | 100 users @ 4K |
|-------|-----------|----------------|
| Llama 3 8B | 32 KB | 12.5 GB |
| Llama 3 70B | 320 KB | 125 GB |
| Llama 3 405B | 504 KB | 197 GB |

## Speculative decoding 체크리스트

- Draft model은 target보다 5-10배 작아야 합니다(예: 70B에는 8B draft).
- 의미 있는 speedup을 위해 acceptance rate가 70%를 넘어야 합니다.
- 예측 가능한 text(code, structured output, natural language)에 가장 좋습니다.
- creative하거나 sampling-heavy한 task에서는 가장 나쁩니다(low temperature가 도움).
- 대부분 workload에서는 EAGLE > draft-target > n-gram 순입니다.

## 흔한 실수

- decode를 batch=1로 실행(memory-bound라 GPU compute가 95% idle)
- contiguous KV cache block을 할당(PagedAttention을 써서 waste를 거의 0으로 줄이세요)
- request의 80%가 같은 system prompt를 공유하는데 prefix caching을 무시
- model weight에 GPU memory를 과하게 provision해 KV cache 공간을 남기지 않음
- latency 없이 throughput만 측정(10초 TTFT에서 높은 throughput은 쓸모없음)
- high temperature와 speculative decoding을 함께 사용(acceptance rate가 50% 아래로 떨어짐)

## 모니터링 체크리스트

- Time to first token(TTFT): prefill latency. interactive use에서는 target < 500ms
- Inter-token latency(ITL): decode speed. streaming에서는 target < 50ms
- Throughput(tokens/second): 모든 concurrent user 전체 합계
- KV cache utilization: 할당된 cache 중 사용 중인 비율
- Batch utilization: iteration마다 채워진 batch slot 비율
- Queue depth: batch slot을 기다리는 request 수
