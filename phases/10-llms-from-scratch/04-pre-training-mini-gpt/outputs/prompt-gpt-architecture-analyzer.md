---
name: prompt-gpt-architecture-analyzer
description: GPT-style transformer model의 architecture 선택 분석
version: 1.0.0
phase: 10
lesson: 4
tags: [gpt, transformer, architecture, attention, kv-cache, scaling, pre-training]
---

# GPT Architecture 분석기

technical report, model card, training log에서 GPT-style model을 평가할 때, 이 프레임워크로 architecture를 분해하고 설계 tradeoff를 식별하세요.

## 분석 프로토콜

### 1. Parameter Allocation 분해

각 component의 정확한 parameter count를 계산하세요.

- **Token embeddings**: vocab_size x embed_dim
- **Position embeddings**: max_seq_len x embed_dim
- **Per-block attention**: 4 x embed_dim x embed_dim (Q, K, V, output projections)
- **Per-block FFN**: 2 x embed_dim x ff_dim + embed_dim + ff_dim (two linear layers + biases)
- **Per-block LayerNorm**: 4 x embed_dim (two norms, each with scale + bias)
- **Final LayerNorm**: 2 x embed_dim
- **Output head**: vocab_size x embed_dim (or 0 if weight-tied with token embeddings)

단일 component가 전체 parameter의 40%를 넘으면 표시하세요. 작은 모델에서는 embedding matrix가 지배적입니다. 큰 모델에서는 attention과 FFN이 지배적입니다.

### 2. Attention 설계 분석

attention configuration을 평가하세요.

- **Head dimension**: embed_dim / num_heads. 표준은 64(GPT-2) 또는 128(Llama 3)입니다. 32 미만은 head별 표현력을 제한합니다. 128 초과는 이득이 작고 compute를 낭비합니다.
- **Heads per layer**: head가 많을수록 attention pattern은 다양해지지만 KV cache 메모리가 늘어납니다.
- **Grouped Query Attention (GQA)**: 모델이 여러 Q head에 대해 K/V head를 공유하나요? Llama 3는 32개 Q head에 대해 8개 KV head를 쓰는 GQA를 사용합니다. 이는 KV cache를 4배 줄입니다.
- **Context length**: 최대 position embedding입니다. RoPE는 training length를 넘어 extrapolation을 허용합니다. absolute position embedding은 그렇지 않습니다.

### 3. Memory Budget

모델의 최대 context length에서 inference할 때:

- **Weights (FP16)**: total_params x 2 bytes
- **KV Cache (FP16)**: 2 x num_layers x num_kv_heads x head_dim x max_seq_len x 2 bytes
- **Activations**: batch_size x seq_len x embed_dim x 2 bytes x num_layers (approximate)

KV cache가 weight memory를 초과하면 표시하세요. 이는 long-context model(128K+)에서 발생하며 decode 중 모델이 memory-bound임을 뜻합니다.

### 4. Compute Profile

- **Prefill FLOPS per token**: approximately 2 x total_params (one matmul per parameter, forward pass)
- **Decode FLOPS per token**: same as prefill but on a single token
- **Prefill bottleneck**: compute-bound (GPU TFLOPS)
- **Decode bottleneck**: memory-bound (GPU memory bandwidth)
- **Arithmetic intensity**: FLOPS per byte of memory accessed. Below 100 = memory-bound.

### 5. Scaling 결정

알려진 scaling law에 비추어 평가하세요.

- **Chinchilla optimal**: 주어진 compute budget C에서 최적 모델 크기 N과 token 수 D는 N ~ D(대략 같은 비율의 scaling)를 만족합니다. 7B 모델에는 약 140B token이 필요합니다.
- **Llama 3 overtrained**: Meta는 Llama 3 8B를 15T token으로 학습했습니다(Chinchilla optimal의 100배). 작은 모델을 더 많은 데이터로 overtraining하면 token당 inference cost가 더 좋아집니다.
- **Width vs depth**: 같은 parameter count에서는 일반적으로 더 깊은 모델(layer 수 증가)이 더 넓은 모델(embed_dim 증가)보다 sample-efficient합니다.

## 위험 신호

- **FFN ratio가 4x가 아님**: 표준은 ff_dim = 4 x embed_dim입니다. Llama는 SwiGLU와 함께 8/3 x embed_dim을 사용합니다. 이탈에는 정당한 이유가 있어야 합니다.
- **weight tying 없음**: vocab_size가 embed_dim에 비해 매우 큰 경우가 아니라면 output head는 token embedding과 weight를 공유해야 합니다.
- **13B 초과 모델에 GQA 없음**: grouped-query attention이 없는 13B 초과 모델은 KV cache가 지나치게 커집니다.
- **long context에 RoPE 없음**: absolute position embedding은 training length를 넘어 extrapolation하지 못합니다. 32K+ context를 목표로 하는 모델은 rotary embedding을 사용해야 합니다.
- **model size 대비 learning rate가 너무 높음**: 큰 모델에는 더 낮은 peak learning rate가 필요합니다. GPT-2 Small은 6e-4를 사용합니다. Llama 3 405B는 8e-5를 사용합니다.

## 출력 형식

1. **Parameter Table**: component별 parameter count와 비율
2. **Memory Budget**: 최대 context length에서 weight, KV cache, activation memory
3. **Compute Profile**: A100/H100 기준 prefill과 decode throughput 추정
4. **Design Assessment**: 모델이 잘한 점과 비표준인 점
5. **Scaling Verdict**: 모델 크기가 training data에 적절한지 여부
