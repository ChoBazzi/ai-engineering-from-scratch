# Gradient Checkpointing과 Activation Recomputation

> Backprop은 모든 intermediate activation을 유지합니다. 70B parameters와 128K context에서는 rank당 activation이 3 TB입니다. Checkpointing은 FLOPs를 memory와 맞바꿉니다. 저장하는 대신 recompute합니다. 문제는 어떤 segment를 버릴지이고, 답은 "전부"가 아닙니다.

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 minutes

## 문제

Transformer를 학습하면 backward에서 미분되는 모든 op의 input을 layer마다 저장합니다. attention input, Q/K/V projection, softmax output, FFN input, norm output, residual stream입니다. hidden size `d`, sequence length `L`, batch `B`인 layer에서는 layer당 대략 `12 * B * L * d` floats입니다.

`d=8192, L=8192, B=1`이면 BF16에서 layer당 800 MB입니다. 64-layer model은 activation만 51 GB입니다. microbatch size를 곱하기 전이고, attention-softmax intermediate(head당 `L^2`)를 더하기 전이며, tensor-parallel partial copy를 고려하기 전입니다.

양쪽 비용이 있습니다. BF16 weights와 optimizer state는 80GB에 들어갈 수 있어도 activations가 밀어냅니다. Gradient checkpointing(activation recomputation이라고도 함)이 표준 해법입니다. 대부분의 activation을 버리고, backward 중 forward를 다시 실행해 되찾습니다. 비용은 extra FLOPs입니다. 이점은 checkpoint segment와 total layer의 비율만큼 memory가 줄어드는 것입니다.

naive하게 하면 checkpointing은 step당 forward-pass FLOPs를 대략 33% 더 요구합니다. 잘하면, Korthikanti et al.의 "smart selection"에 따른 selective checkpointing처럼 5% 미만 FLOP overhead로 memory를 5x 절약합니다. FP8 matmul, FSDP offload, expert-parallel MoE까지 있으면 이것은 정말 중요합니다. memory도 wasted compute도 감당할 수 없습니다.

## 개념

### Backward에 실제로 필요한 것

`output = layer(input)`. Backward는 `grad_input`과 `grad_params`를 원합니다. 이를 계산하려면 다음이 필요합니다:

- `input`(linear layer에서 `grad_params = input.T @ grad_output` 계산용)
- 일부 activation derivative intermediate(ReLU/GELU/softmax의 derivative는 activation value에 의존)

forward pass는 이것들을 autograd graph에 자동 저장합니다. 모든 `tensor.retain_grad()`와 input이 필요한 모든 op가 reference를 유지합니다.

### Naive full checkpointing

network를 `N` segment로 나눕니다. forward 중에는 각 segment의 *input*만 저장합니다. backward가 intermediate를 필요로 하면 segment의 forward pass를 다시 실행해 materialize한 뒤 미분합니다.

예: 32-layer transformer를 layer 1개짜리 32 segment로 나눔.

- Memory: 32 layer-inputs(작음) vs 32 * (activation volume per layer)(큼).
- Extra compute: segment당 extra forward 1회, 즉 총 forward FLOPs가 ~33% 증가합니다(backward가 forward의 2x이므로 full step은 1 + 2 = 3 units 대신 1 + 1 + 2 = 4 units).

이것이 original Chen et al. 2016 recipe입니다. memory와 compute 균형을 맞추기 위해 `sqrt(L)` layer마다 checkpoint 하나를 둡니다. L=64라면 checkpoint 8개입니다.

### Selective checkpointing(Korthikanti 2022)

모든 activation의 비용이 같지는 않습니다. attention softmax output은 `B*L*L*heads`이고 sequence length에 대해 *quadratic*으로 증가합니다. FFN hidden activation은 `B*L*4d`이고 linear로 증가합니다. 긴 sequence에서는 softmax가 지배합니다.

Selective checkpointing은 저장 비용이 싼 activation(linear projection, residual)을 유지하고 비싼 것(attention)만 recompute합니다. recompute FLOPs는 최소로 내면서 O(L^2) memory를 절약합니다.

Megatron-Core는 이를 "selective" activation recomputation으로 구현합니다. 2024년 이후 대부분의 frontier training run에서 표준입니다.

### Offload

recompute의 대안은 forward와 backward 사이에 activation을 CPU RAM으로 보내는 것입니다. PCIe bandwidth가 필요합니다. idle bandwidth가 rematerialization cost보다 클 때 유리합니다. 혼합 전략도 흔합니다. 일부 layer는 checkpoint하고 다른 layer는 offload합니다.

FSDP2는 offload를 first-class option으로 제공합니다. GPU가 memory에 병목이지만 CPU-GPU transfer에는 headroom이 있을 때 offload가 빛납니다.

### Recompute cost model

`L` layer 중 매 `k` layer마다 naive checkpointing을 할 때 step당 FLOPs:

```text
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

Selective checkpointing에서는 전체 layer가 아니라 attention kernel만 recompute합니다:

```text
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory savings model

layer당 activation volume: `A`. `L` layer에서는 total activation memory가 `L * A`입니다.

Full checkpoint(segment size 1): `L * input_volume`만 저장합니다(standard transformer에서는 대략 `L * 1/10 A`). 약 `9 * L * A * 1/10`을 절약합니다.

매 `k` layer마다 checkpoint: `L/k * A`에 active segment 안의 `k-1` layer 분량을 더 저장합니다.

`k = sqrt(L)`에서는 memory와 recompute cost가 모두 `sqrt(L)`로 scale합니다. uniform-cost layer에서 optimal tradeoff입니다.

### Checkpoint하지 말아야 할 때

- 이미 in-flight인 pipeline stage의 innermost layer. 어차피 끝나야 합니다.
- stage compute를 지배하는 first와 last layer(Transformer에서는 드묾).
- 이미 FlashAttention을 쓰는 attention kernel. Flash는 이미 softmax를 빠르게 recompute하므로 추가 layer-level checkpointing은 위에 얹는 이점이 작습니다.

### 구현 pattern

1. **Function wrapper:** segment를 `torch.utils.checkpoint.checkpoint(fn, input)`으로 감쌉니다. PyTorch는 `input`만 저장하고 backward에서 나머지를 recompute합니다.

2. **Decorator-based:** layer를 checkpointable로 label합니다. trainer가 config time에 어떤 segment를 wrap할지 결정합니다.

3. **Manual explicit recompute:** backward pass를 직접 작성하고, 저장된 input으로 forward를 복제하는 custom `recompute_forward`를 호출합니다.

세 방식 모두 같은 functional result를 제공합니다. wrapper가 표준 idiom입니다.

### TP / PP / FP8과의 상호작용

- **Tensor parallel:** recompute 시 checkpoint input을 gather하거나 rescatter해야 합니다. communication cost를 다루세요.
- **Pipeline parallel:** 일반적인 pattern은 각 pipeline-stage forward를 checkpoint해 reverse-order microbatch가 activation memory를 재사용하도록 하는 것입니다.
- **FP8 recompute:** recompute 중 업데이트되는 amax history가 original forward와 일치해야 합니다. 그렇지 않으면 FP8 scale이 drift합니다. 대부분 framework는 scale을 snapshot합니다.

## 직접 만들기

### 1단계: segment가 있는 toy model

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 2단계: 모든 activation이 필요한 naive backward

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 3단계: k마다 checkpoint할 때의 memory

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 4단계: cost model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 5단계: memory estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 6단계: optimal segment size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 7단계: selective checkpoint 결정

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 활용하기

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — PyTorch의 canonical wrapper. function을 감싸고 input만 저장하며 backward에서 recompute합니다.
- **Megatron-Core activation recomputation**: `selective`, `full`, `block` mode를 지원합니다. 2024년 이후 frontier training의 표준입니다.
- **FSDP2 offload**: FSDP2의 `module.to_empty(device="cpu")`와 `offload_policy`는 recompute 대신 activation을 CPU로 shard합니다.
- **DeepSpeed ZeRO-Offload**: optimizer state와 activation을 위한 CPU offload로 checkpointing을 보완합니다.

## 산출물

이 lesson은 `outputs/prompt-activation-recompute-policy.md`를 산출합니다. model config(layers, hidden, seq, batch)와 사용 가능한 GPU memory를 받아 per-layer recompute policy(none / selective / full / offload)를 내는 prompt입니다.

## 연습문제

1. correctness를 검증하세요. `model_forward` + `model_backward`(full activations)와 `model_forward_checkpointed` + `model_backward_checkpointed`(segments)를 비교하세요. parameter gradient는 machine precision까지 동일해야 합니다.

2. segment size `k`를 1부터 `L`까지 sweep하세요. FLOP overhead와 memory를 plot하세요. curve의 knee를 찾으세요.

3. selective checkpointing을 구현하세요. attention-module input은 저장하되 intermediate는 저장하지 않습니다. seq=8192인 32-layer model에서 full-layer checkpointing 대비 FLOP overhead를 측정하세요.

4. offload를 추가하세요. segment input을 simulated "CPU buffer"(별도 list)에 저장하세요. "PCIe bandwidth"를 bytes/time으로 측정하고 offload와 recompute 사이의 breakeven point를 찾으세요.

5. 실제 PyTorch transformer를 `torch.utils.checkpoint` 적용 전후로 benchmark하세요. memory(`torch.cuda.max_memory_allocated` 사용)와 step time을 측정하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|----------------------|
| Gradient checkpointing | "Forward를 다시 해 memory를 절약" | segment input만 저장하고 backward 중 intermediate를 recompute해 gradient-support tensor를 얻음 |
| Activation recomputation | "Checkpointing과 같음" | 같은 기법의 HPC-flavored name |
| Segment size (k) | "Checkpoint당 layer 수" | intermediate를 버리고 함께 rematerialize하는 layer 수 |
| Selective checkpointing | "Korthikanti's trick" | 저장 비용이 큰 activation(attention softmax)만 recompute하고 싼 것은 유지 |
| Full checkpointing | "Naive version" | 모든 segment에서 모든 layer intermediate를 recompute |
| Block checkpointing | "Coarse-grained" | 전체 transformer block을 checkpoint함; 가장 큰 granularity |
| FLOP overhead | "Compute tax" | step당 extra FLOPs = (recompute FLOPs) / (fwd + bwd FLOPs); naive 33%, selective 5% |
| Activation offload | "CPU로 보냄" | forward->backward 사이 activation을 CPU RAM으로 이동; recompute의 대안 |
| sqrt-L rule | "Classical optimum" | uniform-cost layer에서 optimal checkpoint spacing은 sqrt(L) layers |
| Attention-softmax volume | "O(L^2) problem" | L^2 * heads * batch floats; long context에서 activation memory를 지배 |

## 더 읽을거리

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- gradient checkpointing을 formalize한 original paper
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation과 formal cost analysis
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- reverse-mode rematerialization을 통한 alternative constant-memory approach
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- scale에서의 activation offload
