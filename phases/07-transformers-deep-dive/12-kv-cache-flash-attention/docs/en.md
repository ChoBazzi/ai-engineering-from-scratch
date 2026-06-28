# KV Cache, Flash Attention & Inference Optimization

> 학습은 병렬적이고 FLOP-bound입니다. 추론은 직렬적이고 memory-bound입니다. 병목이 다르면 트릭도 다릅니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 minutes

## 문제

순진한 자기회귀 디코더는 `N`개 토큰을 생성하기 위해 `O(N²)` 작업을 합니다. 매 step마다 전체 prefix에 대해 attention을 다시 계산하기 때문입니다. 4K-token 응답이라면 attention 연산이 16M개이고, 대부분은 중복입니다. prefix token의 모든 은닉 상태는 한 번 계산되면 결정적입니다. 새 토큰의 query를 이전 모든 토큰의 cached key/value에 대해서만 실행하면 됩니다.

게다가 attention 자체도 많은 데이터를 옮깁니다. 표준 attention은 N×N score matrix, N×d softmax output, N×d final output을 물리적으로 만듭니다. HBM에 대한 read/write가 너무 많습니다. N≥2K에서는 attention이 FLOP-bound가 되기 전에 memory-bound가 됩니다. 고전적 attention kernel은 현대 GPU를 4–10× 정도 제대로 활용하지 못합니다.

Dao et al.에서 나온 두 최적화가 frontier 추론을 "느림"에서 "빠름"으로 밀어 올렸습니다.

1. **KV cache.** 모든 prefix token의 K와 V 벡터를 저장합니다. 각 새 토큰의 attention은 cached key에 대한 하나의 query입니다. 추론은 generation step당 `O(N²)`에서 `O(N)`으로 줄어듭니다.
2. **Flash Attention.** 전체 N×N matrix가 HBM에 닿지 않도록 attention 계산을 tile로 나눕니다. softmax + matmul 전체가 SRAM에서 일어납니다. A100에서 wall-clock 2–4×, FP8을 쓰는 H100에서 5–10× 빨라집니다.

2026년에는 둘 다 보편적입니다. 모든 프로덕션 추론 스택(vLLM, TensorRT-LLM, SGLang, llama.cpp)이 이를 전제합니다. 모든 frontier 모델은 Flash Attention을 켠 상태로 배포됩니다.

## 개념

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 계산

디코더 레이어당, 토큰당, head당:

```text
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

32 layers, 32 heads, d_head=128, fp16인 7B 모델에서는:

```text
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

Llama 3 70B(80 layers, d_head=128, 8 KV heads를 쓰는 GQA)에서는:

```text
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

이 10 GB 때문에 128K context의 Llama 3 70B는 batch size 1에서도 40 GB A100 대부분을 KV cache에만 써야 합니다.

**GQA가 KV-cache 절감의 핵심입니다.** 64 heads를 쓰는 MHA라면 32 GB가 됩니다. MLA는 더 압축합니다.

차원을 움직여 cache size가 어떻게 변하는지 보세요. sequence length나 batch를 올리면 얼마나 빨리 단일 GPU를 넘어서는지 확인할 수 있습니다.

```figure
kv-cache-sizer
```

### Flash Attention — tiling 트릭

표준 attention:

```text
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

HBM 왕복이 세 번입니다. H100에서 HBM bandwidth는 3 TB/s이고 SRAM은 30 TB/s입니다. 모든 HBM 왕복은 on-chip에 머무는 것 대비 10배 slowdown입니다.

Flash Attention:

```text
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

tile당 HBM 왕복은 한 번입니다. 전체 memory footprint는 `O(N²)`에서 `O(N)`으로 줄어듭니다. backward pass는 forward pass의 값을 저장하는 대신 일부 값을 다시 계산합니다. 이것도 memory win입니다.

**수치적 트릭.** running softmax는 tile 전체에 걸쳐 `(max, sum)`을 유지하므로 최종 정규화가 정확합니다. 근사가 아닙니다. Flash Attention은 표준 attention과 bit-identical한 출력을 계산합니다(fp16 non-associativity는 제외).

**버전 변화:**

| 버전 | 연도 | 핵심 변화 | reference hardware에서의 speedup |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | A100에서 2× |
| Flash 2 | 2023 | 더 나은 parallelism, causal-first ordering | A100에서 3× |
| Flash 3 | 2024 | Hopper asynchrony, FP8 | H100에서 1.5–2×(~740 TFLOPs FP16) |
| Flash 4 | 2026 | Blackwell 5-stage pipeline, software exp2 | 추론 우선(초기에는 forward only) |

Flash 4는 출시 시점에는 forward-pass only입니다. 학습은 여전히 Flash 3를 사용합니다. Flash 4의 GQA와 varlen 지원은 보류 중입니다(2026년 중반).

### Speculative decoding — 또 다른 지연 시간 개선

저렴한 모델이 N개 토큰을 제안합니다. 큰 모델은 N개를 병렬로 검증합니다. 검증이 k개 토큰을 받아들이면, k번의 generation을 큰 모델 forward pass 1번으로 처리한 셈입니다. 코드와 prose에서 전형적인 k는 3–5입니다.

2026년 기본값:
- **EAGLE 2 / Medusa.** verifier의 hidden state를 공유하는 통합 draft head입니다. 품질 손실 없이 2–3× 빨라집니다.
- **Speculative decoding with draft model.** 소비자 하드웨어에서 2–4× 빨라집니다.
- **Lookahead decoding.** Jacobi iteration입니다. draft model이 필요 없습니다. niche이지만 공짜입니다.

### Continuous batching

고전적 batched inference는 가장 느린 sequence가 끝날 때까지 기다린 뒤 새 batch를 시작합니다. 짧은 응답이 일찍 끝나면 GPU가 낭비됩니다.

Continuous batching(처음 Orca에서 제공되었고 지금은 vLLM, TensorRT-LLM, SGLang에 있음)은 기존 요청이 끝나는 즉시 새 요청을 batch에 넣습니다. 일반적인 chat workload에서 throughput이 5–10× 향상됩니다.

### PagedAttention — virtual memory로서의 KV cache

vLLM의 대표 기능입니다. KV cache는 16-token block 단위로 할당되고, page table이 logical position을 physical block에 매핑합니다. 병렬 sample(beam search, parallel sampling) 사이에서 KV를 공유하고, prompt caching을 위해 prefix를 hot-swap하며, memory를 defragment할 수 있습니다. 순진한 contiguous allocation보다 throughput이 4× 개선됩니다.

```figure
flash-attention-memory
```

## 직접 만들기

`code/main.py`를 보세요. 다음을 구현합니다.

1. 순진한 `O(N²)` incremental decoder.
2. `O(N)` KV-cached decoder.
3. Flash Attention의 running-max 알고리즘을 시뮬레이션하는 tiled softmax.

### 1단계: KV cache

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

단순합니다. 레이어별, head별 리스트에 토큰별 K, V 벡터를 계속 추가합니다.

### 2단계: tiled softmax

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

한 번에 계산하는 `softmax(qK) V`와 bit-identical한 출력이지만, 어느 순간의 working set은 전체 `N × d_head`가 아니라 `tile × d_head` block입니다.

### 3단계: 100-token generation에서 naive decoding과 cached decoding 비교

attention 연산 수를 셉니다. Naive는 `O(N²)` = 5050입니다. Cached는 `O(N)` = 100입니다. 코드가 둘 다 출력합니다.

## 사용하기

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

프로덕션 vLLM:

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

요청 간 prefix caching은 2026년의 큰 이득입니다. 같은 system prompt, few-shot examples, 긴 context document는 호출 사이에서 KV를 재사용합니다. 반복되는 tool prompt를 가진 agent workload에서는 prefix caching이 일상적으로 throughput을 5× 올립니다.

## 실전 적용

`outputs/skill-inference-optimizer.md`를 보세요. 이 스킬은 새 추론 배포에 맞는 attention implementation, KV cache strategy, quantization, speculative decoding을 고릅니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. naive decoder와 cached decoder가 같은 출력을 만드는지 확인하고, op-count 차이를 보세요.
2. **보통.** prefix caching을 구현하세요. prompt P와 여러 completion이 있을 때 P에 대해 forward pass를 한 번 실행해 KV cache를 채운 뒤 completion별로 branch합니다. completion마다 P를 다시 encoding하는 방식과 speedup을 비교하세요.
3. **어려움.** 장난감 PagedAttention을 구현하세요. KV cache를 고정 16-token block과 free-list로 관리합니다. sequence가 끝나면 block을 pool에 반환합니다. 길이가 다양한 chat completion 1,000개를 시뮬레이션하세요. contiguous allocation과 memory fragmentation을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 보통 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| KV cache | "decoding을 빠르게 만드는 트릭" | 모든 prefix token의 저장된 K와 V입니다. 새 query는 다시 계산하지 않고 여기에 attend합니다. |
| HBM | "GPU main memory" | High Bandwidth Memory입니다. H100은 80 GB, B200은 192 GB입니다. bandwidth는 약 3 TB/s입니다. |
| SRAM | "on-chip memory" | SM별 빠른 memory입니다. H100에서는 SM당 약 256 KB입니다. bandwidth는 약 30 TB/s입니다. |
| Flash Attention | "tiled attention kernel" | HBM에 N×N을 materialize하지 않고 attention을 계산합니다. |
| Continuous batching | "기다리지 않는 batching" | batch를 비우지 않고 끝난 sequence를 빼고 새 sequence를 넣습니다. |
| PagedAttention | "vLLM의 대표 기능" | KV cache를 page table이 있는 고정 block으로 할당합니다. fragmentation을 제거합니다. |
| Prefix caching | "긴 prompt 재사용" | 요청 사이에서 공유 prefix의 KV를 cache합니다. agent에는 큰 비용 절감입니다. |
| Speculative decoding | "draft + verify" | 저렴한 draft model이 token을 제안하고, 큰 model이 k개를 한 번에 검증합니다. |

## 더 읽을거리

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1입니다.
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2입니다.
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3입니다.
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5-stage pipeline과 software-exp2 트릭입니다. 이 lesson이 언급한 forward-only launch caveat는 repo README를 읽으세요.
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 논문입니다.
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding입니다.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — lesson에서 언급한 integrated-draft approach의 EAGLE-1/2 논문입니다.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — EAGLE와 함께 언급한 Medusa 접근입니다.
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 16-token block과 page-table design에 대한 표준 deep dive입니다.
