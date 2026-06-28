# Multi-Head Attention

> Attention head 하나는 한 번에 하나의 관계를 학습합니다. Head 여덟 개는 여덟 관계를 학습합니다. Head는 싸게 추가할 수 있습니다. 더 많이 쓰세요.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 minutes

## 문제

Single self-attention head는 attention matrix 하나를 계산합니다. 그 matrix는 한 종류의 관계를 포착합니다. 보통 training signal에서 loss를 최소화하는 관계입니다. Data 안에 subject-verb agreement, co-reference, long-range discourse, syntactic chunking이 모두 얽혀 있다면 single head는 그것들을 하나의 softmax distribution으로 뭉개고 signal의 절반을 잃습니다.

2017년 Vaswani 논문의 해결책은 여러 attention function을 parallel하게 실행하는 것이었습니다. 각 function은 자체 Q, K, V projection을 갖고, output을 concatenate합니다. 각 head는 `d_model / n_heads` dimension의 더 작은 subspace에서 동작합니다. 전체 parameter 수는 같습니다. Expressive power는 올라갑니다.

Multi-head attention은 2026년에 배포되는 모든 transformer의 기본값입니다. 논쟁은 오직 *head를 몇 개 쓸지*, 그리고 key와 value가 projection을 공유할지(Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention)에 관한 것입니다.

## 개념

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Split.** Shape `(N, d_model)`인 `X`를 받습니다. Q, K, V 각각을 `(N, d_model)` shape으로 project합니다. `d_head = d_model / n_heads`인 `(N, n_heads, d_head)`로 reshape합니다. 그런 다음 `(n_heads, N, d_head)`로 transpose합니다.

**Attend in parallel.** 각 head 안에서 scaled dot-product attention을 실행합니다. 각 head는 `(N, d_head)`를 만듭니다. Head들은 embedding의 서로 다른 subspace에서 동작하며, attention computation 자체 중에는 서로 대화하지 않습니다.

**Concatenate and project.** Head를 다시 `(N, d_model)`로 stack하고, shape `(d_model, d_model)`인 learned output matrix `W_o`를 곱합니다. `W_o`는 head들이 섞이는 지점입니다.

**Why it works.** 각 head는 representation budget을 두고 서로 경쟁하지 않고 specialize할 수 있습니다. 2019-2024년의 probing study는 서로 다른 head role을 보여 줍니다. positional head, previous token에 attend하는 head, copy head, named-entity head, induction head(in-context learning의 기반) 등이 있습니다.

**2026년 variant 계보:**

| Variant | Q heads | K/V heads | 사용 모델 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (예: N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | low-rank로 compressed | DeepSeek-V2, V3 |

GQA는 KV-cache memory를 `N/G`배 줄이면서도 거의 전체 quality를 유지하므로 현대의 기본값입니다. MLA는 K/V를 latent space로 compress한 다음 compute time에 다시 project하여 더 나아갑니다. FLOPs를 쓰고 훨씬 더 많은 memory를 절약합니다.

```figure
multihead-split
```

## 직접 만들기

### 1단계: 이미 만든 single-head attention에서 head split하기

Lesson 02의 `SelfAttention`을 가져와 split/concat pair로 감쌉니다. NumPy implementation은 `code/main.py`를 보세요. Logic은 다음과 같습니다.

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Reshape 한 번, transpose 한 번입니다. Loop는 없습니다. 이것이 PyTorch가 `nn.MultiheadAttention` 내부에서 하는 일과 같습니다.

### 2단계: head별 scaled-dot-product attention 실행하기

각 head는 Q, K, V의 자체 slice를 받습니다. Attention은 batched matmul이 됩니다.

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

실제 hardware에서 `Qh @ Kh.transpose(...)`는 하나의 `bmm`입니다. GPU는 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` shape의 단일 batched matmul을 봅니다. Head를 추가하는 비용은 작습니다.

### 3단계: Grouped-Query Attention variant

Key와 value projection만 바뀝니다. Q는 `n_heads` group을 갖고, K와 V는 `n_kv_heads < n_heads` group을 가진 뒤 matching을 위해 repeat됩니다.

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

Inference에서는 `n_heads`가 아니라 `n_kv_heads` copy만 KV cache에 존재하므로 memory를 절약합니다. Llama 3 70B는 64 query heads와 8 KV heads를 사용합니다. cache가 8배 줄어듭니다.

### 4단계: 각 head가 학습한 것 probe하기

4개 head로 짧은 문장에서 MHA를 실행하세요. 각 head마다 `(N, N)` attention matrix를 출력하세요. Random initialization에서도 서로 다른 head가 서로 다른 구조를 고르는 것을 볼 수 있습니다. 일부는 signal이고 일부는 subspace의 rotational symmetry입니다.

## 활용하기

PyTorch의 한 줄 version:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+의 GQA:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Head를 몇 개 써야 할까?** 2026년 production model에서 나온 rule of thumb:

| Model size | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head`는 거의 항상 64 또는 128에 놓입니다. 이것은 head 하나가 얼마나 많이 "볼" 수 있는지의 단위입니다. 32 아래로 내려가면 head들이 scaling factor `sqrt(d_head)`와 싸우기 시작합니다. 256을 넘으면 "many small specialists"의 이점을 잃습니다.

## 산출물

`outputs/skill-mha-configurator.md`를 보세요. 이 skill은 parameter budget, sequence length, deployment target이 주어졌을 때 새 transformer의 head count, kv-head count, projection strategy를 추천합니다.

## 연습문제

1. **쉬움.** `code/main.py`의 MHA를 가져와 `d_model=64`를 고정한 채 `n_heads`를 1에서 16으로 바꾸세요. Synthetic copy task에서 tiny one-layer model의 loss를 plot하세요. Head가 많아지면 도움이 되나요, plateau에 닿나요, 아니면 해가 되나요?
2. **보통.** MQA를 구현하세요. 모든 query head가 하나의 KV head를 공유합니다. Full MHA 대비 parameter count가 얼마나 줄어드는지 측정하세요. N=2048 inference에서 KV-cache size가 얼마나 줄어드는지 계산하세요.
3. **어려움.** Multi-head Latent Attention의 tiny version을 구현하세요. K,V를 rank-`r` latent로 compress하고, latent를 KV cache에 저장한 뒤, attention time에 decompress하세요. 어떤 `r`에서 cache memory가 full MHA의 1/8 아래로 내려가면서 validation ppl이 1 bit 이내로 유지되나요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Head | "단일 attention circuit" | 자체 attention matrix를 가진 `d_head = d_model / n_heads` dimension의 Q/K/V projection 하나. |
| d_head | "Head dimension" | Head별 hidden width. Production에서는 거의 항상 64 또는 128입니다. |
| Split / combine | "Reshape trick" | Attention 앞뒤에서 수행하는 `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose. |
| W_o | "Output projection" | Head를 concatenate한 뒤 적용되는 `(d_model, d_model)` matrix. Head가 섞이는 곳입니다. |
| MQA | "One KV head" | Multi-Query Attention: 단일 shared K/V projection. 가장 작은 KV cache, 일부 quality loss. |
| GQA | "Llama 2 이후의 기본값" | `n_kv_heads < n_heads`인 Grouped-Query Attention. Q에 맞추기 위해 repeat합니다. |
| MLA | "DeepSeek의 trick" | Multi-head Latent Attention: K,V를 low-rank latent로 compress하고 attend time에 decompress합니다. |
| Induction head | "In-context learning 뒤의 circuit" | 이전 occurrence를 detect하고 그 뒤에 온 것을 copy하는 head pair. |

## 더 읽을거리

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 원래 multi-head spec.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 논문.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — training 후 MHA를 GQA로 변환하는 방법.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA와 cache memory에서 MHA/GQA를 이기는 이유.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — head가 실제로 하는 일을 보는 mechanistic 분석.
