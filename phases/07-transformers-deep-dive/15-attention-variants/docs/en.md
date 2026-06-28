# Attention 변형 — Sliding Window, Sparse, Differential

> Full attention은 원이다. 모든 토큰이 모든 토큰을 보고, 메모리가 그 대가를 낸다. 네 가지 변형은 원의 모양을 휘게 만들고 비용의 절반을 되찾는다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 minutes

## 문제

Full attention은 시퀀스 길이에 대해 `O(N²)` 메모리와 `O(N²)` 계산을 쓴다. 128K context의 Llama 3 70B라면 layer 하나당 attention entry가 160억 개이고, 여기에 80개 layer를 곱해야 한다. Flash Attention(Lesson 12)은 `O(N²)` activation 메모리를 숨기지만 산술 비용은 바꾸지 않는다. 모든 토큰은 여전히 모든 다른 토큰에 attention한다.

세 종류의 변형은 attention matrix 자체의 topology를 바꾼다.

1. **Sliding window attention (SWA).** 각 토큰은 전체 prefix가 아니라 고정 window 안의 이웃에만 attention한다. 메모리와 계산은 `O(N · W)`로 줄어든다. 여기서 `W`는 window다. Gemma 2/3, Mistral 7B의 첫 layer들, Phi-3-Long.
2. **Sparse / block attention.** 선택된 쌍 `(i, j)`만 점수를 매기고 나머지는 가중치 0으로 강제한다. Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** 별도 Q/K projection으로 attention map 두 개를 계산한 뒤 하나에서 다른 하나를 뺀다. 앞쪽 몇 토큰으로 가중치가 새는 "attention sink"를 제거한다. Microsoft의 DIFF Transformer(2024).

이들은 함께 쓰일 수 있다. 2026년 frontier 모델은 종종 이들을 섞는다. 대부분 layer는 SWA-1024이고, 다섯 번째마다 global full attention이며, retrieval을 정리하는 differential head가 일부 있다. Gemma 3의 5:1 SWA-to-global 비율이 현재 교과서적 기본값이다.

## 개념

### Sliding Window Attention (SWA)

위치 `i`의 각 query는 `[i - W, i]`(causal SWA) 또는 `[i - W/2, i + W/2]`(bidirectional)에 있는 위치에만 attention한다. Window 밖의 토큰은 score matrix에서 `-inf`를 받는다.

```text
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

`N = 8192`, `W = 1024`이면 score matrix에는 기대상 1024 × 8192개의 non-zero row가 있다. 8배 감소다.

**KV cache는 SWA에서 작아진다.** Layer마다 K와 V의 마지막 `W`개 토큰만 보관하면 된다. Gemma-3 비슷한 config(1024 window, 128K context)에서는 KV cache가 128배 줄어든다.

**품질 비용.** SWA-only 트랜스포머는 long-range retrieval에 약하다. 해결책은 SWA layer와 full-attention layer를 교차 배치하는 것이다. Gemma 3는 5:1 SWA:global을 쓴다. Mistral 7B는 causal-SWA stack을 사용했고, 겹치는 window를 통해 정보가 "앞으로 흐른다". 각 layer는 유효 receptive field를 `W`만큼 늘리고, `L`개 layer 뒤에는 모델이 `L × W`토큰 뒤까지 attention할 수 있다.

### Sparse / Block Attention

`N × N` sparsity pattern을 미리 고른다. 세 가지 표준 모양:

- **Local + strided (OpenAI sparse transformer).** 마지막 `W`토큰과 그 이전의 매 `stride`번째 토큰에 attention한다. Local과 long-range를 모두 `O(N · sqrt(N))` 계산으로 포착한다.
- **Longformer / BigBird.** Local window + 모두에게 attention하고 모두가 attention하는 소수의 global token(예: `[CLS]`) + random-sparse link. 같은 품질에서 경험적으로 2배 context.
- **Native Sparse Attention (DeepSeek, 2025).** `(Q, K)`의 어떤 block이 중요한지 학습하고 kernel 수준에서 zero block을 건너뛴다. FlashAttention 호환.

Sparse attention은 kernel engineering 이야기다. 수학은 단순하다(score matrix를 mask한다). 이득은 zero entry를 SRAM에 전혀 올리지 않는 데서 온다. FlashAttention-3와 2026년 FlexAttention API는 PyTorch에서 custom sparse pattern을 first-class로 만든다.

### Differential Attention (DIFF Transformer, 2024)

일반 attention에는 "attention sink" 문제가 있다. Softmax는 모든 row 합을 1로 강제하므로, 특별히 attention할 대상이 없는 토큰은 첫 토큰(또는 처음 몇 토큰)에 가중치를 버린다. 이 때문에 실제 내용에 가야 할 용량이 빼앗긴다.

Differential attention은 **두 개**의 attention map을 계산한 뒤 빼서 이를 고친다.

```text
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

여기서 `λ`는 학습되는 scalar(보통 0.5-0.8)다. A1은 실제 내용 가중치를 포착하고, A2는 sink를 포착한다. 빼기는 sink를 상쇄하고 관련 토큰으로 가중치를 재할당한다.

보고된 결과(Microsoft 2024): perplexity 5-10% 감소, 같은 훈련 길이에서 유효 context 1.5-2배 증가, 더 선명한 needle-in-haystack retrieval.

### 변형 비교

| 변형 | 계산 | KV cache | Full 대비 품질 | Production 사용 |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | layer당 O(N) | baseline | 모든 모델의 기본 layer |
| SWA (window 1024) | O(N·W) | layer당 O(W) | -0.1 ppl, global layer와 함께 좋음 | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | 혼합 | SWA와 비슷 | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) 근사 | 혼합 | 2배 context에서 full과 일치 | 초기 long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | 0.05 ppl 이내 | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5에서 -10% ppl | DIFF Transformer, 2026년 초기 모델 |

```figure
gqa-kv-sharing
```

## 직접 만들기

`code/main.py`를 보라. 장난감 시퀀스에서 full, SWA, local+strided, differential attention을 나란히 보여 주는 causal mask 비교기를 구현한다.

### 1단계: full causal mask (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Lesson 07의 baseline이다. 아래 삼각형이며, 대각선 위에는 가중치가 0이다.

### 2단계: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

매개변수는 하나, `window`다. `window >= n`이면 full causal attention을 되찾는다. `window = 1`이면 각 토큰은 자기 자신에만 attention한다.

### 3단계: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window에 더해 시퀀스 시작까지 매 `stride`번째 토큰을 추가한다. Layer가 더해질수록 receptive field가 로그 단계로 커진다.

### 4단계: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Attention pass 두 번을 수행하고 학습된 mixing coefficient로 뺀다. 코드에서는 single과 differential의 attention-sink heatmap을 비교하고 sink가 무너지는 것을 관찰한다.

### 5단계: KV cache 크기

각 변형에 대해 `N = 131072`에서 layer당 cache 크기를 출력한다. SWA와 sparse 변형은 10-100배 줄인다. Differential은 두 배로 만든다. 메모리 비용을 의식적으로 지불하라.

## 활용하기

2026년 production pattern:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+의 FlexAttention은 mask 함수를 받는다.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

이는 custom Triton kernel로 컴파일된다. 흔한 pattern에서는 FlashAttention-3 속도의 10% 이내에 들어오며, mask 함수는 Python callable이다.

**각각을 고르는 시점:**

- **순수 full attention** — 약 16K context까지의 모든 layer, 또는 retrieval 품질이 가장 중요할 때.
- **SWA + global mix** — 긴 context(>32K), 훈련과 추론이 memory-bound일 때. 32K 위의 2026년 기본값.
- **Sparse block attention** — custom kernel, custom pattern. 특수 워크로드(retrieval, audio)에 남겨 둔다.
- **Differential attention** — attention-sink 오염이 해로운 모든 워크로드(long-context RAG, needle-in-haystack).

## 출시하기

`outputs/skill-attention-variant-picker.md`를 보라. 이 스킬은 목표 context 길이, retrieval 요구, 훈련/추론 계산 프로파일이 주어졌을 때 새 모델의 attention topology를 고른다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. `window=4`의 SWA가 각 row에서 마지막 4토큰 밖의 모든 것을 0으로 만드는지 확인하라. `window=n`이 full causal attention을 bit-identical하게 재현하는지 확인하라.
2. **중간.** Lesson 07 캡스톤 위에 `window=1024` causal SWA를 구현하라. tinyshakespeare에서 1,000스텝 학습하라. Full attention 대비 검증 손실이 얼마나 나빠지는가? Peak memory는 얼마나 줄어드는가?
3. **어려움.** 캡스톤 모델에 Gemma-3 스타일 5:1 layer mix(5 SWA, 1 global)를 구현하라. 파라미터를 맞춘 상태에서 pure-SWA와 pure-global baseline 대비 손실, 메모리, 생성 품질을 비교하라.
4. **어려움.** Head마다 학습되는 `λ`가 있는 differential attention을 구현하라. 합성 retrieval task(needle 하나, distractor 2,000개)에서 학습하라. 같은 파라미터의 single-attention baseline 대비 retrieval accuracy를 측정하라.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | 각 query가 마지막 `W`토큰에 attention한다. KV cache는 `O(W)`로 줄어든다. |
| Effective receptive field | "모델이 얼마나 뒤를 보는가" | Window `W`인 `L`-layer SWA stack에서는 최대 `L × W`토큰. |
| Longformer / BigBird | "Local + global + random" | 항상 attention하는 global token 몇 개가 있는 sparse pattern. 초기 long-context 접근. |
| Native Sparse Attention | "DeepSeek의 kernel trick" | Block-level sparsity를 학습하고, 품질을 유지하면서 kernel 수준에서 zero block을 건너뛴다. |
| Differential attention | "두 map, 하나는 뺀다" | DIFF Transformer: attention sink를 상쇄하기 위해 첫 번째 map에서 두 번째 attention map의 학습된 `λ`배를 뺀다. |
| Attention sink | "가중치가 token 0으로 샌다" | Softmax normalization이 row 합을 1로 강제한다. 정보 없는 query는 position 0에 가중치를 버린다. |
| FlexAttention | "Mask-as-Python" | 임의의 mask 함수를 FlashAttention 모양 kernel로 컴파일하는 PyTorch 2.5+ API. |
| Layer type mix | "5:1 SWA-to-global" | 더 낮은 메모리로 품질을 유지하기 위해 stack 안에서 sparse와 full attention layer를 교차 배치한다. |

## 더 읽을거리

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 표준 sliding-window + global-token 논문.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI의 local+strided pattern.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 SWA:global mix.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 지금 교과서적 기본값이 된 window=1024의 5:1 mix.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 논문.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2의 learned-sparsity attention.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — Use It의 mask-as-callable pattern을 위한 API reference.
