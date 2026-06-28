# Native Sparse Attention(DeepSeek NSA)

> 64k token에서는 attention이 decode latency의 70-80%를 먹습니다. 모든 open-model lab은 이를 고칠 계획을 갖고 있습니다. DeepSeek의 NSA(ACL 2025 best paper)는 살아남은 접근입니다. 세 개의 parallel attention branch, 즉 compressed coarse-grained tokens, selectively retained fine-grained tokens, local context용 sliding windows를 learned gate로 결합합니다. Hardware-aligned(kernel-friendly), natively trainable(pre-training에서 작동하고 inference에 덧붙인 것이 아님)하며, 64k decode에서 full attention quality를 맞추거나 넘기면서 FlashAttention보다 빠르게 실행됩니다. 이 lesson은 세 branch를 end-to-end로 만들고 sparsity가 end-to-end differentiable한 이유를 보여 줍니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 minutes

## 학습 목표

- NSA의 세 attention branch와 각 branch가 포착하는 것을 말하기.
- 이전 sparse-attention 방법이 inference-only였던 반면 NSA가 왜 "natively trainable"한지 설명하기.
- Compression block size와 selection top-k의 함수로 64k context에서 full attention 대비 NSA의 attention compute savings 계산하기.
- 짧은 synthetic sequence에서 stdlib Python으로 three-branch combination을 구현하고 gating weights가 타당하게 동작하는지 검증하기.

## 문제

Sequence length N에서 full attention은 `O(N^2)` time과 layer당 `O(N)` KV cache가 듭니다. 64k token에서는 compute와 memory bandwidth 수치가 치명적입니다. NSA paper의 theoretical estimate에 따르면 64k에서 attention은 total decode latency의 70-80%를 차지합니다. TTFT, tokens/sec, cost per million tokens 등 downstream의 모든 것이 attention cost에 지배됩니다.

Sparse attention은 분명한 답입니다. 이전 시도는 두 bucket으로 나뉩니다. Fixed-pattern sparsity(sliding-window, strided, block-local)는 정보를 버리고 long-range recall task에서 실패합니다. Inference-time sparsity(KV cache pruning, H2O, StreamingLLM)는 dense attention으로 pre-trained된 model에 적용되므로 potential speedup의 일부만 회복합니다. Model이 sparse pattern을 통해 정보를 route하도록 훈련된 적이 없기 때문입니다.

Native Sparse Attention(Yuan et al., DeepSeek + PKU + UW, ACL 2025 best paper, arXiv:2502.11089)은 둘 다 합니다. Model이 pre-training 중 학습하는 sparsity pattern이면서, inference에서 실제 compute savings를 제공하는 kernel-aligned algorithm으로 구현됩니다. 2년 뒤에는 NSA 또는 직접 후속이 모든 frontier long-context model의 default attention이 될 가능성이 큽니다.

## 개념

### 세 개의 parallel branch

각 query에 대해 NSA는 KV cache의 세 가지 view를 대상으로 attention을 세 번 실행합니다.

1. **Compressed branch.** Token을 block size `l`(보통 32 또는 64)로 묶습니다. 각 block은 작은 learned MLP로 single summary token으로 압축됩니다. Query는 이 compressed token들에 attend하여 전체 sequence에 대한 coarse-grained view를 얻습니다.

2. **Selected branch.** Compressed branch의 attention score를 사용해 현재 query와 가장 관련 있는 top-k block을 식별합니다. 그 block들의 fine-grained(uncompressed) token을 읽고 query가 모두에 attend합니다. Compressed-branch attention을 selection의 routing signal로 생각하면 됩니다.

3. **Sliding-window branch.** Query는 가장 최근 `W` token(보통 512)에 attend하여 local context를 봅니다. 이 branch는 다른 두 branch가 놓칠 수 있는 structure-heavy short-range pattern(syntax, local coreference)을 잡습니다.

세 branch output은 learned per-position gate로 결합됩니다.

```text
out = g_cmp * out_cmp + g_sel * out_sel + g_win * out_win
```

`g_cmp, g_sel, g_win`은 query에 대한 작은 MLP에서 나온 gate weights입니다. 합이 1일 필요는 없습니다. Branch를 독립적으로 weight할 수 있습니다.

### 이것이 "natively trainable"한 이유

Selection step(top-k blocks)은 discrete입니다. Discrete operation은 gradient flow를 끊습니다. 이전 sparse-attention 작업은 selection을 통한 backprop을 건너뛰거나(training 제한), inference에서 실제 sparsity를 주지 않는 continuous relaxation을 사용했습니다.

NSA는 이를 우회합니다. Compressed-branch attention 자체가 전체 sequence에 대한 differentiable coarse-grained attention입니다. Top-k operation은 compressed branch의 top attention score를 재사용해 어떤 fine-grained block을 load할지 고를 뿐입니다. Gradient는 compressed-branch score를 통해 흐릅니다(그 score는 compressed output과 selection logic 둘 다에 영향을 줍니다). Selected block이 final output에 기여하는 부분도 differentiable합니다. Non-differentiable `top_k` operation은 forward computational graph에서 no-op입니다. Memory에서 어떤 block을 load할지만 제어합니다.

이것이 NSA를 pre-training에서 end to end로 사용할 수 있는 이유입니다. Model은 세 branch를 통해 정보를 jointly route하는 법을 배우고, inference에서는 약속한 speedup을 실제로 제공하는 sparse pattern을 만들어 냅니다.

### Hardware-aligned kernel

NSA의 kernel은 현대 GPU memory hierarchy에 맞게 설계되었습니다. Kernel은 GQA group 단위로 query를 load하고(outer loop), group별 sparse KV block을 가져온 뒤(inner loop), SRAM에서 attention을 실행합니다. 각 query group이 같은 selected block을 보므로(selection은 query-head별이 아니라 query-group별), KV load가 group 전반에 amortize됩니다. Arithmetic intensity가 높게 유지됩니다.

논문은 Triton kernel이 64k decode에서 FlashAttention보다 9x 빠르게 실행되며, sequence length가 커질수록 speedup ratio도 커진다고 보고합니다. Forward와 backward kernel이 모두 제공됩니다.

### Compute budget

`N`을 sequence length, `l`을 compression block size, `k`를 top-k selection count, `w`를 sliding window, `b`를 selected block size(보통 `l`과 같음)라고 합시다.

- Compressed branch: query당 `O(N/l)` keys, 따라서 total `O(N * N / l)`.
- Selected branch: query당 `O(k * b)` keys, 따라서 `O(N * k * b)`.
- Sliding branch: query당 `O(w)` keys, 따라서 `O(N * w)`.

Total: `O(N * (N/l + k*b + w))`.

`N = 64k, l = 64, k = 16, b = 64, w = 512`이면 query당 cost는 `1000 + 1024 + 512 = 2536 keys`입니다. Full attention은 `64000 keys`입니다. 25x compute reduction입니다.

`N = 128k, l = 64, k = 16, b = 64, w = 512`이면 query당 cost는 `2000 + 1024 + 512 = 3536 keys`입니다. Full attention은 `128000 keys`입니다. 36x reduction입니다. 이득은 sequence length와 함께 커집니다. 그것이 핵심입니다.

### 다른 방법과의 비교

| 방법 | 미분 가능 여부 | 실제 inference speedup | Long-range recall |
|--------|---------------|----------------------|-------------------|
| Sliding window only | yes | yes | fails |
| Strided / block-sparse | yes | yes | partial |
| KV pruning (H2O, StreamingLLM) | N/A (inference-time) | yes | partial |
| MoBA (Moonshot) | partial | yes | good |
| NSA | yes (natively) | yes (9x at 64k) | matches full attention |

MoBA(Moonshot, arXiv:2502.13189)는 동시에 발표되었으며, attention block에 MoE principle을 적용하는 비슷한 three-is-better-than-one 접근을 취합니다. NSA와 MoBA는 2026 long-context pre-training을 위해 알아야 할 두 architecture입니다.

```figure
sliding-window-attention
```

## 직접 만들기

`code/main.py`는 짧은 synthetic sequence에서 세 branch를 구현하고 다음을 보여 줍니다.

- Compression MLP(pedagogical clarity를 위해 simple mean-pool baseline 사용; 실제 NSA는 learned MLP 사용).
- Compressed-branch score가 구동하는 top-k block selection.
- 마지막 `w` token에 대한 sliding-window attention.
- Gated combination.
- Full attention과 비교하는 compute-count printout.

### 1단계: token을 block으로 압축하기

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 2단계: compressed-branch attention

Query를 compressed keys에 대해 softmax attention으로 실행합니다. Compressed-branch score는 top-k selection의 signal로도 쓰입니다.

### 3단계: top-k block selection

가장 높은 score를 가진 compressed block `k`개의 index를 고릅니다. 해당 block의 original uncompressed token을 load하고 그 위에서 attention을 실행합니다.

### 4단계: sliding-window attention

마지막 `w` token을 가져와 standard attention을 실행합니다.

### 5단계: gate와 combine

Query에 대한 작은 MLP가 세 gate weight를 생성합니다. Final output은 세 branch output의 weighted sum입니다.

### 6단계: compute counting

각 branch와 total에 대해 query당 attend한 key 수를 출력합니다. `N`(full attention)과 비교하세요. `l = 32, k = 4, w = 128`인 1024-token synthetic에서 NSA는 query당 `32 + 128 + 128 = 288` keys를 봅니다. Full attention은 1024입니다. 3.5x fewer입니다.

## 활용하기

NSA는 DeepSeek 자체 long-context pre-training pipeline에 ship되어 있습니다. 2026년 4월 기준 public inference stack의 integration status:

- **DeepSeek internal**: native, published weights는 NSA 또는 successor DSA(Deepseek Sparse Attention)를 사용합니다.
- **vLLM**: DeepSeek-V3.x weights용 experimental NSA support가 개발 중입니다.
- **SGLang**: NSA benchmark가 공개되었고 production path는 vLLM을 따릅니다.
- **llama.cpp / CPU**: 지원하지 않습니다. Kernel decomposition overhead가 CPU throughput에서는 가치가 없습니다.

NSA를 쓸 때:

- 64k-plus context를 목표로 하는 pre-training 또는 continued-training run이고 serious compute budget이 있는 경우.
- DeepSeek 자체 long-context checkpoint의 inference. Weights가 NSA-native입니다.

쓰지 않을 때:

- 기존 dense-attention pre-trained model을 serving하는 경우. Continued training 없이 NSA를 retrofit할 수 없습니다.
- Context가 16k 미만인 경우. Three-branch overhead가 savings를 압도합니다.
- Batch-1 interactive chat. Latency-sensitive decode가 이득을 보지만 long context에서만 그렇습니다.

## 산출물

이 lesson은 `outputs/skill-nsa-integrator.md`를 생성합니다. Long-context pre-training run specification이 주어지면 compression block size, top-k, sliding window, gate MLP width, kernel choice, architecture change를 정당화할 long-context eval을 포함한 NSA integration plan을 만듭니다.

## 연습문제

1. 1024-token synthetic에서 `code/main.py`를 실행하세요. `(l, k, w)`를 세 preset에 걸쳐 sweep하고 compute count를 출력하세요. Needle-in-haystack test에서 full attention 대비 95% recall을 유지하면서 query당 key-count가 가장 낮은 preset을 식별하세요.

2. Mean-pool compressor를 tiny learned MLP(2-layer, hidden 32)로 교체하세요. Signal이 block 평균인 synthetic task에서 훈련하세요. Held-out data에서 mean-pool baseline 대비 perplexity gap을 측정하세요.

3. Gate MLP를 구현하세요. Query를 input으로 받아 scalar 세 개를 output합니다. Gate가 타당하게 동작함을 보이세요. Random query에서는 near-uniform weighting, query가 멀리 있는 block에 hit할 때는 selected branch에 heavy weight가 나와야 합니다.

4. NSA-enabled 70B model의 128k context KV cache memory budget을 계산하세요. KV heads는 8, head dim은 128, BF16입니다. Full attention 및 MLA(Phase 10 · 14에서 MLA 수치를 보았습니다)와 비교하세요. NSA의 fine-grained branch KV cache가 full attention과 같아지는 sequence length를 식별하세요.

5. NSA paper(arXiv:2502.11089)의 Section 4를 읽고 compressed branch의 attention score를 별도 routing score 계산 대신 top-k selection에 재사용하는 이유를 세 문장으로 설명하세요. 답을 gradient flow와 연결하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| Compressed branch | "Coarse view" | Block-averaged key에 대한 attention입니다. Query당 O(N/l) keys로 global context를 제공합니다 |
| Selected branch | "Top-k blocks" | Compressed-branch score가 가장 높은 `k` block에 대한 fine-grained attention입니다 |
| Sliding window | "Local context" | Short-range pattern을 위해 마지막 `W` token에 attend합니다 |
| Native trainability | "sparsity를 켠 채 pre-train" | Sparsity pattern이 inference에 덧붙는 것이 아니라 pre-training 중 학습됩니다 |
| Compression block size l | "Coarse view의 group size" | 하나의 summary로 병합되는 token 수입니다. 보통 32-64입니다 |
| Top-k | "유지할 block 수" | Uncompressed token을 읽을 compressed block 수입니다. 보통 16입니다 |
| Sliding window W | "Local attention radius" | 보통 512입니다. 더 짧으면 local coherence가 나빠지고, 더 길면 compute가 낭비됩니다 |
| Branch gate | "세 branch를 섞는 방법" | 세 branch contribution을 weight하는 per-position MLP output입니다 |
| Hardware alignment | "Kernel-friendly sparsity" | 실제 GPU kernel이 theoretical speedup을 달성하도록 sparse pattern을 고릅니다 |
| DSA | "NSA's successor" | DeepSeek lineage에서 NSA 뒤에 나온 Deepseek Sparse Attention입니다 |

## 더 읽을거리

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089) — the paper
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — NSA가 겨냥하는 architecture family
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189) — concurrent work, block에 대한 MoE-style attention
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150) — sliding-window origins
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453) — NSA가 개선하는 inference-time sparsity baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691) — NSA kernel이 64k에서 이기는 full-attention baseline
