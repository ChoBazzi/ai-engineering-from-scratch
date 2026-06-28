# Positional Encoding — Sinusoidal, RoPE, ALiBi

> Attention은 permutation-invariant입니다. Positional signal이 없으면 "The cat sat on the mat"와 "mat the on sat cat the"는 같은 output을 냅니다. 세 algorithm이 이 문제를 고칩니다. 각각은 "position"이 무엇을 의미하는지에 대해 서로 다른 선택을 합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 minutes

## 문제

Scaled dot-product attention은 order-blind입니다. Attention matrix `softmax(Q K^T / √d) V`는 pairwise similarity에서 계산됩니다. `X`의 row를 섞으면 output의 row도 같은 방식으로 섞입니다. Attention 내부에는 position을 신경 쓰는 것이 없습니다.

Bag-of-words model이라면 이것은 bug가 아닙니다. 하지만 language, code, audio, video처럼 order가 의미를 담는 모든 것에서는 치명적입니다.

해결책은 어떤 방식으로든 position을 embedding에 주입하는 것입니다. 세 시대의 답이 있습니다.

1. **Absolute sinusoidal** (Vaswani 2017). Position의 `sin/cos`를 embedding에 더합니다. 단순하고, learned parameter가 없지만, trained length 밖으로 extrapolate하기 어렵습니다.
2. **RoPE — Rotary Position Embeddings** (Su 2021). Q와 K vector를 position에 비례한 angle만큼 rotate합니다. Dot product 안에 *relative* position을 직접 encode합니다. 2026년의 지배적 방식입니다.
3. **ALiBi — Attention with Linear Biases** (Press 2022). Embedding을 완전히 건너뛰고, distance에 따라 attention score에 head별 linear penalty를 더합니다. Length extrapolation이 뛰어납니다.

2026년 기준 거의 모든 frontier open model은 RoPE를 사용합니다. Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi가 그렇습니다. Long-context model 일부는 ALiBi 또는 그 현대적 variant를 씁니다. Absolute sinusoidal은 역사적 방식입니다.

## 개념

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

Shape `(max_len, d_model)`의 fixed matrix `PE`를 미리 계산합니다.

```text
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

그런 다음 attention 전에 `X' = X + PE[:N]`을 계산합니다. 각 dimension은 서로 다른 frequency의 sinusoid입니다. Model은 phase pattern에서 position을 읽는 법을 학습합니다. `max_len`을 넘으면 실패합니다. training 중 position 0-2047만 봤다면 position 2048에서 무슨 일이 생기는지 model에게 알려 준 적이 없습니다.

### RoPE

Embedding이 아니라 Q와 K vector를 rotate합니다. Dimension pair `(2i, 2i+1)`에 대해:

```text
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

Position `pos_k`의 key에도 같은 rotation을 적용합니다. Dot product `q'_m · k'_n`은 `(m - n)`만의 function이 됩니다. 즉, rotation은 absolute position에 묶여 있었지만 **attention score는 relative distance에만 의존합니다**. 아름다운 trick입니다.

RoPE 확장: retraining 없이 더 긴 context로 extrapolate하기 위해 `base`를 scale할 수 있습니다(NTK-aware, YaRN, LongRoPE). Llama 3는 이런 방식으로 8K에서 128K context로 확장되었습니다.

### ALiBi

Embedding trick을 건너뜁니다. Attention score에 직접 bias를 줍니다.

```text
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

여기서 `m_h`는 head-specific slope입니다. 예: `1 / 2^(8·h/H)`. 가까운 token은 boost되고 먼 token은 penalize됩니다. Training-time cost는 없습니다. 논문은 length extrapolation이 sinusoidal을 이기고 원래 trained length에서는 RoPE와 맞먹는다고 보입니다.

### 2026년에 무엇을 고를까

| Variant | Extrapolation | Training cost | 사용 모델 |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | 나쁨 | free | original transformer, early BERT |
| Learned absolute | 없음 | 작음 | GPT-2, GPT-3 |
| RoPE | scaling하면 좋음 | free | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 뛰어남 | fine-tune stage | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 뛰어남 | free | BLOOM, MPT, Baichuan |

RoPE가 이긴 이유는 architecture를 바꾸지 않고 attention 안에 들어가며, relative position을 encode하고, `base` hyperparameter가 long-context fine-tuning을 위한 깔끔한 knob를 제공하기 때문입니다.

```figure
rope-explorer
```

## 직접 만들기

### 1단계: sinusoidal encoding

`code/main.py`를 보세요. 4줄짜리 computation입니다.

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

첫 attention layer 전에 이를 embedding matrix에 더합니다.

### 2단계: Q, K에 RoPE 적용하기

RoPE는 Q와 K에 in-place로 동작합니다. 각 dimension pair에 대해:

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

중요한 점은 position `m`의 Q와 position `n`의 K에 같은 function을 적용하는 것입니다. 둘의 dot product는 모든 coordinate pair에서 `cos((m-n)·θ_i)` factor를 얻습니다. Attention은 공짜로 relative position을 학습합니다.

### 3단계: ALiBi slope와 bias

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

Head `h`의 `(seq_len, seq_len)` attention score matrix에 `bias[h]`를 더한 다음 softmax를 적용합니다.

### 4단계: RoPE의 relative-distance property 검증하기

Random vector `a, b`를 두 개 고릅니다. `(pos_a, pos_b)`만큼 rotate합니다. 그런 다음 `(pos_a + k, pos_b + k)`만큼 rotate합니다. 두 dot product는 floating-point error 안에서 같아야 합니다. 이 property가 RoPE의 핵심입니다. Absolute offset에는 invariant하고, relative gap만 중요합니다.

## 활용하기

PyTorch 2.5+는 `torch.nn.functional`에 RoPE utility를 제공합니다. 대부분의 production code는 attention kernel 내부에서 RoPE를 적용하는 `flash_attn` 또는 `xformers`를 사용합니다.

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026년 long-context trick:**

- **NTK-aware interpolation.** 4K에서 16K+로 확장할 때 `base`를 `base * (scale_factor)^(d/(d-2))`로 rescale합니다.
- **YaRN.** Long context에서 attention entropy를 보존하는 더 똑똑한 interpolation입니다. Llama 3.1 128K가 사용합니다.
- **LongRoPE.** Per-dimension scale factor를 고르기 위해 evolutionary search를 사용하는 Microsoft의 2024년 방법입니다. Phi-3-Long이 사용합니다.
- **Position interpolation + fine-tuning.** Position을 extension factor만큼 줄이고 1-5B token으로 fine-tune합니다. 놀랄 만큼 효과적입니다.

## 산출물

`outputs/skill-positional-encoding-picker.md`를 보세요. 이 skill은 target context length, extrapolation needs, training budget이 주어졌을 때 새 model의 encoding strategy를 고릅니다.

## 연습문제

1. **쉬움.** `max_len=512, d=128`에서 sinusoidal `PE` matrix를 heatmap으로 plot하세요. "dimension index가 커질수록 stripe가 넓어진다"는 pattern을 확인하세요.
2. **보통.** NTK-aware RoPE scaling을 구현하세요. Tiny LM을 length 256 sequence로 train한 뒤 scaling이 있을 때와 없을 때 length 1024에서 test하세요. Perplexity를 측정하세요.
3. **어려움.** 같은 attention module 안에 ALiBi와 RoPE를 구현하세요. Length 512 sequence의 copy task에서 4-layer transformer를 train하세요. Test time에 2048로 extrapolate하세요. Degradation을 비교하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Positional encoding | "Attention에 order를 알려 준다" | Position을 encode하기 위해 embedding 또는 attention에 더하는 모든 signal. |
| Sinusoidal | "원래 방식" | Geometric frequency의 `sin/cos`를 embedding에 더하는 방식. Extrapolate하지 못합니다. |
| RoPE | "Rotary embeddings" | Q, K를 position-dependent angle로 rotate합니다. Dot product가 relative distance를 encode합니다. |
| ALiBi | "Linear bias trick" | Attention score에 `-m·\|i-j\|`를 더합니다. Embedding이 필요 없고 extrapolation이 뛰어납니다. |
| base | "RoPE의 knob" | RoPE의 frequency scaler. Inference에서 context를 늘리려면 증가시킵니다. |
| NTK-aware | "RoPE scaling trick" | Context가 늘어날 때 high-frequency dimension이 눌리지 않도록 `base`를 rescale합니다. |
| YaRN | "세련된 방식" | Attention entropy를 보존하는 per-dimension interpolation+extrapolation. |
| Extrapolation | "Trained length 밖에서도 작동한다" | Training에서 본 `max_len`을 지나서도 position scheme이 올바른 output을 낼 수 있는지. |

## 더 읽을거리

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — original sinusoidal.
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 논문.
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi.
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — 최신 RoPE scaling.
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta의 Llama 2 long-context 논문.
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long에서 사용되고 Use It section에서 언급한 Microsoft 방식.
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 모든 RoPE scaling scheme(default, linear, dynamic, YaRN, LongRoPE, Llama-3)의 production-grade implementation.
