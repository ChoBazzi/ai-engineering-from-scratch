# 전체 Transformer — 인코더 + 디코더

> Attention은 주역이다. 그 밖의 모든 것, 즉 residual, normalization, feed-forward, cross-attention은 그것을 깊게 쌓을 수 있게 해 주는 발판이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## 문제

attention layer 하나는 특징 추출기이지, 모델이 아니다. layer마다 matmul 하나만으로는 언어를 다룰 충분한 용량이 없다. 깊이가 필요하다. 그리고 올바른 배관이 없으면 깊이는 무너진다.

2017년 Vaswani 논문은 attention layer 하나를 쌓을 수 있는 block으로 바꾼 여섯 가지 설계 결정을 묶어 냈다. 이후 모든 transformer, 즉 encoder-only(BERT), decoder-only(GPT), encoder-decoder(T5)는 같은 골격을 물려받았다. 2026년에는 block이 더 다듬어졌지만(RMSNorm, SwiGLU, pre-norm, RoPE), 골격은 동일하다.

이 lesson은 그 골격을 다룬다. 다음 lesson들은 이를 특화한다. 06은 encoder, 07은 decoder, 08은 encoder-decoder를 다룬다.

## 개념

![연결된 encoder와 decoder block 내부 구조](../assets/full-transformer.svg)

### 여섯 가지 구성 요소

1. **Embedding + positional signal.** token을 vector로 바꾼다. 위치는 RoPE(현대적 방식) 또는 sinusoidal(고전적 방식)로 주입한다.
2. **Self-attention.** 모든 position이 다른 모든 position을 본다. decoder에서는 mask를 적용한다.
3. **Feed-forward network (FFN).** position별 two-layer MLP: `W_2 · activation(W_1 · x)`. 기본 expansion ratio는 4×다.
4. **Residual connection.** `x + sublayer(x)`. 이것이 없으면 약 6개 layer를 지나 gradient가 사라진다.
5. **Layer normalization.** `LayerNorm` 또는 `RMSNorm`(현대적 방식). residual stream을 안정화한다.
6. **Cross-attention (decoder only).** query는 decoder에서 오고, key와 value는 encoder output에서 온다.

vector 하나가 block을 통과하는 흐름을 보자. attention은 position 사이를 섞고, residual은 정보를 앞으로 운반하며, FFN은 정보를 변환하고, norm은 stream을 안정적으로 유지한다.

```figure
transformer-block
```

### Encoder block (BERT, T5 encoder에서 사용)

```text
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

encoder는 양방향이다. masking이 없다. 모든 position이 모든 position을 볼 수 있다.

### Decoder block (GPT, T5 decoder에서 사용)

```text
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

decoder는 block마다 세 개의 sublayer를 가진다. 가운데 sublayer인 cross-attention은 encoder에서 decoder로 정보가 흐르는 유일한 지점이다. GPT 같은 순수 decoder-only architecture에서는 cross-attention을 생략하고 masked self-attention + FFN만 둔다.

### Pre-norm vs post-norm

원 논문은 `x + sublayer(LN(x))`와 `LN(x + sublayer(x))`를 다뤘다. post-norm은 2019년 무렵부터 선호도가 떨어졌다. 세심한 warmup 없이는 깊게 학습하기 어렵기 때문이다. pre-norm(sublayer *앞*의 `LN`)은 2026년 기본값이다. Llama, Qwen, GPT-3+, Mistral이 모두 이를 사용한다.

### 2026년식 현대화 block

Vaswani 2017은 LayerNorm + ReLU를 사용했다. 현대 stack은 둘 다 교체했다. production block의 실제 모습은 다음과 같다.

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU는 matrix 세 개를 사용하므로 총 parameter 수가 맞는다) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (또는 MLA) |
| Bias terms | Yes | No |

RMSNorm은 LayerNorm의 mean-centering을 제거한다(뺄셈 하나가 줄어든다). 그만큼 compute를 아끼며, 실험적으로 적어도 같은 수준으로 안정적이다. SwiGLU(`Swish(W1 x) ⊙ W3 x`)는 Llama, PaLM, Qwen 논문에서 ReLU/GELU FFN보다 LM perplexity를 꾸준히 약 0.5 point 개선했다.

### 파라미터 수

`d_model = d`, FFN expansion `r`인 block 하나에 대해:

- MHA: `4 · d²` (Q, K, V, O projection)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: 무시할 수 있을 만큼 작음

`d = 4096, r = 2.6, layers = 32`(대략 Llama 3 8B)에 대해 총합은 `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B`다(embedding과 head 제외). 공개된 count와 맞아떨어진다.

## 직접 만들기

### 1단계: building block

Lesson 03의 작은 `Matrix` class를 사용한다(독립 실행을 위해 이 파일에 복사해 두었다).

- `layer_norm(x, eps=1e-5)` — mean을 빼고 std로 나눈다.
- `rms_norm(x, eps=1e-6)` — RMS로 나눈다. mean subtraction은 없다.
- `gelu(x)`와 `silu(x) * W3 x`(SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)`와 `decoder_block(x, enc_out, params)`.

전체 wiring은 `code/main.py`를 보라.

### 2단계: 2-layer encoder와 2-layer decoder 연결

쌓아 올린다. encoder output을 모든 decoder cross-attention에 전달한다. output projection 앞에 final LN을 추가한다.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 3단계: toy example으로 forward 실행

6-token source와 5-token target을 통과시킨다. output shape이 `(5, vocab)`인지 확인한다. training은 하지 않는다. 이 lesson은 loss가 아니라 architecture를 다룬다.

### 4단계: RMSNorm + SwiGLU로 교체

LayerNorm과 ReLU-FFN을 RMSNorm과 SwiGLU로 바꾼다. shape이 여전히 맞는지 확인한다. 함수 하나를 바꾸는 것만으로 2026년식 현대화가 된다.

## 활용하기

PyTorch/TF reference implementation은 `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`다. 하지만 2026년 production code 대부분은 자체 block을 만든다. 이유는 다음과 같다.

- Flash Attention은 `nn.MultiheadAttention`을 통하지 않고 attention 내부에서 호출된다.
- GQA / MLA는 stdlib reference에 없다.
- RoPE, RMSNorm, SwiGLU는 PyTorch 기본값이 아니다.

HF `transformers`에는 읽어 볼 만한 깔끔한 reference block이 있다. `modeling_llama.py`는 2026년 decoder-only block의 canonical 구현이다. 약 500줄이며 한 번은 따라 읽을 가치가 있다.

**Encoder vs decoder vs encoder-decoder — 언제 무엇을 고를까:**

| 필요 | 선택 | 예시 |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

decoder-only는 가장 깔끔하게 scale되고 이해와 생성을 모두 처리하기 때문에 language에서 우세해졌다. encoder-decoder는 input에 명확한 "source sequence" 정체성이 있을 때(translation, speech recognition, structured tasks) 여전히 가장 좋다.

## 실전 적용

`outputs/skill-transformer-block-reviewer.md`를 보라. 이 skill은 새 transformer block 구현을 2026년 기본값과 비교해 검토하고, 빠진 요소(pre-norm, RoPE, RMSNorm, GQA, FFN expansion ratio)를 표시한다.

## 연습문제

1. **쉬움.** `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`에서 `encoder_block`의 parameter 수를 세라. block을 구현하고 `sum(p.numel() for p in block.parameters())`로 검증하라.
2. **보통.** post-norm에서 pre-norm으로 바꿔라. 둘을 초기화한 뒤 random input에서 12개 layer를 쌓은 후 activation norm을 측정하라. post-norm의 activation은 폭주하고, pre-norm은 bounded 상태를 유지해야 한다.
3. **어려움.** toy copy task(`x`를 뒤집어 복사)에 대해 4-layer encoder-decoder를 구현하라. 100 step 학습하고 loss를 보고하라. RMSNorm + SwiGLU + RoPE로 바꾸면 loss가 내려가는가?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Block | "Transformer layer 하나" | norm + attention + norm + FFN을 residual connection으로 감싼 stack. |
| Residual | "Skip connection" | `x + f(x)` output. 깊은 stack에서 gradient 흐름을 가능하게 한다. |
| Pre-norm | "뒤가 아니라 앞에서 normalize" | 현대적 방식: `x + sublayer(LN(x))`. warmup 묘기 없이 더 깊게 학습한다. |
| RMSNorm | "mean 없는 LayerNorm" | RMS로 나눈다. 연산 하나가 적고, 실험적 안정성은 같다. |
| SwiGLU | "모두가 갈아탄 FFN" | `Swish(W1 x) ⊙ W3 x → W2`. LM perplexity에서 ReLU/GELU보다 낫다. |
| Cross-attention | "decoder가 encoder를 보는 방식" | decoder의 Q, encoder output의 K/V를 사용하는 MHA. |
| FFN expansion | "가운데 MLP의 폭" | hidden size와 d_model의 비율. 보통 4(LayerNorm) 또는 2.6(SwiGLU). |
| Bias-free | "+b term 제거" | 현대 stack은 linear layer의 bias를 생략한다. perplexity가 약간 좋아지고 model이 작아진다. |

## 더 읽을거리

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 원래 block spec.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — pre-norm이 깊은 모델에서 post-norm보다 나은 이유.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical 2026 decoder-only block.
