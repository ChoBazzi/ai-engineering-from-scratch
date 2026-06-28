# Attention Mechanism — 돌파구

> Decoder가 압축된 summary를 힘겹게 들여다보는 일을 멈추고 전체 source를 보기 시작한다. 이후의 모든 것은 attention에 engineering을 더한 것이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45분

## 문제

Lesson 09는 측정된 실패로 끝났다. Toy copy task로 train한 GRU encoder-decoder는 length 5에서 89% accuracy를 보이다가 length 80에서는 chance 수준에 가까워진다. 이유는 training bug가 아니라 구조적 문제다. Encoder가 얻은 모든 bit의 정보가 하나의 fixed-size hidden state에 들어가야 하고, decoder는 그 외의 어떤 것도 보지 못한다.

Bahdanau, Cho, Bengio는 2014년에 세 줄짜리 fix를 발표했다. Decoder에 마지막 encoder state만 주지 말고 모든 encoder state를 보관한다. 각 decoder step에서 encoder state의 weighted average를 계산한다. 이때 weight는 "decoder가 지금 encoder position `i`를 얼마나 봐야 하는가?"를 뜻한다. 그 weighted average가 context이며, decoder step마다 바뀐다.

이것이 전체 아이디어다. Transformer는 이를 확장했다. Self-attention은 이를 단일 sequence에 적용했다. Multi-head attention은 이를 parallel로 실행했다. 하지만 2014년 버전은 이미 bottleneck을 깼고, 이것을 이해하면 transformer로의 전환은 conceptual 문제가 아니라 engineering 문제가 된다.

## 개념

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

각 decoder step `t`에서:

1. 이전 decoder hidden state `s_{t-1}`를 **query**로 사용한다.
2. 모든 encoder hidden state `h_1, ..., h_T`와 score를 계산한다. Encoder position마다 scalar 하나다.
3. Score에 softmax를 적용해 합이 1인 attention weight `α_{t,1}, ..., α_{t,T}`를 얻는다.
4. Context vector `c_t = Σ α_{t,i} * h_i`. Encoder state의 weighted average다.
5. Decoder는 `c_t`와 previous output token을 받아 다음 token을 만든다.

Weighted average가 핵심이다. Decoder가 "Je"를 "I"로 번역해야 할 때는 "Je" 위의 encoder state에 높은 weight를 주고 나머지는 낮춘다. "not"이 필요할 때는 "pas"에 높은 weight를 준다. Context vector는 step마다 다시 형성된다.

## Shapes(모두가 한 번씩 물리는 지점)

여기가 모든 attention implementation이 처음에 틀리는 곳이다. 천천히 읽어라.

| 항목 | Shape | 메모 |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | BiLSTM이면 `d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | Vector 하나 |
| Attention score `e_{t,i}` | scalar | Encoder position마다 하나 |
| Attention weight `α_{t,i}` | scalar | 모든 `i`에 대해 softmax를 적용한 뒤 |
| Context vector `c_t` | `(d_h,)` | Encoder state와 같은 shape |

**Bahdanau(additive) score.** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}`의 shape는 `(d_s,)`, `h_i`의 shape는 `(d_h,)`다.
- `W_a`의 shape는 `(d_attn, d_s)`다. `U_a`의 shape는 `(d_attn, d_h)`다.
- Tanh 안의 합은 shape `(d_attn,)`를 가진다.
- `v_α`의 shape는 `(d_attn,)`다. `v_α`와의 inner product가 scalar로 collapse된다. **이것이 `v_α`가 하는 일이다.** 마법이 아니다. Attention-dim vector를 scalar score로 바꾸는 projection이다.

**Luong(multiplicative) score.** 세 가지 variant:

- `dot`: `e_{t,i} = s_t^T * h_i`. `d_s == d_h`가 필요하다. 강한 constraint다. Encoder가 bidirectional이면 건너뛰어라.
- `general`: `e_{t,i} = s_t^T * W * h_i`, `W` shape는 `(d_s, d_h)`다. Equal-dim constraint를 제거한다.
- `concat`: 본질적으로 Bahdanau form이다. 앞의 두 방식이 더 싸기 때문에 거의 쓰이지 않는다.

**이름 붙일 가치가 있는 Bahdanau / Luong gotcha.** Bahdanau는 `s_{t-1}`를 사용한다(current word를 생성하기 *전* decoder state). Luong은 `s_t`를 사용한다(*후* state). 둘을 섞으면 매우 debug하기 어려운 미묘하게 잘못된 gradient가 생긴다. Paper 하나를 고르고 그 convention을 지켜라.

```figure
attention-heatmap
```

## 직접 만들기

### 1단계: additive(Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

위 table과 shape를 대조하라. `encoder_states`의 shape는 `(T_enc, d_h)`다. `projected_enc`의 shape는 `(T_enc, d_attn)`다. `projected_dec`의 shape는 `(d_attn,)`이고 broadcast된다. `combined`의 shape는 `(T_enc, d_attn)`다. `scores`의 shape는 `(T_enc,)`다. `weights`의 shape는 `(T_enc,)`다. `context`의 shape는 `(d_h,)`다. 완료다.

### 2단계: Luong dot과 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

각각 세 줄이다. 이것이 Luong paper가 자리 잡은 이유다. 대부분의 task에서 같은 accuracy를 내면서 code는 훨씬 적다.

### 3단계: 수치 예제

세 encoder state(대략 "cat", "sat", "mat")와 첫 번째에 가장 잘 align되는 decoder state가 주어지면 attention distribution은 position 0에 집중된다. Decoder state가 마지막 encoder state에 더 가깝게 이동하면 attention도 position 2로 이동한다. Context vector가 이를 따라간다.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```text
weights: [0.464 0.305 0.231]
```

첫 row가 이긴다. 그다음 decoder state를 세 번째 encoder state에 더 가깝게 옮기고 weight가 이동하는 것을 보라. 그게 전부다. Attention은 explicit alignment다.

### 4단계: 이것이 transformer로 가는 bridge인 이유

위의 언어를 Q/K/V로 번역하면 다음과 같다.

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder states(score를 계산할 대상)
- **Value** = encoder states(weight를 곱해 합산할 대상)

Classical attention에서 key와 value는 같은 것이다. Self-attention은 둘을 분리한다. K와 V에 대해 서로 다른 learned projection을 사용해 sequence를 자기 자신에 query할 수 있다. Multi-head attention은 서로 다른 learned projection으로 이를 parallel하게 실행한다. Transformer는 전체 stage를 여러 번 쌓고 RNN을 버린다.

수학은 같다. Shape도 같다. Bahdanau attention에서 scaled dot-product attention으로 넘어가는 교육적 도약은 대부분 notation의 차이다.

## 사용하기

PyTorch와 TensorFlow는 attention을 직접 제공한다.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```text
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

이것이 transformer attention layer다. 5 position의 query batch, 10 position의 key/value batch, 각각 128-dim, 8 heads. `output`은 새로운 context-augmented queries다. `weights`는 visualize할 수 있는 5x10 alignment matrix다.

### Classical attention이 여전히 중요한 때

- 교육. Single-head, single-layer, RNN-based version은 모든 concept을 보이게 만든다.
- Transformer가 맞지 않는 on-device sequence task.
- 2014-2017년의 모든 paper. Bahdanau convention을 모르면 잘못 읽게 된다.
- MT의 fine-grained alignment analysis. Raw attention weight는 transformer model에서도 interpretability tool이며, 이를 읽으려면 그것이 무엇인지 알아야 한다.

### Attention-weight-as-explanation 함정

Attention weight는 해석 가능해 보인다. Position 전체에 걸쳐 합이 1인 weight이고, plot할 수 있으며, 높다는 것은 "여기를 봤다"는 뜻처럼 보인다. Reviewer들이 좋아한다.

하지만 보이는 것만큼 해석 가능하지 않다. Jain and Wallace(2019)는 일부 task에서 attention distribution을 permute하거나 임의의 alternative로 바꿔도 model prediction이 변하지 않을 수 있음을 보였다. Ablation이나 counterfactual check 없이 attention weight를 reasoning의 evidence로 보고하지 마라.

## 내보내기

`outputs/prompt-attention-shapes.md`로 저장한다.

```markdown
---
name: attention-shapes
description: Attention implementation의 shape bug를 debug한다.
phase: 5
lesson: 10
---

깨진 attention implementation이 주어지면 shape mismatch를 식별한다. 다음을 출력한다.

1. 어떤 matrix의 shape가 잘못되었는지. Tensor 이름을 명시한다.
2. 그 shape가 무엇이어야 하는지. `(d_s, d_h, d_attn, T_enc, T_dec, batch_size)`에서 도출한다.
3. 한 줄 수정. Transpose, reshape, 또는 project.
4. Regression을 잡는 test. 일반적으로 `output.shape == (batch, T_dec, d_h)`와 `weights.shape == (batch, T_dec, T_enc)`를 assert하고, `weights.sum(dim=-1)`이 1에 가까운지 확인한다.

조용히 broadcast되는 fix는 추천하지 않는다. Broadcast에 가려진 bug는 나중에 silent accuracy degradation으로 드러난다.

Bahdanau 혼동에서는 decoder input이 `s_{t-1}`(pre-step state)라고 insist한다. Luong에서는 `s_t`(post-step state)다. Dot-product attention에서 처음 가장 흔한 error는 query/key dimension mismatch이므로 명시적으로 flag한다.
```

## 연습문제

1. **쉬움.** Encoder의 padding token이 attention weight zero를 갖도록 `softmax` masking을 구현하라. Variable-length sequence가 있는 batch로 test하라.
2. **중간.** Luong `general` form에 multi-head attention을 추가하라. `d_h`를 `n_heads` group으로 나누고, head마다 attention을 실행한 뒤 concatenate하라. Single-head case가 이전 implementation과 일치하는지 확인하라.
3. **어려움.** Lesson 09의 toy copy task에서 Bahdanau attention을 사용하는 GRU encoder-decoder를 train하라. Accuracy vs sequence length를 plot하라. No-attention baseline과 비교하라. Length가 커질수록 gap이 벌어지는 것을 볼 수 있어야 하며, 이는 attention이 bottleneck을 완화한다는 것을 확인해 준다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| Attention | 무언가를 보는 것 | Value sequence의 weighted average이며, weight는 query-key similarity에서 계산된다. |
| Query, Key, Value | QKV | 세 projection: Q는 묻고, K는 match 대상이며, V는 반환할 대상이다. |
| Additive attention | Bahdanau | Feed-forward score: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Score는 `q^T k` 또는 `q^T W k`다. 더 싸고, 대부분의 task에서 accuracy가 같다. |
| Alignment matrix | 예쁜 그림 | `(T_dec, T_enc)` grid로 된 attention weight. Model이 무엇에 attend했는지 보기 위해 읽는다. |

## 더 읽을거리

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 해당 paper.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 세 score variant와 그 비교.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — interpretability caveat.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — PyTorch로 실행 가능한 walkthrough.
