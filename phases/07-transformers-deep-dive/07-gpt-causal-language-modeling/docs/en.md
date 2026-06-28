# GPT — Causal Language Modeling

> BERT는 양쪽을 본다. GPT는 과거만 본다. triangle mask는 현대 AI에서 가장 큰 결과를 낳은 코드 한 줄이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## 문제

language model은 한 가지 질문에 답한다. 처음 `t-1`개 token이 주어졌을 때 token `t`에 대한 probability distribution은 무엇인가? 이 signal, 즉 next-token prediction으로 학습하면 token을 하나씩 생성해 임의의 text를 만들 수 있는 model을 얻는다.

전체 sequence를 parallel하게 end-to-end로 학습하려면 각 position의 prediction이 이전 position에만 의존해야 한다. 그렇지 않으면 model은 정답을 들여다보며 trivial하게 cheating한다.

causal mask가 이것을 해낸다. softmax 전에 attention score에 더하는 `-inf` 값의 upper-triangular matrix 하나다. softmax 후에는 해당 position들이 0이 된다. 각 position은 자기 자신과 이전 position에만 attend할 수 있다. 그리고 전체 sequence에 한 번 적용하므로 forward pass 하나에서 N개의 parallel next-token prediction을 얻는다.

GPT-1(2018), GPT-2(2019), GPT-3(2020), GPT-4(2023), GPT-5(2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi는 모두 같은 core loop를 가진 decoder-only causal transformer다. 더 클 뿐이고, data가 더 좋고, RLHF가 더 좋아졌을 뿐이다.

## 개념

![Causal mask는 삼각형 attention matrix를 만든다](../assets/causal-attention.svg)

### 마스크

길이 `N`의 sequence가 주어지면 `N × N` matrix를 만든다.

```text
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

softmax 전에 raw attention score에 `M`을 더한다. `exp(-inf) = 0`이므로 masked position은 weight에 0만 기여한다. attention matrix의 각 row는 이전 position들에 대한 probability distribution만 된다.

구현 비용은 `torch.tril()` 호출 하나다. compute 시간은 nanosecond다. field에 미친 영향은 전부라고 해도 된다.

### 병렬 training, 직렬 inference

training에서는 전체 `(N, d_model)` sequence를 한 번 forward-pass하고, N개의 cross-entropy loss(position마다 하나)를 계산해 더한 뒤 backprop한다. sequence 방향으로 parallel하다. 이것이 GPT training이 scale되는 이유다. GPU pass 한 번으로 batch 안의 1M token을 처리한다.

inference에서는 token을 하나씩 생성한다. `[t1, t2, t3]`를 넣어 `t4`를 얻는다. `[t1, t2, t3, t4]`를 넣어 `t5`를 얻는다. `[t1, t2, t3, t4, t5]`를 넣어 `t6`를 얻는다. KV cache(Lesson 12)는 `t1…tn`의 hidden state를 저장해 매 step마다 다시 계산하지 않게 한다. 하지만 inference의 serial depth는 output length와 같다. 이것이 autoregressive tax이며, 모든 LLM에서 decoding이 latency bottleneck인 이유다.

### 손실 — shift-by-one

token `[t1, t2, t3, t4]`가 주어졌을 때:

- 입력: `[t1, t2, t3]`
- Targets: `[t2, t3, t4]`

각 position `i`마다 `-log P(target_i | inputs[:i+1])`를 계산한다. 모두 더한다. 이것이 전체 sequence의 cross-entropy다.

들어 본 모든 transformer LM은 이 loss로 학습한다. pre-training, fine-tuning, SFT 모두 같은 loss를 쓰고 data만 다르다.

### Decoding 전략

학습 후에는 sampling 선택이 사람들이 생각하는 것보다 더 중요하다.

| 방법 | 동작 | 사용 시점 |
|--------|--------------|-------------|
| Greedy | 매 step argmax | Deterministic task, code completion |
| Temperature | logit을 T로 나누고 sample | Creative task, T가 높을수록 diversity 증가 |
| Top-k | top-k token에서만 sample | 낮은 probability tail 제거 |
| Top-p (nucleus) | cumulative prob ≥ p가 되는 가장 작은 set에서 sample | 2020년 이후 기본값. distribution shape에 적응 |
| Min-p | `p > min_p * max_p`인 token 유지 | 2024년 이후. 긴 tail을 거부하는 데 top-p보다 좋음 |
| Speculative decoding | draft model이 N개 token을 제안하고 큰 model이 검증 | 같은 quality에서 2-3× latency 감소 |

2026년에는 open-weights model에 min-p + temperature 0.7이 합리적인 기본값이다. speculative decoding은 production inference stack의 기본 요건이다.

### "GPT recipe"가 통하게 만든 것

1. **Decoder-only.** encoder overhead가 없다. layer마다 attention + FFN을 한 번 통과한다.
2. **Scaling.** 124M → 1.5B → 175B → trillions. Chinchilla scaling laws(Lesson 13)는 compute를 어떻게 쓸지 알려 준다.
3. **In-context learning.** 6B-13B 무렵 emergence했다. model은 fine-tuning 없이 few-shot example을 따를 수 있다.
4. **RLHF.** human preference에 대한 post-training이 raw pretrained text를 chat assistant로 바꿨다.
5. **Pre-norm + RoPE + SwiGLU.** scale에서 안정적인 training.

core architecture는 GPT-2 이후 크게 바뀌지 않았다. 흥미로운 변화는 모두 data, scale, post-training에서 일어났다.

```figure
causal-mask
```

## 직접 만들기

### 1단계: causal mask

`code/main.py`를 보라. 한 줄이면 된다.

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

attention score에 softmax 전에 더한다. 이것이 전체 mechanism이다.

### 2단계: 2-layer GPT-ish model

decoder block 두 개를 쌓는다(masked self-attention + FFN, cross-attention 없음). token embedding, positional encoding, unembedding을 추가한다(unembedding은 token embedding matrix와 묶는다. GPT-2 이후 표준 trick이다).

### 3단계: next-token prediction end-to-end

20-token toy vocab에서 모든 position의 logit을 만든다. shift-by-one target에 대해 cross-entropy loss를 계산한다. gradient는 쓰지 않는다. forward-pass sanity check다.

### 4단계: sampling

greedy, temperature, top-k, top-p, min-p를 구현한다. 고정 prompt에서 각각을 실행하고 output을 비교한다. sampling function은 10줄이면 된다.

## 활용하기

PyTorch, 2026 idiom:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

내부에서 `generate()`는 forward pass를 실행하고 final-position logit을 뽑아 next token을 sample한 뒤 append하고 반복한다. 모든 production LLM inference stack(vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX)은 같은 loop를 강하게 optimize해 구현한다. batched prefill, continuous batching, KV cache paging, speculative decoding이 그 예다.

**GPT vs BERT, 한 줄씩:** GPT는 `P(x_t | x_{<t})`를 예측한다. BERT는 `P(x_masked | x_unmasked)`를 예측한다. loss가 model이 generate할 수 있는지를 결정한다.

## 실전 적용

`outputs/skill-sampling-tuner.md`를 보라. 이 skill은 새로운 generation task에 맞는 sampling parameter를 고르고, deterministic decoding이 필요한 경우를 표시한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하고 softmax 이후 causal attention matrix가 lower-triangular인지 확인하라. spot-check: row 3은 column 0-3에만 weight가 있어야 한다.
2. **보통.** width 4의 beam search를 구현하라. 10개의 짧은 prompt에서 beam-4와 greedy의 perplexity를 비교하라. beam이 항상 이기는가? (힌트: 보통 translation에서는 그렇지만 open-ended chat에서는 아니다.)
3. **어려움.** speculative decoding을 구현하라. tiny 2-layer model을 draft로, 6-layer model을 verifier로 사용한다. 길이 64 completion 100개에서 wall-clock speedup을 측정하라. output이 verifier의 greedy와 일치하는지 확인하라.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Causal mask | "Triangle" | position `i`가 `≤ i` position만 보도록 attention score에 더하는 upper-triangular `-inf` matrix. |
| Next-token prediction | "Loss" | 모든 position에서 true next token에 대한 model distribution의 cross-entropy. |
| Autoregressive | "하나씩 생성" | output을 다시 input으로 넣는다. parallelism은 training 중에만 있고 generation 중에는 없다. |
| Logits | "Pre-softmax score" | softmax 전 LM head의 raw output. sampling은 여기서 일어난다. |
| Temperature | "Creativity knob" | logit을 T로 나눈다. T→0 = greedy, T→∞ = uniform. |
| Top-p | "Nucleus sampling" | 합이 ≥p가 되는 가장 작은 set으로 distribution을 잘라 남은 것에서 sample한다. |
| Min-p | "Top-p보다 낫다" | `p ≥ min_p × max_p`인 token을 유지한다. distribution의 sharpness에 맞춰 cutoff를 조정한다. |
| Speculative decoding | "Draft + verify" | 싼 model이 N개 token을 제안하고 큰 model이 parallel하게 검증한다. |
| Teacher forcing | "Training trick" | training 중에는 model prediction이 아니라 true previous token을 넣는다. 모든 seq2seq LM의 표준이다. |

## 더 읽을거리

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3와 in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 논문.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical causal-LM reference code.
