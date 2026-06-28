# Transformers 이전의 Text Generation — N-gram Language Models

> 어떤 단어가 놀랍다면 model이 나쁜 것입니다. Perplexity는 그 놀라움을 숫자로 만듭니다. Smoothing은 그 값을 finite하게 유지합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제

transformers 이전, RNNs 이전, word embeddings 이전에 language model은 이전 `n-1`개 단어 뒤에 다음 단어가 얼마나 자주 나왔는지 세어서 next word를 예측했습니다. "the cat" → "sat"은 47번, "the cat" → "jumped"는 12번, "the cat" → "refrigerator"는 0번 세는 식입니다. normalize하면 probability distribution이 됩니다.

이것이 n-gram language model입니다. 1980년부터 2015년까지 거의 모든 speech recognizer, spell checker, phrase-based machine translation system을 움직였습니다. cheap on-device language modeling이 필요할 때는 지금도 실행됩니다.

흥미로운 문제는 unseen n-grams를 어떻게 처리하느냐입니다. raw count-based model은 보지 못한 모든 것에 zero probability를 할당합니다. 문장은 길고 거의 모든 긴 문장에는 적어도 하나의 unseen sequence가 있으므로 이는 치명적입니다. 50년의 smoothing 연구가 이 문제를 해결했습니다. 그 결과가 Kneser-Ney smoothing이고, modern deep learning은 그 empirical tradition을 물려받았습니다.

## 개념

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-gram probability:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. `n`을 고정합니다(보통 trigrams는 3, 4-grams는 4). counts에서 계산합니다:

```text
P(w | context) = count(context, w) / count(context)
```

**zero-count problem.** training에서 보지 못한 모든 n-gram은 probability zero를 받습니다. Brown corpus에 대한 2007년 연구는 4-gram model조차 held-out 4-grams의 30%가 training에서 unseen이었다는 것을 보였습니다. smoothing 없이는 어떤 real text에서도 evaluation할 수 없습니다.

**Smoothing approaches, 정교한 순서:**

1. **Laplace (add-one).** 모든 count에 1을 더합니다. 단순하지만 rare events에는 형편없습니다.
2. **Good-Turing.** frequency-of-frequencies를 바탕으로 higher-frequency events의 probability mass를 unseen events로 재할당합니다.
3. **Interpolation.** n-gram, (n-1)-gram 등의 estimates를 조정 가능한 weights로 결합합니다.
4. **Backoff.** n-gram count가 zero이면 (n-1)-gram으로 fallback합니다. Katz backoff는 이를 normalize합니다.
5. **Absolute discounting.** 모든 counts에서 고정 discount `D`를 빼고 unseen에 재분배합니다.
6. **Kneser-Ney.** Absolute discounting에 더해 lower-order model을 똑똑하게 고릅니다. raw frequency 대신 *continuation probability*(단어가 등장하는 contexts 수)를 사용합니다.

Kneser-Ney의 insight는 깊습니다. "San Francisco"는 흔한 bigram입니다. Unigram "Francisco"는 대부분 "San" 뒤에 등장합니다. naive absolute discounting은 "Francisco"에 높은 unigram probability를 줍니다(count가 높기 때문). Kneser-Ney는 "Francisco"가 오직 하나의 context에만 등장한다는 점을 알아차리고 그에 맞게 continuation probability를 낮춥니다. 결과적으로 "Francisco"로 끝나는 새로운 bigram은 적절히 낮은 probability를 받습니다.

**Evaluation: perplexity.** held-out test set에서 단어당 average negative log-likelihood의 exponent입니다. 낮을수록 좋습니다. perplexity 100은 model이 100개 단어 중 uniform하게 고르는 것만큼 혼란스러워한다는 뜻입니다.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 직접 만들기

### 1단계: trigram counts

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Input은 tokenized sentences의 list입니다. Output은 n-gram counts와 context counts입니다. `<s>`와 `</s>`는 sentence boundaries입니다.

### 2단계: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

모든 count에 1을 더합니다. smoothing은 되지만 unseen events에 mass를 과도하게 할당하여 rare-known events도 해칩니다.

### 3단계: Kneser-Ney (bigram, interpolated)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

움직이는 부분은 세 가지입니다. `continuation_prob`는 "이 단어가 몇 개의 서로 다른 contexts에 등장하는가?"를 포착합니다(Kneser-Ney의 innovation). `lambda_prev`는 discount로 해방된 mass이며 backoff에 weight를 주는 데 사용됩니다. 최종 probability는 discounted main term과 weighted continuation term의 합입니다.

### 4단계: sampling으로 text 생성하기

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

probability에 비례해 sampling합니다. seed마다 항상 다른 output을 냅니다. beam-search 같은 output을 원하면 각 step에서 argmax를 고르고(greedy), 작은 randomness knob(temperature)를 추가하세요.

### 5단계: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

낮을수록 좋습니다. Brown corpus에서 잘 tuned된 4-gram KN model은 perplexity 약 140에 도달합니다. 같은 test set에서 transformer LM은 15-30에 도달합니다. 격차는 약 10x입니다. 이 격차 때문에 분야가 다음 단계로 이동했습니다.

## 활용하기

- **Classical NLP teaching.** smoothing, MLE, perplexity를 가장 명확하게 접할 수 있는 방법입니다.
- **KenLM.** Production n-gram library입니다. low latency가 중요한 speech와 MT systems에서 rescorer로 사용됩니다.
- **On-device autocomplete.** keyboards의 trigram models입니다. 지금도 그렇습니다.
- **Baselines.** neural LM이 좋다고 말하기 전에 항상 n-gram LM perplexity를 계산하세요. transformer가 KN을 큰 차이로 이기지 못하면 무언가 잘못된 것입니다.

## 결과물 만들기

`outputs/prompt-lm-baseline.md`로 저장하세요:

```markdown
---
name: lm-baseline
description: neural LM을 training하기 전에 reproducible n-gram language model baseline을 만듭니다.
phase: 5
lesson: 16
---

corpus와 target use(next-word prediction, rescoring, perplexity baseline)가 주어지면 다음을 출력하세요:

1. N-gram order. 일반 English에는 trigram, corpus가 크면 4-gram, speech rescoring에는 5-gram.
2. Smoothing. Modified Kneser-Ney가 default입니다. Laplace는 teaching용으로만 사용합니다.
3. Library. production에는 `kenlm`, teaching에는 `nltk.lm`, 직접 구현은 학습 목적으로만 사용합니다.
4. Evaluation. train과 test sets 사이에 consistent tokenization을 사용한 held-out perplexity.

비교 중인 systems 사이에 서로 다른 tokenization으로 계산한 perplexity는 보고하지 마세요. perplexity numbers는 identical tokenization에서만 비교 가능합니다. test set의 OOV rate를 flag하세요. training 중 special <UNK> token을 reserve하지 않으면 KN은 OOV를 잘 처리하지 못합니다.
```

## 연습문제

1. **Easy.** 1,000-sentence Shakespeare corpus에서 trigram LM을 train하세요. 20개 sentences를 generate하세요. locally plausible하지만 globally incoherent할 것입니다. 이것이 canonical demo입니다.
2. **Medium.** held-out Shakespeare split에서 KN model의 perplexity를 구현하세요. Laplace와 비교하세요. KN이 perplexity를 30-50% 낮추는 것을 볼 수 있어야 합니다.
3. **Hard.** trigram spell corrector를 만드세요. misspelled word와 그 context가 주어지면 corrections를 생성하고 LM 아래의 context probability로 rank하세요. Birkbeck spelling corpus(public)에서 evaluate하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | Word sequence | 연속된 `n`개 tokens의 sequence입니다. |
| Smoothing | zeros 피하기 | unseen events가 non-zero probability를 받도록 probability mass를 재할당하는 것입니다. |
| Perplexity | LM quality metric | held-out data에서 `exp(-average log-prob)`입니다. 낮을수록 좋습니다. |
| Backoff | 더 짧은 context로 fallback | trigram count가 zero이면 bigram을 사용합니다. Katz backoff가 이를 formalize합니다. |
| Kneser-Ney | n-grams용 best smoothing | Absolute discounting + lower-order model용 continuation probability입니다. |
| Continuation probability | KN-specific | raw count가 아니라 `w`가 등장하는 contexts 수로 weight된 `P(w)`입니다. |

## 더 읽을거리

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LMs와 smoothing의 canonical treatment.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — Kneser-Ney를 best n-gram smoother로 자리 잡게 한 paper.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — original KN paper.
- [KenLM](https://kheafield.com/code/kenlm/) — 빠른 production n-gram LM이며, latency-sensitive applications에서 2026년에도 사용됩니다.
