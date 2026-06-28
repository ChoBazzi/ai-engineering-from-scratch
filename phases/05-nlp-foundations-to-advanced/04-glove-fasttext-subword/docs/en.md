# GloVe, FastText, Subword Embeddings

> Word2Vec은 단어마다 embedding 하나를 학습했습니다. GloVe는 co-occurrence matrix를 factorize했습니다. FastText는 조각을 embedding했습니다. BPE는 transformer로 가는 다리를 놓았습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 minutes

## 문제

Word2Vec은 두 가지 열린 질문을 남겼습니다.

첫째, online skip-gram update 대신 co-occurrence matrix를 직접 factorize하는 연구 흐름이 따로 있었습니다(LSA, HAL). Word2Vec의 iterative approach가 근본적으로 더 나았던 것일까요, 아니면 두 방법이 count를 다루는 방식 때문에 생긴 차이였을까요? **GloVe**가 이에 답했습니다. 신중하게 고른 loss를 사용한 matrix factorization은 Word2Vec과 맞먹거나 이기며, 학습 비용은 더 낮습니다.

둘째, 두 방법 모두 한 번도 본 적 없는 단어를 처리할 방법이 없었습니다. `Zoomer-approved`, `dogecoin`, 지난주에 만들어진 proper noun, 드문 root의 모든 inflected form이 여기에 해당합니다. **FastText**는 character n-gram을 embedding해 이 문제를 고쳤습니다. 단어는 morpheme을 포함한 부분들의 합이므로 out-of-vocabulary word도 그럴듯한 vector를 얻습니다.

셋째, transformer가 등장하자 질문이 다시 바뀌었습니다. word-level vocabulary는 백만 항목쯤에서 한계에 부딪힙니다. 실제 언어는 그보다 훨씬 열려 있습니다. **Byte-pair encoding (BPE)**과 그 친척들은 모든 것을 덮는 빈번한 subword unit의 vocabulary를 학습해 이 문제를 해결했습니다. 모든 현대 LLM의 모든 현대 tokenizer는 subword tokenizer입니다.

이 lesson은 세 가지를 모두 살펴본 뒤, 언제 무엇을 선택해야 하는지 설명합니다.

## 개념

**GloVe (Global Vectors).** word-word co-occurrence matrix `X`를 만듭니다. 여기서 `X[i][j]`는 word `j`가 word `i`의 context에 등장한 횟수입니다. `v_i · v_j + b_i + b_j ≈ log(X[i][j])`가 되도록 vector를 학습합니다. frequent pair가 loss를 지배하지 않도록 가중치를 둡니다. 끝입니다.

**FastText.** 단어는 자기 자신과 character n-gram의 합입니다. `where`는 `<wh, whe, her, ere, re>, <where>`가 됩니다. word vector는 이런 component vector의 합입니다. Word2Vec처럼 학습합니다. 장점: unseen word(`whereupon`)도 알려진 n-gram에서 조합됩니다.

**BPE (Byte-Pair Encoding).** 개별 byte(또는 character)의 vocabulary에서 시작합니다. corpus의 모든 adjacent pair를 셉니다. 가장 frequent한 pair를 새 token으로 merge합니다. 이를 `k`번 반복합니다. 결과는 frequent sequence(`ing`, `tion`, `the`)가 single token이 되고 rare word는 익숙한 조각으로 쪼개지는 `k + 256` token vocabulary입니다. 모든 문장은 무언가로 tokenize됩니다.

## 직접 만들기

### GloVe: co-occurrence matrix factorize하기

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

이름 붙일 만한 moving piece가 두 개 있습니다. weighting function `f(x) = (x/x_max)^alpha`는 `(the, and)`처럼 매우 frequent한 pair가 loss를 지배하지 않도록 downweight합니다. 최종 embedding은 `W`(center)와 `W_tilde`(context) table의 합입니다. 둘을 더하는 것은 하나만 쓰는 것보다 성능이 좋은 경향이 있는 published trick입니다.

### FastText: subword-aware embedding

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

각 단어는 n-gram 집합(보통 3-6 character)으로 표현됩니다. word embedding은 n-gram embedding의 합입니다. skip-gram training에서는 Word2Vec이 single vector를 사용하던 자리에 이것을 꽂습니다.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

unseen word라도 그 n-gram 일부를 알고 있으면 vector를 얻습니다. `whereupon`은 `where`와 `<wh`, `her`, `ere`, `<where`를 공유하므로 둘은 서로 가까운 곳에 놓입니다.

### BPE: 학습된 subword vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

첫 번째 iteration은 가장 common한 adjacent pair를 merge합니다. 충분히 반복하면 frequent substring(`low`, `est`, `tion`)은 single token이 되고 rare word는 깔끔하게 쪼개집니다.

실제 GPT / BERT / T5 tokenizer는 30k-100k merge를 학습합니다. 결과: 어떤 text든 알려진 ID의 bounded-length sequence로 tokenize되고, OOV는 절대 없습니다.

## 사용하기

실무에서는 이 중 어떤 것도 직접 학습하는 일이 드뭅니다. pre-trained checkpoint를 로드합니다.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

transformer 시대의 BPE-style subword tokenization은 다음과 같습니다.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```text
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` prefix는 word boundary를 표시합니다(GPT-2 convention). 모든 현대 tokenizer는 BPE variant, WordPiece(BERT), 또는 SentencePiece(T5, LLaMA)입니다.

### 무엇을 선택할까

| 상황 | 선택 |
|------|------|
| pre-trained general-purpose word vector가 필요하고 OOV tolerance가 필요 없음 | GloVe 300d |
| pre-trained general-purpose word vector가 필요하고 misspelling / neologism / morphologically rich language를 처리해야 함 | FastText |
| transformer(training 또는 inference)에 들어가는 모든 것 | 모델과 함께 제공된 tokenizer. 절대 바꾸지 마세요. |
| language model을 직접 처음부터 학습 | corpus에서 BPE 또는 SentencePiece tokenizer를 먼저 학습 |
| linear model 기반 production text classification | 여전히 TF-IDF. Lesson 02. |

## 내보내기

`outputs/skill-embeddings-picker.md`로 저장합니다.

```markdown
---
name: tokenizer-picker
description: 새 language model 또는 text pipeline을 위한 tokenization approach를 선택합니다.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

task와 dataset description이 주어지면 다음을 출력합니다.

1. Tokenization strategy(word-level, BPE, WordPiece, SentencePiece, byte-level). 한 문장짜리 이유.
2. Vocabulary size target(예: English-only LM은 32k, multilingual은 64k-100k).
3. 정확한 training command가 포함된 library call. library 이름을 말하세요. argument를 quote하세요.
4. reproducibility pitfall 하나. Tokenizer-model mismatch는 production에서 가장 흔한 silent bug입니다. 어떤 pair를 함께 사용해야 하는지 지적하세요.

사용자가 pretrained LLM을 fine-tuning하는 경우 custom tokenizer 학습을 권장하지 마세요. production inference를 목표로 하는 어떤 model에도 word-level tokenization을 권장하지 마세요. non-English / multi-script corpus에는 byte fallback이 있는 SentencePiece가 필요하다고 표시하세요.
```

## 연습 문제

1. **Easy.** `char_ngrams("playing")`와 `char_ngrams("played")`를 실행하세요. 두 n-gram set의 Jaccard overlap을 계산하세요. 공유 조각(`pla`, `lay`, `play`)이 상당히 많다는 것을 볼 수 있습니다. 그래서 FastText는 morphological variant 사이에서 잘 전이됩니다.
2. **Medium.** `learn_bpe`를 확장해 vocabulary growth를 추적하세요. merge 수의 함수로 tokens-per-corpus-character를 plot하세요. 처음에는 빠르게 compression되고, 이후 token당 ~2-3 character 근처에서 asymptote하는 모습을 볼 수 있습니다.
3. **Hard.** Shakespeare complete works에서 1k-merge BPE를 학습하세요. common word와 rare proper noun의 tokenization을 비교하세요. 전후의 average tokens per word를 측정하세요. 무엇이 의외였는지 정리하세요.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Co-occurrence matrix | word-word frequency table | `X[i][j]` = word `j`가 word `i` 주변 window에 등장하는 횟수. |
| Subword | 단어의 조각 | character n-gram(FastText) 또는 learned token(BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | vocabulary가 target size에 도달할 때까지 most-frequent adjacent pair를 반복적으로 merge. |
| OOV | Out of vocabulary | 모델이 한 번도 본 적 없는 단어. Word2Vec/GloVe는 실패합니다. FastText와 BPE는 처리합니다. |
| Byte-level BPE | raw byte 위의 BPE | GPT-2의 방식입니다. Vocabulary가 256 byte에서 시작하므로 어떤 것도 OOV가 되지 않습니다. |

## 더 읽을거리

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 논문입니다. 7쪽이며, loss 유도는 여전히 이 글이 가장 좋습니다.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE를 현대 NLP에 도입한 논문입니다.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE, WordPiece, SentencePiece가 실제로 어떻게 다른지 설명합니다.
