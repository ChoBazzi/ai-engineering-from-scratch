# Word Embeddings — Word2Vec 직접 구현

> 단어는 함께 등장하는 이웃으로 알 수 있습니다. 이 생각으로 얕은 신경망을 학습하면 기하 구조가 드러납니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 minutes

## 문제

TF-IDF는 `dog`와 `puppy`가 서로 다른 단어라는 사실은 압니다. 하지만 둘이 거의 같은 뜻이라는 사실은 모릅니다. `dog`로 학습한 분류기는 `puppy`가 들어간 리뷰로 일반화할 수 없습니다. 동의어 목록을 만들어 덮어볼 수는 있지만, 드문 용어, 도메인 전문어, 예상하지 못한 모든 언어에서 실패합니다.

우리가 원하는 것은 `dog`와 `puppy`가 공간에서 가까운 곳에 놓이는 표현입니다. `king - man + woman`이 `queen` 근처에 도착하는 공간입니다. `dog`로 학습한 모델이 공짜로 일부 신호를 `puppy`에도 전달하는 표현입니다.

Word2Vec이 그 공간을 제공했습니다. 2층 신경망, 조 단위 토큰 학습 실행, 2013년 발표. 아키텍처는 민망할 정도로 단순합니다. 결과는 이후 10년간 NLP를 다시 만들었습니다.

## 개념

**Distributional hypothesis**(Firth, 1957): "You shall know a word by the company it keeps." 두 단어가 비슷한 문맥에 등장한다면, 둘은 비슷한 뜻일 가능성이 큽니다.

Word2Vec에는 이 아이디어를 활용하는 두 가지 방식이 있습니다.

- **Skip-gram.** 중심 단어가 주어졌을 때 주변 단어를 예측합니다. window size 2라면 `cat -> (the, sat, on)`입니다.
- **CBOW (continuous bag of words).** 주변 단어가 주어졌을 때 중심 단어를 예측합니다. `(the, sat, on) -> cat`입니다.

Skip-gram은 학습이 더 느리지만 드문 단어를 더 잘 다룹니다. 그래서 기본 선택지가 되었습니다.

네트워크에는 비선형성이 없는 은닉층 하나가 있습니다. 입력은 vocabulary 전체에 대한 one-hot vector입니다. 출력은 vocabulary 전체에 대한 softmax입니다. 학습이 끝나면 출력층은 버립니다. 은닉층 가중치가 embedding입니다.

```text
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          이것이 embedding
```

핵심 요령: 10만 단어에 대해 softmax를 계산하는 것은 지나치게 비쌉니다. Word2Vec은 **negative sampling**으로 이를 binary classification 문제로 바꿉니다. "이 context word가 이 center word 근처에 실제로 등장했는가, 예 또는 아니오"를 예측합니다. 전체 vocabulary에 softmax를 계산하는 대신, 각 학습 쌍마다 소수의 negative(함께 등장하지 않은) 단어만 샘플링합니다.

```figure
word-vector-arithmetic
```

## 직접 만들기

### 1단계: corpus에서 학습 쌍 만들기

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

window 안의 모든 `(center, context)` 쌍은 positive training example입니다.

### 2단계: embedding table

두 개의 행렬이 있습니다. `W`는 center-word embedding table입니다(나중에 보관할 것). `W'`는 context-word table입니다(대개 버리지만, 때로는 `W`와 평균냅니다).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

작은 random initialization입니다. Vocab size 10k와 dim 100은 현실적인 설정이고, 교육용으로는 50 vocab x 16 dim만으로도 기하 구조를 볼 수 있습니다.

### 3단계: negative sampling objective

각 positive pair `(center, context)`마다 vocabulary에서 임의 단어 `k`개를 negative로 샘플링합니다. positive에서는 dot product `W[center] · W'[context]`가 높아지고, negative에서는 낮아지도록 모델을 학습합니다.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

마법 같은 공식은 이것입니다. positive pair에는 sigmoid가 1에 가깝기를 원하는 logistic loss를, negative pair에는 sigmoid가 0에 가깝기를 원하는 logistic loss를 더합니다. Gradient는 두 table 모두로 흐릅니다. 전체 유도는 원 논문에 있습니다. 오래 기억하고 싶다면 연필과 종이로 한 번 따라가 보세요.

### 4단계: toy corpus에서 학습하기

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

큰 corpus에서 충분한 epoch를 돌리면 문맥을 공유하는 단어들이 비슷한 center embedding을 갖습니다. toy corpus에서는 그 효과가 희미하게 보입니다. 수십억 토큰에서는 극적으로 보입니다.

### 5단계: analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

pre-trained 300d Google News vector에서는 다음과 같습니다.

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`입니다. 모델이 왕족이 무엇인지 알아서가 아닙니다. `(king - man)` vector가 "royal" 비슷한 것을 포착하고, 이것을 `woman`에 더하면 royal-female 영역 근처에 도착하기 때문입니다.

## 사용하기

Word2Vec을 직접 작성하는 일은 교육입니다. Production NLP에서는 `gensim`을 사용합니다.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

실제 작업에서는 Word2Vec을 직접 학습하는 일이 거의 없습니다. pre-trained vector를 내려받습니다.

- **GloVe** — Stanford의 co-occurrence-matrix factorization 접근입니다. 50d, 100d, 200d, 300d checkpoint가 있습니다. 일반적인 coverage가 좋습니다. Lesson 04에서 GloVe를 구체적으로 다룹니다.
- **fastText** — character n-gram을 embedding하는 Facebook의 Word2Vec 확장입니다. subword를 조합해 out-of-vocabulary word를 처리합니다. Lesson 04.
- **Pretrained Word2Vec on Google News** — 300d, 3M word vocabulary, 2013년 공개. 지금도 매일 다운로드됩니다.

### 2026년에도 Word2Vec이 여전히 이기는 경우

- 가벼운 domain-specific retrieval. 노트북에서 medical abstract로 한 시간 학습하면 일반 모델이 포착하지 못하는 specialized vector를 얻습니다.
- analogy-style feature engineering. `gender_vector = mean(man - woman pairs)`. 이 vector를 다른 단어에서 빼면 gender-neutral axis를 얻습니다. Fairness research에서 여전히 사용됩니다.
- Interpretability. 100d는 PCA나 t-SNE로 plot해서 cluster가 생기는 모습을 실제로 볼 만큼 작습니다.
- GPU 없이 on-device로 inference가 돌아가야 하는 곳. Word2Vec lookup은 row 하나를 가져오는 작업입니다.

### Word2Vec이 실패하는 곳

polysemy의 벽입니다. `bank`에는 vector가 하나뿐입니다. `river bank`와 `financial bank`가 그것을 공유합니다. `table`(spreadsheet vs. furniture)도 공유합니다. downstream classifier는 vector만 보고 sense를 구분할 수 없습니다.

Contextual embedding(ELMo, BERT, 그리고 이후 모든 transformer)은 주변 context를 바탕으로 단어의 각 occurrence마다 다른 vector를 만들어 이 문제를 해결했습니다. 이것이 Word2Vec에서 BERT로 넘어가는 도약입니다. static에서 contextual로의 이동입니다. Phase 7에서 transformer 쪽 절반을 다룹니다.

out-of-vocabulary 문제는 또 다른 실패입니다. 학습 데이터에 없었다면 Word2Vec은 `Zoomer-approved`를 본 적이 없습니다. fallback이 없습니다. fastText는 subword composition으로 이를 고칩니다(Lesson 04).

## 내보내기

`outputs/skill-embedding-probe.md`로 저장합니다.

```markdown
---
name: embedding-probe
description: word2vec model을 점검합니다. analogy를 실행하고, neighbor를 찾고, quality를 진단합니다.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

학습된 word embedding이 제대로 작동하는지 확인합니다. `gensim.models.KeyedVectors` object와 vocabulary가 주어지면 다음을 실행합니다.

1. 세 가지 canonical analogy test. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. top-1 결과와 cosine을 보고합니다.
2. 사용자가 제공한 domain-specific word에 대해 다섯 가지 nearest-neighbor test. top-5 neighbor와 cosine을 출력합니다.
3. symmetry check 하나. float precision 범위 안에서 `similarity(a, b) == similarity(b, a)`인지 확인합니다.
4. degenerate check 하나. embedding의 norm이 0.01보다 작거나 100보다 크면 모델에 training bug가 있습니다. 표시합니다.

analogy accuracy만으로 모델이 좋다고 선언하지 않습니다. Analogy benchmark는 쉽게 맞춤형으로 조작될 수 있고 downstream task로 전이되지 않습니다. intrinsic evaluation과 downstream evaluation을 함께 권장합니다.
```

## 연습 문제

1. **Easy.** 작은 corpus(고양이와 개에 관한 문장 20개)에서 training loop를 실행하세요. 200 epoch 후 `nearest(vocab, W, W[vocab["cat"]])`의 top 3 안에 `dog`가 들어오는지 확인하세요. 아니라면 epoch나 vocabulary를 늘리세요.
2. **Medium.** frequent word subsampling을 추가하세요. frequency가 `10^-5`보다 큰 단어는 frequency에 비례한 확률로 training pair에서 제거합니다. rare-word similarity에 미치는 영향을 측정하세요.
3. **Hard.** 20 Newsgroups corpus에서 모델을 학습하세요. 두 bias axis인 `he - she`와 `doctor - nurse`를 계산하세요. occupation word를 두 axis에 project하세요. 어떤 occupation이 가장 큰 bias gap을 갖는지 보고하세요. 이것이 fairness researcher가 사용하는 probe의 한 종류입니다.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Word embedding | 단어를 vector로 표현 | context에서 학습한 dense, low-dim(보통 100-300) representation. |
| Skip-gram | Word2Vec trick | center word에서 context word를 예측합니다. CBOW보다 느리지만 rare word에 더 좋습니다. |
| Negative sampling | 학습 shortcut | full vocab에 대한 softmax를 `k`개의 random word와 맞서는 binary classification으로 바꿉니다. |
| Static embedding | 단어마다 vector 하나 | context와 무관하게 같은 vector입니다. polysemy에서 실패합니다. |
| Contextual embedding | context-sensitive vector | 주변 단어를 바탕으로 각 occurrence마다 다른 vector입니다. transformer가 만드는 것입니다. |
| OOV | Out of vocabulary | training에서 보지 못한 단어입니다. Word2Vec은 이런 단어의 vector를 만들 수 없습니다. |

## 더 읽을거리

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — negative-sampling 논문입니다. 짧고 읽기 쉽습니다.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 원 논문의 수식이 빽빽하게 느껴진다면 gradient를 가장 명확하게 유도한 글입니다.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 실제로 동작하는 production training setting입니다.
