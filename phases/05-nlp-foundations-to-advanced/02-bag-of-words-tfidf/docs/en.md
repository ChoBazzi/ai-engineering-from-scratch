# Bag of Words, TF-IDF, 텍스트 표현

> 먼저 세고, 나중에 생각하라. 2026년에도 TF-IDF는 잘 정의된 task에서 embedding을 이긴다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 minutes

## 문제

모델에는 숫자가 필요하다. 당신에게는 문자열이 있다.

모든 NLP pipeline은 같은 질문에 답해야 한다. 가변 길이 token stream을 classifier가 소비할 수 있는 고정 크기 vector로 어떻게 바꿀 것인가. 이 분야가 처음 도달한 답은 작동하는 것 중 가장 단순한 답이었다. 단어를 세라. vector를 만들라.

그 vector는 어떤 embedding model보다 더 많은 production NLP를 떠받쳐 왔다. Spam filter, topic classifier, log anomaly detection, search ranking(BM25 이전), sentiment analysis의 첫 물결, 학술 NLP benchmark의 첫 10년. 2026년 practitioner들도 좁은 classification task에서는 여전히 이것을 먼저 집어 든다. 빠르고, 해석 가능하며, 단어 존재 여부가 중요한 task에서는 400M-parameter embedding model과 구분하기 어려울 때가 많다.

이 lesson은 bag of words를 만든 다음 TF-IDF를 처음부터 만든다. 그 다음 scikit-learn이 같은 일을 세 줄로 수행하는 모습을 보여 준다. 마지막으로 embedding이 필요해지는 실패 모드를 이름 붙인다.

## 개념

**Bag of Words (BoW)**는 순서를 버린다. 각 document마다 각 vocabulary word가 몇 번 등장하는지 센다. Vector length는 vocabulary size다. 위치 `i`는 단어 `i`의 count다.

**TF-IDF**는 BoW에 다시 가중치를 준다. 모든 document에 등장하는 단어는 정보량이 낮으므로 낮춘다. corpus 전체에서는 드물지만 한 document에서는 자주 등장하는 단어는 signal이므로 높인다.

```text
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

여기서 `TF`는 document 안의 term frequency이고, `df`는 document frequency(그 단어를 포함하는 doc 수), `N`은 전체 document 수다. `log`는 어디에나 나오는 단어의 weight가 과도해지지 않게 한다.

핵심 속성: 둘 다 해석 가능한 축을 가진 sparse vector를 만든다. 학습된 classifier의 weight를 보고 어떤 단어가 document를 각 class 쪽으로 밀어 주는지 읽을 수 있다. 768-dimensional BERT embedding에서는 이렇게 할 수 없다.

```figure
bow-tfidf
```

## 직접 만들기

### 1단계: vocabulary 만들기

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Input: tokenized document의 list(어떤 word-level tokenizer든 된다. 이 lesson의 `code/main.py`는 단순화한 lowercase variant를 사용한다). Output: `{word: index}` dict. 안정적인 insertion order 덕분에 word index 0은 첫 번째 document에서 처음 본 단어다. 관례는 다양하다. scikit-learn은 alphabetically sort한다.

### 2단계: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

행은 document다. 열은 vocabulary index다. Entry `[i][j]`는 "document `i`에 word `j`가 몇 번 등장하는가"다. Doc 1에는 `cat`이 실제로 두 번 있었기 때문에 두 번이다. Doc 0에는 `ran`이 없었기 때문에 0이다.

### 3단계: term frequency와 document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

이름 붙일 만한 smoothing trick이 두 가지 있다. `(n+1)/(d+1)`은 `log(x/0)`을 피한다. 뒤의 `+1`은 모든 document에 등장하는 단어도 IDF 1(0이 아님)을 갖게 하며, scikit-learn의 default와 일치한다. 다른 구현은 raw `log(N/df)`를 쓴다. 둘 다 작동하지만 smoothed version이 더 친절하다.

### 4단계: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

세 document, 다섯 vocabulary word(`the`, `cat`, `sat`, `dog`, `ran`)가 있다. `the`는 셋 모두에 등장하므로 IDF가 낮다. `dog`는 하나에만 등장하므로 IDF가 높다. Vector는 sparse하고(대부분 entry가 작다) 구분력 있는 단어가 튀어나온다.

### 5단계: 행 L2-normalize하기

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Normalization이 없으면 긴 document가 더 큰 vector를 얻고 similarity score를 지배한다. L2 normalization은 모든 document를 unit hypersphere 위에 올려놓는다. 이제 행 사이의 cosine similarity는 dot product일 뿐이다.

## 사용하기

scikit-learn은 production version을 제공한다.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer`는 tokenization, vocabulary, BoW를 한 번에 수행한다. `TfidfVectorizer`는 IDF weighting과 L2 normalization을 추가한다. 둘 다 sparse matrix를 반환한다. 100k document에서는 dense version이 memory에 들어가지 않는다. classifier가 dense를 요구하기 전까지는 sparse로 유지한다.

모든 것을 바꾸는 knob:

| Arg | 효과 |
|-----|--------|
| `ngram_range=(1, 2)` | bigram을 포함한다. 보통 classification을 끌어올린다. |
| `min_df=2` | 2개 미만의 doc에 등장하는 단어를 버린다. noisy data에서 vocabulary를 줄인다. |
| `max_df=0.95` | 95% 초과의 doc에 등장하는 단어를 버린다. hardcoded list 없이 stopword removal에 가깝게 동작한다. |
| `stop_words="english"` | scikit-learn의 builtin stopword list. task-dependent다. sentiment analysis에서는 negation을 *제거하면 안 된다*. |
| `sublinear_tf=True` | raw `tf` 대신 `1 + log(tf)`를 사용한다. 한 doc에서 term이 여러 번 반복될 때 도움이 된다. |

### TF-IDF가 여전히 이기는 경우(2026년 기준)

- Spam detection, topic labeling, log anomaly flagging. 중요한 것은 word presence이지 semantic nuance가 아니다.
- Low-data regime(labeled example 수백 개). TF-IDF plus logistic regression은 pretraining cost가 없다.
- Latency가 중요한 곳. TF-IDF plus linear model은 microsecond 단위로 답한다. transformer로 document를 embedding하면 10-100ms가 걸린다.
- 예측을 설명해야 하는 system. classifier의 coefficient를 inspect한다. top positive word가 이유다.

### TF-IDF가 실패하는 경우

Semantic blindness failure. 다음 두 document를 보라:

- "The movie was not good at all."
- "The movie was excellent."

하나는 부정 review다. 하나는 긍정이다. 둘의 TF-IDF overlap은 정확히 `{the, movie, was}`다. bag-of-words classifier는 `good` 근처의 `not`이 label을 뒤집는다는 사실을 외워야 한다. 충분한 data가 있으면 배울 수 있지만, syntax를 이해하는 model만큼 우아하게 하지는 못한다.

다른 실패는 inference 시점의 out-of-vocabulary word다. IMDb review로 학습한 BoW model은 training에 없던 `Zoomer-approved` token이 들어오면 무엇을 해야 할지 모른다. Subword embedding(lesson 04)은 이를 처리한다. TF-IDF는 할 수 없다.

### Hybrid: TF-IDF weighted embeddings

2026년 medium-data classification의 실용적 default: TF-IDF weight를 word embedding 위의 attention처럼 사용한다.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Embedding에서 semantic capacity를 얻고, TF-IDF에서 rare-word emphasis를 얻는다. Classifier는 pooled vector 위에서 학습한다. 약 50k labeled example 아래의 sentiment, topic, intent classification에서는 둘 중 하나만 쓴 것보다 더 잘 작동한다.

## 내보내기

`outputs/prompt-vectorization-picker.md`로 저장한다:

```markdown
---
name: vectorization-picker
description: text-classification task가 주어지면 BoW, TF-IDF, embeddings, 또는 hybrid를 추천한다.
phase: 5
lesson: 02
---

당신은 text-vectorization strategy를 추천한다. task description이 주어지면 다음을 출력한다:

1. Representation(BoW, TF-IDF, transformer embeddings, 또는 hybrid). 한 문장으로 이유를 설명한다.
2. 구체적인 vectorizer configuration. library 이름을 말한다. argument(`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`)를 인용한다.
3. shipping 전에 test할 failure mode 하나.

사용자의 labeled example이 500개 미만이면 TF-IDF baseline에서 semantic failure의 증거를 보이지 않는 한 embeddings를 추천하지 않는다. Sentiment analysis에서는 stopword 제거를 추천하지 않는다(negation은 signal을 가진다). Class imbalance는 vectorizer 변경만으로 해결되지 않는다고 표시한다.

예시 입력: "30k customer support ticket을 12개 category로 classify한다. 대부분 ticket은 2-3문장이다. English only. audit log를 위해 explainability가 필요하다."

예시 출력:

- Representation: TF-IDF. 30k example은 작지 않으며, explainability requirement 때문에 dense embedding은 제외된다.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. category keyword가 때로 stopword이므로 stopword를 유지한다("not working" vs "working").
- Failure to test: `min_df=3`이 rare category keyword를 drop하지 않는지 확인한다. class별로 filtering한 `get_feature_names_out`을 실행하고 눈으로 확인한다.
```

## 연습문제

1. **Easy.** L2-normalized TF-IDF output 위에 `cosine_similarity(doc_vec_a, doc_vec_b)`를 구현하라. 동일한 document는 1.0, vocabulary가 겹치지 않는 document는 0.0을 내는지 확인하라.
2. **Medium.** `bag_of_words`에 `n-gram` support를 추가하라. Parameter `n`은 `n`-gram count를 만든다. `["the", "cat", "sat"]`에 `n=2`를 적용하면 `["the cat", "cat sat"]`의 bigram count가 만들어지는지 test하라.
3. **Hard.** 위의 TF-IDF-weighted-embedding hybrid를 GloVe 100d vector로 구현하라(한 번 download하고 cache). 20 Newsgroups dataset에서 plain TF-IDF 및 plain mean-pooled embeddings와 classification accuracy를 비교하라. 어디에서 무엇이 이기는지 report하라.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-----------------|-----------------------|
| BoW | Word frequency vector | 한 document 안 vocabulary word의 count. 순서는 버린다. |
| TF | Term frequency | document 안 단어의 count. 선택적으로 document length로 normalize한다. |
| DF | Document frequency | 단어를 한 번 이상 포함하는 document 수. |
| IDF | Inverse document frequency | smoothed `log(N / df)`. 어디에나 등장하는 단어의 weight를 낮춘다. |
| Sparse vector | 대부분 0 | Vocabulary는 보통 10k-100k word이며, 대부분은 특정 document에 없다. |
| Cosine similarity | Vector angle | L2-normalized vector의 dot product. 1은 동일, 0은 orthogonal이다. |

## 더 읽을거리

- [scikit-learn - feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) - canonical API reference와 모든 knob에 대한 note.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) - TF-IDF를 10년 동안 default로 만든 논문.
- ["Why TF-IDF Still Beats Embeddings" - Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) - 낡은 방법이 언제 왜 이기는지에 대한 2026년 관점.
