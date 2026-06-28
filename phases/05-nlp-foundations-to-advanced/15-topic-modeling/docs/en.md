# 토픽 모델링 — LDA와 BERTopic

> LDA: 문서는 토픽의 혼합이고, 토픽은 단어 위의 분포입니다. BERTopic: 문서는 임베딩 공간에서 클러스터링되고, 클러스터가 토픽입니다. 목표는 같지만 분해 방식이 다릅니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## 문제

고객 지원 티켓 10,000개, 뉴스 기사 50,000개, 트윗 200,000개가 있다고 해봅시다. 이 모음을 읽지 않고 무엇에 관한 것인지 알아야 합니다. 라벨이 붙은 category는 없습니다. category가 몇 개 있는지도 모릅니다.

토픽 모델링(topic modeling)은 이 질문에 비지도 방식으로 답합니다. corpus를 입력하면 일관된 소수의 topic 집합과 각 document별 topic 분포를 돌려받습니다.

두 알고리즘 계열이 주로 쓰입니다. LDA(2003)는 각 document를 latent topic의 혼합으로, 각 topic을 단어 위의 분포로 봅니다. inference는 Bayesian 방식입니다. mixed-membership topic assignment와 설명 가능한 단어 수준 probability distribution이 필요할 때 여전히 production에서 사용됩니다.

BERTopic(2020)은 BERT로 document를 encode하고, UMAP으로 dimensionality를 줄이고, HDBSCAN으로 clustering한 뒤, class-based TF-IDF로 topic words를 추출합니다. short text, social media, 그리고 word overlap보다 semantic similarity가 더 중요한 경우에 강합니다. document 하나가 topic 하나만 받는다는 점은 긴 글에서는 한계입니다.

이 레슨에서는 두 방식의 직관을 만들고, 주어진 corpus에 어떤 것을 선택해야 하는지 정리합니다.

## 개념

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA generative story.** 각 topic은 단어 위의 분포입니다. 각 document는 topic의 혼합입니다. document 안의 단어를 생성하려면 document의 mixture에서 topic을 하나 sample한 다음, 그 topic의 distribution에서 word를 하나 sample합니다. inference는 이 과정을 거꾸로 풉니다. 관측된 words가 주어졌을 때 document별 topic distribution과 topic별 word distribution을 추론합니다. Collapsed Gibbs sampling이나 variational Bayes가 이 수학을 수행합니다.

핵심 LDA 출력:

- `doc_topic`: matrix `(n_docs, n_topics)`, 각 row의 합은 1입니다(document의 topic mixture).
- `topic_word`: matrix `(n_topics, vocab_size)`, 각 row의 합은 1입니다(topic의 word distribution).

**BERTopic pipeline.**

1. sentence transformer(예: `all-MiniLM-L6-v2`)로 각 document를 encode합니다. 384-dim vector입니다.
2. UMAP으로 dimensionality를 약 5차원까지 줄입니다. BERT embeddings는 clustering하기에 차원이 너무 높습니다.
3. HDBSCAN으로 cluster를 만듭니다. density-based 방식이며, 크기가 다양한 cluster와 "outlier" label을 만듭니다.
4. 각 cluster에 대해 cluster의 documents 위에서 class-based TF-IDF를 계산하여 top words를 추출합니다.

출력은 document당 topic 하나입니다(`-1` outlier label 포함). 선택적으로 HDBSCAN의 probability vector를 통해 soft membership을 얻을 수 있습니다.

## 직접 만들기

### 1단계: scikit-learn으로 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

주의할 점: stopwords를 제거하고, `min_df`와 `max_df`로 너무 드문 term과 어디에나 나오는 term을 거르며, LDA는 raw counts를 기대하므로 TfidfVectorizer가 아니라 CountVectorizer를 사용합니다.

### 2단계: BERTopic(production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 필터는 BERTopic의 outlier bucket, 즉 HDBSCAN이 cluster에 넣지 못한 documents를 제거합니다. `min_topic_size`는 HDBSCAN의 최소 cluster size를 제어합니다. BERTopic library default는 10입니다. 이 예제에서는 레슨 규모에 맞춰 15로 명시합니다. 10,000개가 넘는 documents가 있는 corpus라면 50이나 100으로 올리세요.

### 3단계: evaluation

두 방법 모두 topic words를 출력합니다. 중요한 질문은 그 단어들이 서로 잘 응집되는지입니다.

- **Topic coherence (c_v).** sliding-window context에서 top-word pair의 NPMI(normalized pointwise mutual information)를 결합하고, score를 topic vector로 aggregate한 뒤 cosine similarity로 비교합니다. 높을수록 좋습니다. `gensim.models.CoherenceModel`을 `coherence="c_v"`로 사용하세요.
- **Topic diversity.** 모든 topic의 top words 중 unique words의 비율입니다. 높을수록 좋습니다(topic들이 서로 겹치지 않음).
- **Qualitative inspection.** 각 topic의 top words를 읽습니다. 실제 대상을 가리키고 있나요? human judgment는 여전히 마지막 방어선입니다.

## 무엇을 언제 고를까

| Situation | Pick |
|-----------|------|
| Short text (tweets, reviews, headlines) | BERTopic |
| Long documents with topic mixtures | LDA |
| No GPU / limited compute | LDA or NMF |
| Need document-level multi-topic distributions | LDA |
| LLM integration for topic labeling | BERTopic (direct support) |
| Resource-constrained edge deployment | LDA |
| Max semantic coherence | BERTopic |

가장 큰 실무 고려사항은 document length입니다. BERT embeddings는 truncate됩니다. LDA counts는 길이에 관계없이 작동합니다. embedding model의 context보다 긴 documents라면 chunk + aggregate를 하거나 LDA를 사용하세요.

## 활용하기

2026년 기준 stack:

- **BERTopic.** short text와 semantics가 중요한 모든 경우의 default입니다.
- **`gensim.models.LdaModel`.** production용 classic LDA입니다. 성숙하고 검증되었습니다.
- **`sklearn.decomposition.LatentDirichletAllocation`.** experiment에 쓰기 쉬운 LDA입니다.
- **NMF.** Non-negative matrix factorization. LDA의 빠른 대안이며 short text에서 비슷한 품질을 냅니다.
- **Top2Vec.** BERTopic과 비슷한 설계입니다. community는 더 작지만 일부 benchmark에서 좋습니다.
- **FASTopic.** 더 최신 방식이며, 아주 큰 corpora에서 BERTopic보다 빠릅니다.
- **LLM-based labeling.** 어떤 clustering이든 실행한 다음 model에 각 cluster 이름을 붙이도록 prompt합니다.

## 결과물 만들기

`outputs/skill-topic-picker.md`로 저장하세요:

```markdown
---
name: topic-picker
description: corpus에 맞는 LDA 또는 BERTopic을 고릅니다. library, knobs, evaluation을 지정합니다.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

corpus 설명(document count, avg length, domain, language, compute budget)이 주어지면 다음을 출력하세요:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. 한 문장짜리 이유.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, 40,000개 미만 docs의 corpus에서는 200으로 clamp합니다. corpus가 정말 큰 경우(>40k)에만 >200을 허용하고 compute cost 증가를 언급하세요. `min_df` / `max_df` filters와 neural approaches용 embedding model도 여기에 포함합니다.
3. Evaluation. `gensim.models.CoherenceModel`을 통한 topic coherence (c_v), topic diversity, 20-sample human read.
4. Failure mode to probe. LDA에서는 stopwords와 frequent terms를 흡수하는 "junk topics"입니다. BERTopic에서는 ambiguous documents를 삼키는 -1 outlier cluster입니다.

chunking strategy 없이 embedding model의 context window보다 긴 documents에 BERTopic을 쓰는 것은 거절하세요. coherence가 무너지므로 매우 짧은 text(tweets, 10 tokens 미만 reviews)에 LDA를 쓰는 것도 거절하세요. 5 미만의 `n_topics` 선택은 likely wrong으로 flag하세요. 40k docs 미만 corpora에서 >200은 likely over-splitting으로 flag하세요.
```

## 연습문제

1. **Easy.** 20 Newsgroups dataset에 5개 topic으로 LDA를 fit하세요. topic별 top 10 words를 출력하세요. 각 topic에 손으로 label을 붙이세요. 알고리즘이 실제 categories를 찾았나요?
2. **Medium.** 같은 20 Newsgroups subset에 BERTopic을 fit하세요. 찾은 topic 수, top words, qualitative coherence를 LDA와 비교하세요. 어느 쪽이 실제 categories를 더 깔끔하게 드러내나요?
3. **Hard.** 여러분의 corpus에서 LDA와 BERTopic 양쪽의 c_v coherence를 계산하세요. 각 방법을 5, 10, 20, 50 topics로 실행하세요. coherence vs topic count를 plot하세요. topic count 전반에서 어느 방법이 더 안정적인지 보고하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Topic | corpus가 다루는 대상 | words 위의 probability distribution(LDA)이거나 similar documents의 cluster(BERTopic)입니다. |
| Mixed membership | doc이 여러 topic임 | LDA는 각 document에 모든 topics 위의 distribution을 할당합니다. |
| UMAP | Dimensionality reduction | local structure를 보존하는 manifold learning이며 BERTopic에서 사용됩니다. |
| HDBSCAN | Density clustering | variable-size clusters를 찾고, outlier에 대해 "noise" label(-1)을 생성합니다. |
| c_v coherence | Topic quality metric | sliding windows 안에서 top topic words의 average pointwise mutual information입니다. |

## 더 읽을거리

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA paper.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic paper.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — c_v와 관련 metric을 소개한 paper.
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — production reference. 예제가 훌륭합니다.
