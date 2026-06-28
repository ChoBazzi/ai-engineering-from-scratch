# 정보 검색과 Search

> BM25는 precise하지만 brittle합니다. Dense는 넓게 찾지만 keyword를 놓칩니다. Hybrid가 2026년의 default입니다. 나머지는 tuning입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 문제

사용자가 "what happens if someone lies to get money"라고 입력하면 실제로 그것을 다루는 statute인 "Section 420 IPC."를 찾기를 기대합니다. Keyword search는 공유 vocabulary가 없어서 완전히 놓칩니다. Semantic search도 embedding이 legal text로 train되지 않았다면 놓칠 수 있습니다. Real search는 둘 다 처리해야 합니다.

IR은 모든 RAG system, 모든 search bar, 모든 docs site의 fuzzy lookup 아래에 있는 pipeline입니다. Production에서 동작하는 2026 architecture는 단일 method가 아닙니다. 서로의 failure를 잡아 주는 complementary method의 chain입니다.

이 lesson은 각 piece를 만들고, 각 piece가 어떤 failure를 잡는지 이름 붙입니다.

## 개념

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

네 layer입니다. 필요한 것을 고르세요.

1. **Sparse retrieval (BM25).** 빠르고 exact match에는 precise하지만 semantics에는 약합니다. Inverted index 위에서 실행됩니다. 수백만 document에서도 query당 sub-10ms입니다. Statute reference, product code, error message, named entity를 잘 잡습니다.
2. **Dense retrieval.** Query와 document를 vector로 encode합니다. Nearest neighbor search입니다. Paraphrase와 semantic similarity를 잡습니다. 한 글자 차이의 exact keyword match는 놓칠 수 있습니다. FAISS 또는 vector DB를 쓰면 query당 50-200ms입니다.
3. **Fusion.** Sparse와 dense의 ranked list를 merge합니다. Reciprocal Rank Fusion (RRF)은 raw score(서로 scale이 다름)를 무시하고 rank position만 쓰기 때문에 쉬운 default입니다. Domain에서 한 signal이 지배적이라는 것을 알 때는 weighted fusion도 option입니다.
4. **Cross-encoder rerank.** Fusion의 top-30을 가져옵니다. Cross-encoder(query + document를 함께 넣어 pair마다 score)를 실행합니다. Top-5를 유지합니다. Cross-encoder는 bi-encoder보다 pair당 느리지만 훨씬 정확합니다. Top-30에만 실행해서 amortize합니다.

Three-way retrieval(BM25 + dense + SPLADE 같은 learned-sparse)은 2026 benchmark에서 two-way보다 우수하지만 learned-sparse index infrastructure가 필요합니다. 대부분의 team에는 two-way plus cross-encoder rerank가 sweet spot입니다.

## 직접 만들기

### 1단계: scratch에서 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

알아둘 만한 parameter는 두 개입니다. `k1=1.5`는 term-frequency saturation을 control합니다. 높을수록 term repetition에 더 많은 weight를 둡니다. `b=0.75`는 length normalization을 control합니다. 0은 document length를 무시하고, 1은 완전히 normalize합니다. Default는 original paper의 Robertson recommendation이며 거의 tuning할 필요가 없습니다.

### 2단계: bi-encoder로 dense retrieval

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

Dot product가 cosine과 같아지도록 embedding을 L2-normalize하세요. `all-MiniLM-L6-v2`는 384-dim이고 빠르며 대부분의 English retrieval에 충분히 강합니다. Multilingual work에는 `paraphrase-multilingual-MiniLM-L12-v2`를 사용하세요. 최고 accuracy가 필요하면 `bge-large-en-v1.5` 또는 `e5-large-v2`를 사용합니다.

### 3단계: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` constant는 original RRF paper에서 왔습니다. 더 높은 `k`는 rank difference의 contribution을 flatten합니다. 더 낮은 `k`는 top rank를 지배적으로 만듭니다. 60은 published default이며 거의 tuning할 필요가 없습니다.

### 4단계: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

세 stage가 compose되었습니다. BM25는 lexical match를 찾습니다. Dense는 semantic match를 찾습니다. RRF는 score calibration 없이 두 ranking을 merge합니다. Cross-encoder는 top-30을 query-document pair로 함께 rescore해서 bi-encoder가 놓친 fine-grained relevance를 잡습니다. Top-5를 유지합니다.

### 5단계: evaluation

| Metric | Meaning |
|--------|---------|
| Recall@k | Correct document가 존재하는 query 중 top-k에 들어오는 비율은 얼마인가요? |
| MRR (Mean Reciprocal Rank) | First relevant document의 1/rank 평균입니다. |
| nDCG@k | Binary relevant/not뿐 아니라 relevance gradation까지 반영합니다. |

RAG에서는 retriever의 **Recall@k**가 가장 중요한 숫자입니다. Right passage가 retrieved set에 없으면 reader는 answer할 수 없습니다.

Debugging tip: 실패한 query에서는 sparse ranking과 dense ranking을 diff하세요. 하나는 right document를 찾고 다른 하나는 찾지 못한다면 vocabulary mismatch(fix: missing half 추가) 또는 semantic ambiguity(fix: better embedding 또는 reranker)입니다.

## 활용하기

2026년 stack입니다.

| Scale | Stack |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF. Separate DB 없음. |
| 100k-10M docs | Dense에는 FAISS 또는 pgvector + BM25에는 Elasticsearch / OpenSearch. Parallel 실행. |
| 10M+ docs | Hybrid support가 있는 Qdrant / Weaviate / Vespa / Milvus. Top-30 위에 cross-encoder rerank. |
| Best-quality frontier | Three-way(BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

무엇을 고르든 evaluation budget을 잡으세요. End-to-end RAG accuracy를 benchmark하기 전에 retrieval recall을 benchmark하세요. Reader는 retriever가 놓친 것을 고칠 수 없습니다.

### 2026 production RAG에서 얻은 hard-won lesson

- **RAG failure의 80%는 model이 아니라 ingestion과 chunking에서 옵니다.** Team은 LLM을 바꾸고 prompt를 tuning하느라 몇 주를 쓰지만, retrieval은 세 query마다 wrong context를 조용히 반환합니다. Chunking을 먼저 고치세요.
- **Chunking strategy는 chunk size보다 중요합니다.** Fixed-size split은 table, code, nested header를 깨뜨립니다. Sentence-aware가 default입니다. Technical docs와 product manual에는 semantic 또는 LLM-based chunking이 payoff를 냅니다.
- **Parent-doc pattern.** Precision을 위해 작은 "child" chunk를 retrieve합니다. 같은 parent section에서 여러 child가 나타나면 context를 보존하기 위해 parent block으로 swap합니다. 이 방식은 retraining 없이 answer quality를 꾸준히 올립니다.
- **k_rerank=3이 보통 optimal입니다.** 그 이후의 extra chunk는 answer quality를 올리지 못하면서 token cost와 generation latency만 추가합니다. k=8이 여전히 k=3보다 낫다면 reranker가 underperforming 중입니다.
- **HyDE / query expansion.** Query에서 hypothetical answer를 generate하고, 그것을 embed해서 retrieve합니다. Short question과 long document 사이의 phrasing gap을 이어 줍니다. Training 없이 precision lift를 얻습니다.
- **Context budget은 8K token 아래로 유지하세요.** 그 limit에서 계속 hit한다면 reranker threshold가 너무 느슨하다는 뜻입니다.
- **모든 것을 version하세요.** Prompt, chunking rule, embedding model, reranker. Drift는 answer quality를 조용히 망가뜨립니다. Faithfulness, context precision, unanswered-question rate에 대한 CI gate가 user가 보기 전에 regression을 막습니다.
- **Three-way retrieval(BM25 + dense + SPLADE 같은 learned-sparse)은 2026 benchmark에서 two-way보다 우수합니다.** 특히 proper noun과 semantics가 섞인 query에서 그렇습니다. Infrastructure가 SPLADE index를 support하면 ship하세요.

적절한 retrieval design은 2026 industry measurement 기준 hallucination을 70-90% 줄입니다. 대부분의 RAG performance gain은 model fine-tuning이 아니라 더 나은 retrieval에서 옵니다.

## 출시하기

`outputs/skill-retrieval-picker.md`로 저장하세요.

```markdown
---
name: retrieval-picker
description: 주어진 corpus와 query pattern에 맞는 retrieval stack을 고릅니다.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Requirement(corpus size, query pattern, latency budget, quality bar, infra constraints)가 주어지면 다음을 출력합니다.

1. Stack. BM25 only, dense only, hybrid(BM25 + dense + RRF), hybrid + cross-encoder rerank, 또는 three-way(BM25 + dense + learned-sparse).
2. Dense encoder. Specific model 이름. Language, domain, context length에 맞춥니다.
3. Reranker. 사용한다면 specific cross-encoder model 이름. Rerank가 top-30에 30-100ms latency를 추가한다고 flag합니다.
4. Evaluation plan. Recall@10이 primary retriever metric입니다. Multi-answer에는 MRR. 먼저 baseline을 세우고 incremental improvement를 그것과 비교해 측정합니다.

Corpus에 named entity, error code, product SKU가 있는데 dense가 exact match를 처리한다는 evidence가 없다면 dense-only 추천을 거부합니다. Final top-5가 user의 answer를 결정하는 high-stakes retrieval(legal, medical)에서는 reranking 생략을 거부합니다.
```

## 연습문제

1. **Easy.** 500-document corpus에서 위 `hybrid_search`를 구현하세요. Query 20개로 test하세요. BM25-only, dense-only, hybrid 사이의 recall at 5를 비교합니다.
2. **Medium.** MRR 계산을 추가하세요. Correct document가 알려진 각 test query에 대해 BM25, dense, hybrid ranking에서 correct doc의 rank를 찾습니다. 각각의 MRR을 report하세요.
3. **Hard.** Sentence Transformers의 MultipleNegativesRankingLoss를 사용해 domain에 맞게 dense encoder를 fine-tune하세요. Query-document pair 500개로 training set을 만듭니다. Fine-tune 전후 recall을 비교하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| BM25 | Keyword search | Okapi BM25. Term frequency, IDF, length로 document를 score합니다. |
| Dense retrieval | Vector search | Query + doc을 vector로 encode하고 nearest neighbor를 찾습니다. |
| Bi-encoder | Embedding model | Query와 doc을 독립적으로 encode합니다. Query time에 빠릅니다. |
| Cross-encoder | Reranker model | Query + doc을 함께 encode합니다. 느리지만 정확합니다. |
| RRF | Rank fusion | `1/(k + rank)`를 sum해 두 ranking을 결합합니다. |
| Recall@k | Retrieval metric | Relevant doc이 top-k에 들어 있는 query의 fraction입니다. |

## 더 읽을거리

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) - definitive BM25 treatment입니다.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - canonical bi-encoder인 DPR입니다.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) - dense와의 gap을 좁히는 learned-sparse retriever입니다.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - RRF paper입니다.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) - late-interaction retrieval입니다.
