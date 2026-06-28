# Embedding Models — 2026년 심층 분석

> Word2Vec는 word마다 하나의 vector를 주었습니다. modern embedding model은 passage마다 vector를 주고, cross-lingual이며, sparse, dense, multi-vector view를 제공하고, index에 맞는 크기로 조절할 수 있습니다. 잘못 고르면 RAG가 잘못된 것을 retrieve합니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## 문제

RAG system이 40%의 경우 잘못된 passage를 retrieve합니다. 원인은 vector database나 prompt가 아닌 경우가 많습니다. embedding model입니다.

2026년에 embedding을 선택한다는 것은 다섯 축에서 선택한다는 뜻입니다.

1. **Dense vs sparse vs multi-vector.** passage마다 하나의 vector인지, token마다 하나의 vector인지, 아니면 sparse weighted bag of words인지.
2. **Language coverage.** monolingual English model은 여전히 English-only task에서 이깁니다. corpus가 섞여 있으면 multilingual model이 이깁니다.
3. **Context length.** 512 token vs 8,192 vs 32,768입니다. 실제 effective capacity는 광고된 max의 60-70%인 경우가 많습니다.
4. **Dimension budget.** full precision의 3,072 float = vector당 12 KB입니다. vector 100M개면 storage가 월 $1,300입니다. Matryoshka truncation은 이를 4× 줄입니다.
5. **Open vs hosted.** open-weight는 stack과 data를 직접 control한다는 뜻입니다. hosted는 control을 always-latest와 맞바꾼다는 뜻입니다.

이 lesson은 지난 분기에 인기 있었던 것에 기대지 않고 evidence로 선택할 수 있도록 tradeoff의 이름을 붙입니다.

## 개념

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings.** passage마다 하나의 vector입니다(보통 384-3,072 dimension). Cosine similarity가 semantic proximity로 passage를 rank합니다. OpenAI `text-embedding-3-large`, BGE-M3 dense mode, Voyage-3가 여기에 속합니다. default choice입니다.

**Sparse embeddings.** SPLADE-style입니다. transformer가 모든 vocab token에 대한 weight를 예측한 뒤 대부분을 0으로 만듭니다. 결과는 |vocab| 크기의 sparse vector입니다. BM25처럼 lexical matching을 포착하지만 learned term weight를 사용합니다. keyword-heavy query에 강합니다.

**Multi-vector (late interaction).** ColBERTv2, Jina-ColBERT입니다. token마다 하나의 vector입니다. MaxSim으로 score합니다. 각 query token에 대해 가장 유사한 document token을 찾고 score를 합산합니다. 저장과 scoring 비용은 더 크지만 long query와 domain-specific corpus에서 이깁니다.

**BGE-M3: 세 가지를 한 번에.** 단일 model이 dense, sparse, multi-vector representation을 동시에 출력합니다. 각각 독립적으로 query할 수 있고, score는 weighted sum으로 fuse합니다. 하나의 checkpoint에서 flexibility를 원할 때 2026년 default입니다.

**Matryoshka Representation Learning.** vector의 첫 N dimension만으로도 유용한 standalone embedding이 되도록 train합니다. 1,536-dim vector를 256 dim으로 truncate하면 6× storage saving을 얻고 accuracy는 약 1%만 지불합니다. OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+가 지원합니다.

### MTEB leaderboard는 일부만 말해 줍니다

Massive Text Embedding Benchmark입니다. 출시 시점(2022년)에는 8개 task type에 걸친 56개 task였고, MTEB v2에서는 100+ task로 확장되었습니다. 2026년 초 기준 Gemini Embedding 2가 retrieval에서 1위(67.71 MTEB-R)입니다. Cohere embed-v4는 general에서 선두(65.2 MTEB)입니다. BGE-M3는 open-weight multilingual에서 선두(63.0)입니다. leaderboard는 필요하지만 충분하지 않습니다. 항상 자신의 domain에서 benchmark하세요.

### 세 계층 pattern

| 사용 사례 | Pattern |
|----------|---------|
| Fast first-pass | Dense bi-encoder (BGE-M3, text-3-small) |
| Recall boost | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Precision on top-50 | Multi-vector (ColBERTv2) or cross-encoder reranker |

대부분의 production stack은 세 가지를 모두 사용합니다.

## 직접 만들기

### 1단계: baseline — Sentence-BERT로 dense embedding 만들기

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True`는 dot product가 cosine similarity와 같아지게 합니다. 항상 설정하세요.

### 2단계: Matryoshka truncation

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

truncation 후에는 다시 normalize하세요. Nomic v1.5, OpenAI text-3, Voyage-4는 앞쪽 몇 level에 대해 거의 lossless가 되도록 train되어 있습니다. Non-Matryoshka model(original Sentence-BERT)은 truncate하면 급격히 degrade됩니다.

### 3단계: BGE-M3 multi-functionality

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

세 개의 index, 한 번의 inference call입니다. Score fusion:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

weight는 자신의 domain에서 tune하세요.

### 4단계: custom task에서 MTEB eval

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

candidate model을 *representative* subset에서 실행하세요. leaderboard rank만 믿지 마세요. domain이 중요합니다.

### 5단계: 손으로 만든 cosine from scratch

`code/main.py`를 보세요. Averaged Hashing Trick embedding(stdlib-only)입니다. transformer embedding과 경쟁할 수준은 아니지만, 형태를 보여 줍니다. tokenize → vector → normalize → dot product입니다.

## 함정

- **query와 doc에 같은 model.** 일부 model(Voyage, Jina-ColBERT)은 asymmetric encoding을 사용합니다. query와 document가 다른 path를 통과합니다. 항상 model card를 확인하세요.
- **prefix 누락.** `bge-*` model은 query 앞에 `"Represent this sentence for searching relevant passages: "`를 붙여야 합니다. 잊으면 recall이 3-5 point 떨어집니다.
- **Matryoshka 과도한 trimming.** 1,536 → 256은 보통 안전합니다. 1,536 → 64는 아닙니다. eval set에서 validate하세요.
- **Context truncation.** 대부분의 model은 max length를 넘는 input을 조용히 truncate합니다. long doc에는 chunking이 필요합니다(lesson 23 참조).
- **latency tail 무시.** MTEB score는 p99 latency를 숨깁니다. 600M model이 335M model보다 2 point 높을 수 있지만 query당 비용은 3×일 수 있습니다.

## 사용하기

2026년 stack:

| 상황 | 선택 |
|-----------|------|
| English-only, fast, API | `text-embedding-3-large` or `voyage-3-large` |
| Open-weight, English | `BAAI/bge-large-en-v1.5` |
| Open-weight, multilingual | `BAAI/bge-m3` or `Qwen3-Embedding-8B` |
| Long context (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2 (137M params, MoE) |
| Storage-constrained | Matryoshka-truncated + int8 quantization |
| Keyword-heavy queries | Add SPLADE sparse, RRF-fuse with dense |

2026년 pattern: BGE-M3 또는 text-3-large로 시작하고, MTEB로 자신의 domain에서 evaluate한 뒤 domain-specific model이 3 point 이상 이기면 교체하세요.

## 출시하기

`outputs/skill-embedding-picker.md`로 저장하세요.

```markdown
---
name: embedding-picker
description: 주어진 corpus와 deployment를 위한 embedding model, dimension, retrieval mode를 고릅니다.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

corpus(size, languages, domain, avg length), deployment target(cloud / edge / on-prem), latency budget, storage budget이 주어지면 다음을 출력하세요.

1. 모델. 이름이 있는 checkpoint 또는 API. 한 문장짜리 이유.
2. 차원. Full / Matryoshka-truncated / int8-quantized. storage budget과 연결된 이유.
3. 모드. Dense / sparse / multi-vector / hybrid. 이유.
4. model card에서 요구하는 경우 query prefix / template.
5. 평가 계획. domain과 관련 있는 MTEB task + held-out domain eval with nDCG@10.

domain validation 없이 Matryoshka를 <64 dims로 truncate하는 recommendation은 거부하세요. 10k passage 미만 corpus에 ColBERTv2를 추천하지 마세요(overhead가 정당화되지 않습니다). 512-token window model로 routing된 long-document corpus(>8k tokens)는 flag하세요.
```

## 연습문제

1. **쉬움.** `bge-small-en-v1.5`로 100개 sentence를 full dim(384)에서 encode한 뒤 Matryoshka 128에서도 encode하세요. query 10개에서 MRR drop을 측정하세요.
2. **보통.** 자신의 domain에서 가져온 500개 passage에 대해 BGE-M3 dense, sparse, colbert를 비교하세요. recall@10에서 무엇이 이기나요? RRF fusion이 best single mode를 이기나요?
3. **어려움.** top-2 domain task에 대해 candidate model 세 개로 MTEB를 실행하세요. MTEB score, 100-query batch의 p99 latency, $/1M queries를 report하세요. Pareto-optimal한 것을 고르세요.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Dense embedding | vector | text마다 하나의 fixed-size vector입니다. ranking에는 cosine similarity를 사용합니다. |
| Sparse embedding | Learned BM25 | vocab token마다 하나의 weight입니다. 대부분 0이며 end-to-end로 train됩니다. |
| Multi-vector | ColBERT-style | token마다 하나의 vector입니다. MaxSim scoring을 사용합니다. index는 더 크고 recall은 더 좋습니다. |
| Matryoshka | Russian doll trick | 첫 N dim이 그 자체로 valid한 더 작은 embedding입니다. |
| MTEB | benchmark | Massive Text Embedding Benchmark — 출시 시점 56개 task, v2에서는 100+개. |
| BEIR | retrieval benchmark | 18개 zero-shot retrieval task입니다. cross-domain robustness 때문에 자주 인용됩니다. |
| Asymmetric encoding | Query ≠ doc path | model이 query와 document에 서로 다른 projection을 사용합니다. |

## 더 읽을거리

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder paper.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — leaderboard paper.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — unified three-mode model.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — dimension-ladder training objective.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — production에서의 late interaction.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — live ranking.
