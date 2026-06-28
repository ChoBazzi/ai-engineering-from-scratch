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
2. Dense encoder. Specific model 이름(`all-MiniLM-L6-v2`, `bge-large-en-v1.5`, `e5-large-v2`, `paraphrase-multilingual-MiniLM-L12-v2`). Language, domain, context length에 맞춥니다.
3. Reranker. 사용한다면 cross-encoder model 이름(`cross-encoder/ms-marco-MiniLM-L-6-v2`, `BAAI/bge-reranker-large`). Top-30에 약 30-100ms latency가 추가된다고 flag합니다.
4. Evaluation plan. Recall@10이 primary retriever metric입니다. Multi-answer에는 MRR. 먼저 baseline을 세우고 incremental improvement를 그것과 비교해 측정합니다.

Corpus에 named entity, error code, product SKU가 있는데 dense가 exact match를 처리한다는 evidence가 없다면 dense-only 추천을 거부합니다. Final top-5가 user의 answer를 결정하는 high-stakes retrieval(legal, medical)에서는 reranking 생략을 거부합니다.
