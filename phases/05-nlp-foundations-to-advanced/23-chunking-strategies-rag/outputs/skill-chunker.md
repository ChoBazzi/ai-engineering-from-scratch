---
name: chunker
description: 주어진 corpus와 query distribution에 맞는 chunking strategy, size, overlap을 고른다.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

corpus(document types, avg length, domain)와 query distribution(factoid / analytical / multi-hop)이 주어지면 다음을 출력하라.

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

min/max chunk size enforcement가 없는 chunking strategy는 거부하라. 도움이 된다는 ablation 없이 20%를 넘는 overlap은 거부하라. min-token floor가 없는 semantic chunking recommendation은 flag하라.
