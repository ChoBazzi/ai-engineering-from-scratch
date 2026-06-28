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
