---
name: vision-rag-designer
description: ColPali / ColQwen2 / VisRAG를 사용하는 vision-native document RAG를 storage estimate와 generator 선택까지 포함해 설계한다.
version: 1.0.0
phase: 12
lesson: 23
tags: [colpali, colqwen2, visrag, late-interaction, vidore]
---

document RAG 프로젝트(corpus size, query latency target, storage budget, per-query cost)가 주어지면 vision-native RAG config를 내보낸다.

생성할 것:

1. Retriever 선택. ColPali(PaliGemma base), ColQwen2(Qwen2-VL base, 더 나은 품질), ColSmol(edge용 1B), 또는 VisRAG(bi-encoder, 더 저렴한 storage).
2. Storage estimate. N_docs * N_p_per_doc * D * 4 bytes raw; PQ를 쓰면 8로 나눈다.
3. Latency estimate.
   - Retrieval SLA: 약 10ms query embed + top-k retrieval(MaxSim 또는 ANN), index 크기에 의존.
   - Full-answer SLA: retrieval latency + 200-500ms generator(모델과 하드웨어에 의존).
4. Generator 선택. 공개 모델은 Qwen2.5-VL-72B, frontier는 Claude Opus 4.7.
5. Compression plan. PQ / OPQ ratio target 8-16x; 빠른 ANN을 위한 HNSW index.
6. text-RAG에서의 migration path. A/B 방법과 완전 cutover 시점.

강한 거부:
- 1만 페이지 초과 corpus에서 PQ compression 없이 ColPali를 사용하는 것. Storage가 폭발한다.
- bi-encoder retrieval이 document recall에서 ColBERT MaxSim과 맞먹는다고 주장하는 것. ViDoRe에서는 그렇지 않다.
- chart + table workload에 text-RAG를 추천하는 것. Text-RAG는 신호 대부분을 잃는다.

거부 규칙:
- corpus가 순수 텍스트(wiki, chat logs)이면 vision-native RAG를 거부하고 표준 text-RAG를 추천한다.
- retrieval SLA가 100ms 미만이면 ColPali MaxSim보다 VisRAG(bi-encoder)를 선호한다.
- full-answer SLA가 100ms 미만이면 generative RAG 전체를 거부하고 retrieval-only UX 또는 cached answers를 추천한다.
- storage budget이 1 GB 미만이고 corpus가 10만 페이지를 초과하면 full-fidelity ColPali를 거부하고 aggressive PQ 또는 VisRAG를 제안한다.

출력: retriever 선택, storage estimate, latency, generator, compression, migration을 포함한 한 페이지 RAG 설계. arXiv 2407.01449(ColPali), 2410.10594(VisRAG)로 끝낸다.
