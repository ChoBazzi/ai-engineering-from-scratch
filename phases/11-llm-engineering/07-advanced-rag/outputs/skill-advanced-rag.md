---
name: skill-advanced-rag
description: hybrid search, reranking, evaluation으로 프로덕션급 RAG를 만듭니다
version: 1.0.0
phase: 11
lesson: 7
tags: [rag, hybrid-search, bm25, reranking, hyde, evaluation]
---

# Advanced RAG 패턴

Basic RAG: embed query -> vector search -> top-k -> generate.
Advanced RAG: embed query + BM25 -> fuse ranks -> rerank -> top-k -> generate.

```text
query -> [vector search (top-50)] -+-> RRF fusion -> reranker (top-5) -> prompt -> LLM
                                   |
query -> [BM25 search (top-50)]  --+
```

## basic RAG에서 업그레이드할 때

- 검색 품질이 Recall@5 기준 70% 아래로 떨어질 때
- 사용자가 틀리거나 무관한 답변을 보고할 때
- Corpus가 100K 청크를 넘어설 때
- 질의가 문서와 다른 어휘를 사용할 때
- Multi-hop 질문이 계속 실패할 때

## 구현 체크리스트

1. vector index와 나란히 BM25 index를 추가합니다
2. 두 검색을 병렬로 실행합니다(각각 top-50)
3. Reciprocal Rank Fusion(k=60)으로 병합합니다
4. 상위 후보를 cross-encoder로 rerank합니다
5. 최종 프롬프트에는 top-5를 사용합니다
6. 테스트 세트에 faithfulness evaluation을 추가합니다

## 기법 선택 가이드

- **Hybrid search**: 프로덕션에서는 항상 사용하세요. 질의 시점 추가 비용이 없습니다.
- **Reranking**: Recall@50은 좋지만 Recall@5가 나쁠 때 사용하세요. 50-200ms 지연 시간을 추가합니다.
- **HyDE**: 질의가 모호하거나 문서와 다른 어휘를 사용할 때 사용하세요. LLM 호출이 하나 추가됩니다.
- **Parent-child chunks**: 작은 청크는 컨텍스트가 부족하고 큰 청크는 관련성을 희석할 때 사용하세요.
- **Metadata filtering**: corpus에 날짜, 출처 유형, 부서 같은 명확한 카테고리가 있을 때 사용하세요.
- **Query decomposition**: 여러 문서의 정보가 필요한 multi-hop 질문에 사용하세요.

## 흔한 실수

- 서로 다른 청크 집합으로 BM25와 vector search를 실행함. 둘은 같은 corpus를 검색해야 합니다
- reranking 후보 풀이 너무 작음. top-10은 너무 적으니 top-50을 사용하세요
- 모든 질의에 HyDE를 추가함. 어휘 불일치가 병목일 때만 도움이 됩니다
- 변경을 평가하지 않음. 각 기법 적용 전후에 Recall@k를 측정하세요
- 어디서 실패하는지 측정하기 전에 파이프라인을 과도하게 엔지니어링함

## 평가 workflow

1. 알려진 답변 청크가 있는 테스트 질문을 50개 이상 만듭니다
2. 각 검색 방법에 대해 Recall@5와 Recall@10을 측정합니다
3. 검색이 성공한 질의에 대해 생성 답변의 faithfulness를 측정합니다
4. corpus가 커지는 동안 매주 지표를 추적합니다
5. 더 많은 기법을 추가하기 전에 개별 실패를 조사합니다
