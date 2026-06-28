---
name: skill-embedding-patterns
description: embeddings, vector search, similarity를 위한 production pattern
version: 1.0.0
phase: 11
lesson: 4
tags: [embeddings, vectors, similarity, search, chunking, quantization]
---

# 임베딩 패턴

모든 embedding workflow는 이 contract를 따릅니다.

```text
text -> embed(text) -> vector (float array)
similarity(vector_a, vector_b) -> score (float)
```

embedding model과 similarity metric이 중요한 두 가지 결정입니다. 나머지는 plumbing입니다.

## embedding을 사용할 때

- 문서 전체의 semantic search(keyword가 아니라 의미를 찾음)
- 유사 item clustering(support ticket, product review, bug report)
- nearest neighbor 기반 classification(labeled example과의 유사성으로 새 item에 label 부여)
- recommendation system(사용자가 좋아한 것과 유사한 item 찾기)
- deduplication(similarity threshold로 near-duplicate content 찾기)

## embedding을 사용하지 말아야 할 때

- exact keyword matching(full-text search 사용)
- structured query(SQL, filter 사용)
- 수동 labeling이 더 빠른 작은 dataset(<100 items)
- 정확도보다 explainability가 더 중요한 작업(embedding은 opaque함)

## 모델 선택

제약에 따라 선택하세요.

- **API와 best value 필요**: OpenAI text-embedding-3-small (1536d, $0.02/1M tokens)
- **최대 정확도 필요**: Voyage-3 (1024d, $0.06/1M tokens, highest MTEB)
- **local/private 필요**: BGE-M3 (1024d, free, multilingual, GPU 권장)
- **빠른 local prototyping 필요**: all-MiniLM-L6-v2 (384d, free, CPU에서 실행)
- **multilingual 필요**: Cohere embed-v3 (1024d) 또는 BGE-M3(둘 다 strong multilingual)

규칙: indexing과 querying 사이에 embedding model을 절대 섞지 마세요. 서로 다른 모델의 vector는 호환되지 않는 space에 존재합니다.

## Chunking 규칙

1. chunk당 256-512 token, 50-token overlap을 목표로 하세요.
2. 가능하면 문장 중간에서 자르지 마세요.
3. 모든 chunk에 metadata(source file, section title, position)를 포함하세요.
4. structured docs(Markdown, HTML)는 먼저 heading boundary에서 나누세요.
5. 알려진 답을 검색하고 retrieval을 확인해 chunk quality를 테스트하세요.

## Similarity metric 선택

- **Cosine similarity**: default choice, variable-length text 처리, normalized
- **Dot product**: vector가 이미 unit-normalized된 경우 사용(OpenAI model이 해당), 약간 더 빠름
- **Euclidean distance**: absolute position이 중요한 clustering에 사용

vector가 normalized되어 있으면 세 metric은 같은 ranking을 제공합니다. 선택은 non-normalized vector에서만 중요합니다.

## 저장소 최적화

세 단계 compression이 있으며 서로 쌓을 수 있습니다.

1. **Matryoshka truncation**: dimension 축소(1536 -> 256 = 6x 절감, 정확도 손실 3-5%)
2. **Float16 quantization**: dimension당 storage 절반(2x 절감, 정확도 손실 <1%)
3. **Binary quantization**: dimension당 1 bit(32x 절감, 정확도 손실 5-10%, rescoring과 함께 사용)

production pattern: 전체 corpus에서 binary search를 수행하고 top-1000을 float32 vector로 rescore합니다.

## 검색 후 rerank

최고 정확도를 위한 2단계 pipeline:

1. bi-encoder가 top-100 candidate를 retrieve합니다(빠름, pre-computed embedding 사용).
2. cross-encoder가 top-10으로 rerank합니다(느림, 각 query-doc pair를 처리).

이는 precision metric에서 single-stage retrieval보다 10-15% 더 좋습니다. latency보다 accuracy가 중요할 때 사용하세요.

## 흔한 실수

- indexing과 querying에 서로 다른 embedding model 사용
- chunk 대신 전체 document를 embedding함(embedding이 모든 것의 평균이 됨)
- cosine similarity 전에 vector를 normalize하지 않음(대부분의 모델은 pre-normalize하지만 확인하세요)
- chunk overlap 무시(boundary에서 잘린 sentence는 context를 잃음)
- original text 없이 vector만 저장(retrieval에는 둘 다 필요함)
- 모델이 바뀌었는데 re-embedding하지 않음(old vector는 호환되지 않음)
- accuracy만 보고 dimension 선택(storage와 latency는 dimension에 선형으로 증가)

## 임베딩 디버깅

search result가 좋지 않다면:

1. query embedding이 non-zero인지 확인하세요(empty 또는 whitespace input은 zero vector를 만듦).
2. 관련성이 알려진 document의 similarity score를 수동으로 확인하세요.
3. document vocabulary와 맞도록 query를 다시 표현해 보세요.
4. 관련 content가 chunk 사이에서 갈라지지 않았는지 chunk boundary를 검사하세요.
5. normalization issue를 찾기 위해 metric(cosine, dot, euclidean)별 top-k result를 비교하세요.
6. pipeline이 작동하는지 확인하기 위해 trivially matching query(document의 문장 복사)로 테스트하세요.

## 프로덕션 매개변수

- Chunk size: 256-512 tokens
- Chunk overlap: 50 tokens(chunk size의 10-20%)
- Top-k retrieval: 직접 사용은 5-10, reranking은 50-100
- Similarity threshold: cosine 기준 0.7+(이보다 낮으면 보통 결과가 무관함)
- Batch embedding: throughput을 위해 API call당 100-500 text 처리
- Index rebuild: 모델이 바뀌거나 document가 크게 업데이트되면 re-embed
