---
name: skill-rag-pipeline
description: 첫 원리부터 RAG 파이프라인을 만들고 디버그합니다
version: 1.0.0
phase: 11
lesson: 6
tags: [rag, retrieval, embeddings, vector-search, llm-engineering]
---

# RAG 파이프라인 패턴

모든 RAG 시스템은 이 패턴을 따릅니다.

```text
documents -> chunk -> embed -> store
query -> embed -> search(top_k) -> build_prompt -> generate
```

색인은 문서마다 한 번 수행됩니다. 질의는 모든 사용자 요청마다 수행됩니다.

## RAG를 사용할 때

- LLM이 비공개 문서나 최신 문서에 접근해야 할 때
- Fine-tuning이 너무 비싸거나 업데이트가 너무 느릴 때
- 답변에 출처를 인용해야 할 때
- 지식 베이스가 자주 바뀔 때

## RAG를 사용하지 말아야 할 때

- 답이 LLM이 이미 가진 일반 지식일 때
- 작업이 사실 기반이 아니라 창의적 작업(글쓰기, 브레인스토밍)일 때
- 모델이 특정 reasoning 스타일을 채택해야 할 때(fine-tuning 사용)

## 구현 체크리스트

1. 문서를 50-token overlap을 둔 256-512 token 세그먼트로 청킹합니다
2. 일관된 embedding model로 각 청크를 임베딩합니다
3. 임베딩을 원문 텍스트와 함께 벡터 데이터베이스에 저장합니다
4. 질의 시점에는 같은 모델로 사용자의 질문을 임베딩합니다
5. 코사인 유사도로 가장 유사한 top-k(5-10) 청크를 검색합니다
6. 프롬프트를 만듭니다: 시스템 지시 + 검색된 컨텍스트 + 사용자 질문
7. 검색된 컨텍스트에 근거해 답변을 생성합니다
8. 출처 참조와 함께 답변을 반환합니다

## 흔한 실수

- 색인과 질의에 서로 다른 embedding model을 사용함. 벡터가 호환되지 않습니다
- 청크가 너무 작아 컨텍스트를 잃거나, 너무 커서 관련성이 희석됨
- 청크 사이에 overlap을 포함하지 않아 경계에서 문장이 잘림
- 문서가 바뀌었을 때 재색인을 잊음
- 일관된 답변을 생성하지 않고 검색된 청크를 그대로 사용자에게 반환함
- 사실 기반 RAG 질의에 temperature=0을 설정하지 않음. 더 높은 temperature는 더 많은 hallucination을 만듭니다

## 검색 디버깅

올바른 청크가 검색되지 않는다면:
1. 질의 임베딩을 출력하고 0이 아닌지 확인합니다
2. 관련 있음이 알려진 청크에 대해 코사인 유사도를 수동으로 확인합니다
3. 문서 어휘와 맞도록 질의를 다시 표현해 봅니다
4. 색인 시점과 질의 시점의 embedding model이 일치하는지 확인합니다
5. 관련 내용이 청킹 중에 사라졌는지 확인합니다

## 프로덕션 파라미터

- Chunk size: 256-512 tokens
- Overlap: 50 tokens(청크 크기의 10-20%)
- Top-k: 대부분의 사용 사례에서 5-10
- Temperature: 사실 기반 답변에는 0
- Embedding model: text-embedding-3-small(비용 효율적) 또는 text-embedding-3-large(더 높은 정확도)
