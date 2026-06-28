# ColPali와 비전 네이티브 문서 RAG

> 전통적인 RAG는 PDF를 텍스트로 파싱하고, chunk로 나누고, chunk를 embed하고, vector를 저장한다. 모든 단계가 신호를 잃는다. OCR은 chart data를 떨어뜨리고, chunking은 표의 행을 끊고, text embedding은 그림을 무시한다. ColPali(Faysse et al., 2024년 7월)는 더 단순한 질문을 던졌다. 왜 텍스트를 추출해야 하는가? PaliGemma로 페이지 이미지를 직접 embed하고, 검색에는 ColBERT 스타일 late interaction을 사용하며, 문서가 가진 레이아웃, 그림, 글꼴, formatting 신호를 모두 유지하라. 공개 benchmark에서는 시각적으로 풍부한 문서에서 text-RAG보다 end-to-end 정확도가 20-40% 높았다. ColQwen2, ColSmol, VisRAG가 이 패턴을 확장했다. 이 lesson은 vision-native RAG thesis를 읽고 작은 ColPali 유사 indexer를 만든다.

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 minutes

## 학습 목표

- bi-encoder retrieval(문서당 하나의 vector)과 late-interaction retrieval(문서당 여러 vector)의 차이를 설명한다.
- ColBERT의 MaxSim 연산과 ColPali가 이를 텍스트 token에서 이미지 patch로 일반화하는 방식을 설명한다.
- 작은 ColPali 유사 indexer를 만든다: page → patch embeddings → query-term embeddings에 대한 MaxSim → top-k pages.
- 송장 / 재무 보고서 use case에서 ColPali + Qwen2.5-VL generator와 text-RAG + GPT-4를 비교한다.

## 문제

PDF에 text-RAG를 적용하면 문서 대부분을 버리게 된다. 재무 보고서의 Q3 매출 성장은 보통 chart에 있고, 의료 보고서의 findings는 주석이 달린 이미지에 있으며, 법률 계약의 서명 블록은 텍스트 사실이 아니라 레이아웃 사실이다.

text-RAG pipeline:

1. PDF → OCR / pdftotext로 텍스트.
2. 텍스트 → 300-500 token chunk.
3. Chunk → bi-encoder embedding(하나의 vector).
4. 사용자 query → embedding → cosine similarity → top-k chunk.
5. Chunk + query → LLM.

손실이 있는 다섯 단계다. Chart는 포착되지 않는다. 표는 chunk 사이에서 깨진다. 다단 레이아웃은 평탄화된다. 그림 주석은 사라진다.

ColPali의 해결책은 OCR을 건너뛰고 페이지 이미지를 직접 embed하는 것이다. 검색에는 ColBERT 스타일 late interaction을 사용하여 모델이 query time에 세밀한 patch에 attend할 수 있게 한다.

## 개념

### ColBERT(2020)

ColBERT(Khattab & Zaharia, arXiv:2004.12832)는 텍스트 검색 방법이다. 문서당 하나의 vector 대신 token당 하나의 vector를 만든다. query time에는:

- Query token이 각자 embedding을 얻는다(N_q vectors).
- Document token이 embedding을 얻는다(N_d vectors, 보통 cache됨).
- Score = query token별로 document token 중 최대 cosine similarity를 고르고 합산: Σ_i max_j cos(q_i, d_j).

이것이 MaxSim 연산이다. 각 query token은 가장 잘 맞는 document token을 "고른다." 최종 score는 그 합이다.

장점: 강한 recall, term-level semantics 처리. 단점: 문서당 N_d vectors라 storage가 비싸다.

### ColPali

ColPali(Faysse et al., arXiv:2407.01449)는 ColBERT 패턴을 이미지에 적용한다.

- 각 페이지는 PaliGemma(ViT + language)로 patch embeddings, 즉 페이지당 N_p vectors로 인코딩된다.
- 각 사용자 query(텍스트)는 query-token embeddings, 즉 N_q vectors로 인코딩된다.
- Score = Σ_i max_j cos(q_i, p_j), 즉 query-text-token과 page-image-patch 사이의 MaxSim이다.
- 전체 score로 top-k 페이지를 검색한다.

문서 ingestion time에는 모든 페이지를 PaliGemma로 embed하고 모든 patch embedding을 저장한다. Query time에는 query token을 embed하고, 저장된 모든 페이지 embedding과 MaxSim을 계산한 뒤 top-k 페이지를 반환한다.

장점: 시각적으로 풍부한 문서에서 text-RAG보다 end-to-end 성능이 20-40% 높다. 각 patch-vector는 로컬 레이아웃과 내용을 포착한다.

단점: 페이지당 N_p patches × 4-byte floats × D-dim vectors라 storage가 빠르게 증가한다. PQ / OPQ quantization으로 완화한다.

### ColQwen2와 ColSmol

ColQwen2(illuin-tech, 2024-2025)는 PaliGemma를 Qwen2-VL로 바꾼다. 더 나은 base encoder, 더 나은 retrieval이다.

ColSmol은 local / edge 사용을 위한 소형 variant다. 약 1B params의 ColSmol retriever는 소비자 GPU에서 실행된다.

### VisRAG

VisRAG(Yu et al., arXiv:2410.10594)는 다른 variant다. patch에 MaxSim을 적용하는 대신 VLM으로 각 페이지를 하나의 vector로 pool한 뒤 bi-encoder retrieval을 한다. indexing이 빠르고 storage가 작지만 recall은 약하다.

품질-vs-비용 trade-off: 품질은 ColPali, 규모는 VisRAG.

### M3DocRAG

M3DocRAG(Cho et al., arXiv:2411.04952)는 multi-modal retrieval을 multi-page multi-document reasoning으로 확장한다. 여러 문서에서 페이지를 검색하고, VLM을 위한 multi-page context를 구성한다.

### ViDoRe - 벤치마크

ColPali의 companion benchmark다. Visual Document Retrieval Evaluation. 작업에는 재무 보고서, 과학 논문, 행정 문서, 의료 기록, 매뉴얼이 포함된다. Metric은 nDCG@5다.

ColPali-v1은 ViDoRe에서 약 80% nDCG@5를 기록한다. 같은 문서에 대한 text-RAG는 약 50-60%다.

### End-to-end RAG 파이프라인

vision-native RAG의 경우:

1. Ingest: PDF → page images → PaliGemma encoding → 모든 patch embeddings 저장.
2. Query: 사용자 텍스트 → query-token embeddings → indexed pages 전체에 MaxSim → top-k pages.
3. Generate: top-k page images + query → VLM(Qwen2.5-VL 또는 Claude) → answer.

OCR은 어디에도 없다. 그림, chart, 글꼴, 레이아웃이 모두 답변으로 흐른다.

### 저장소 계산

50페이지 재무 보고서가 페이지당 729 patches와 128-dim embeddings를 가진다면:

- ColPali: 50 * 729 * 128 * 4 bytes = 원시 약 18 MB, PQ 후 약 4 MB.
- Text-RAG: 50 chunks * 768-dim * 4 bytes = 약 150 kB.

ColPali는 문서당 storage가 약 30배 더 크다. 규모가 커지면 OPQ / PQ가 이를 약 5-10배까지 낮추며, 대개 감당할 수 있다.

### text-RAG가 여전히 이기는 경우

- 레이아웃 신호가 없는 순수 텍스트 문서(wiki articles, chat logs). Text-RAG가 더 단순하고 storage가 저렴하다.
- storage가 비용을 지배하는 수백만 페이지 archive.
- retrieval과 함께 추출 가능한 OCR 텍스트를 요구하는 엄격한 규제 요구사항.

2026년의 나머지 모든 것, 즉 재무 보고서, 과학 논문, 법률 계약, 의료 기록, UX 문서에서는 vision-native RAG가 이긴다.

## 활용하기

`code/main.py`:

- 장난감 patch encoder: "page"(작은 feature vector grid)를 patch embedding 배열로 매핑한다.
- MaxSim scorer: query token embedding set과 page patch set 사이의 ColBERT 스타일 score를 계산한다.
- 장난감 페이지 5개를 index하고, query 3개를 실행한 뒤 score와 함께 top-k를 반환한다.

## 산출물

이 lesson은 `outputs/skill-vision-rag-designer.md`를 만든다. document-RAG 프로젝트가 주어지면 ColPali / ColQwen2 / VisRAG / text-RAG를 고르고 storage를 산정한다.

## 연습 문제

1. 페이지당 729 patches, 128-dim emb, 4-byte floats를 쓰는 200페이지 연례 보고서가 있다. 원시 storage와 PQ 압축(8x) storage를 계산하라.

2. MaxSim은 Σ_i max_j cos(q_i, p_j)다. 이 합은 단순 평균 similarity가 포착하지 못하는 무엇을 포착하는가?

3. ColPali는 페이지를 patch set으로 index한다. 대신 word level(ColBERT처럼)로 index하면 무엇이 바뀌는가? Trade-off는?

4. query당 500ms latency budget을 가진 100만 페이지 corpus의 end-to-end pipeline을 설계하라. ColQwen2 / VisRAG 중 고르고 정당화하라.

5. M3DocRAG(arXiv:2411.04952)를 읽어라. multi-page attention pattern을 설명하고 single-page ColPali retrieval과 어떻게 다른지 말하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | 단일 doc vector가 아니라 token별 또는 patch별 embedding + MaxSim을 사용하는 retrieval |
| MaxSim | "Max-over-patches" | 각 query token에 대해 가장 similarity가 높은 document token을 고르고 query 전체에 합산 |
| Bi-encoder | "Single-vector" | 문서당 하나의 vector; 빠르지만 granularity를 잃음 |
| Multi-vector | "Many-vectors-per-doc" | 문서 / 페이지당 N_p vectors 저장; storage 비용은 늘지만 recall 개선 |
| Patch embedding | "Page feature" | VLM encoder가 만든 이미지 patch당 하나의 vector이며 페이지별로 cache됨 |
| ViDoRe | "Vision doc bench" | visual document retrieval을 위한 ColPali benchmark suite |
| PQ quantization | "Product quantization" | vector similarity를 유지하면서 storage를 약 8배 줄이는 압축 |

## 더 읽을거리

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
