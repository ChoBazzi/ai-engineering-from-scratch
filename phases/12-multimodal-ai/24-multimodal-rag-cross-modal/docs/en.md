# 멀티모달 RAG와 크로스모달 검색

> Vision-native document RAG는 한 조각일 뿐이다. 프로덕션 multimodal RAG는 text, image, audio, video를 가로질러 더 넓게 검색한다. 예를 들면 여행 계획("자연광이 드는 조용한 비건 브런치 장소를 찾아줘"), 의료 triage("이 사진 + 이 증상 노트와 맞는 부상은 무엇인가"), e-commerce("이 셀카와 비슷한 outfit을 내 사이즈로"), field service("이 엔진 소리와 부품 사진으로 진단해줘") 같은 workflow다. 2025년의 세 survey, Abootorabi et al., Mei et al., Zhao et al.은 하위 문제를 체계화했다: cross-modal retrieval, retrieval fusion, generation grounding, multimodal evaluation. 이 lesson은 survey를 읽고 production pipeline을 설계한다.

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 minutes

## 학습 목표

- cross-modal retrieval을 설계한다: text → image, image → text, audio → video 등.
- 세 가지 fusion 전략을 비교한다: score fusion, attention-based fusion, MoE fusion.
- generation grounding을 설명한다. source가 여러 modality의 혼합일 때 "cite your sources"가 어떤 모습인지 설명한다.
- 2025년의 대표 multimodal RAG survey 세 편과 그 하위 문제 taxonomy를 말한다.

## 문제

Single-modality RAG는 해결된 패턴이다. query를 embed하고, chunk를 embed하고, retrieve하고, LLM에 넣는다. Multimodal RAG에는 다음이 필요하다.

1. 여러 retrieval head(각 modality는 호환 가능한 공간의 embedding이 필요).
2. modality를 가로지르는 retrieval result fusion.
3. modality를 가로질러 source를 인용하는 generation grounding.
4. cross-modal signal을 다루는 evaluation metric.

2025년 survey들은 모두 같은 taxonomy에 도달한다.

## 개념

### 크로스모달 검색

modality A의 query가 주어졌을 때 modality B의 문서를 검색한다. 세 가지 패턴이 있다.

1. 공유 embedding space. CLIP과 CLAP은 text + image / text + audio embedding을 공유 공간에 만든다. modality 간 cosine similarity가 직접 작동한다. CLIP으로 학습된 pair에 제한된다.

2. modality별 encoder + translation. Text encoder + image encoder + 공간 사이를 매핑하는 작은 translator module. Gupta et al.의 Sen2Sen과 다른 2024년 설계들이다. 유연하지만 복잡성이 늘어난다.

3. VLM as encoder. VLM의 hidden state를 retrieval representation으로 사용한다. VLM이 지원하는 모든 modality에 작동한다. 품질은 더 높고 비용도 더 크다.

선택: text+image에는 CLIP / SigLIP 2, text+audio에는 CLAP, frontier 품질의 cross-modal에는 VLM hidden states.

### 융합 전략

결과 10개를 검색했다고 하자. 이미지 5개, 텍스트 passage 3개, 오디오 clip 2개다. 어떻게 합칠까?

Score fusion(가장 저렴). 각 modality가 자체 retriever를 가지고 score를 반환한다. modality 내부에서 score를 정규화한 뒤 합산한다. 단순하고 대개 작동한다.

Attention-based fusion. 검색된 모든 item을 연결하고 작은 attention network가 가중치를 매기게 한다. 학습이 필요하다.

MoE fusion. Gating network가 modality-specific expert로 routing한다. query type에 따라 다르게 routing된다. 시각 질문은 이미지 가중치가 더 높다.

프로덕션 기본값은 query의 dominant modality에 약간의 bias를 둔 score fusion이다. 도메인 A/B에서 명확한 이득이 보이면 MoE로 올린다.

### 생성 그라운딩

LLM은 각 claim을 만든 검색 item을 인용해야 한다. multi-modal의 경우:

- Text source: 표준 citation `[1]`.
- Image source: 짧은 caption과 함께 `[img 3]`.
- Audio: `[audio 2 at 0:34]`.

Grounding-aware data로 generator를 학습한다. training target의 각 claim은 source index로 tag된다. 추론 시 모델은 자연스럽게 citation을 내보낸다.

### 2025년 survey

Abootorabi et al.(arXiv:2502.08826, "Ask in Any Modality"): multimodal RAG taxonomy. retrieval, fusion, generation을 다룬다. 가장 넓은 coverage다.

Mei et al.(arXiv:2504.08748, "A Survey of Multimodal RAG"): 하위 작업 benchmark와 failure mode에 초점을 둔다. evaluation 설계에 유용하다.

Zhao et al.(arXiv:2503.18016): vision 중심 survey. ColPali 계열 작업에 강하다.

세 편을 모두 읽으면 2025년 봄 기준 state of the art를 얻는다. 하위 문제 대부분은 아직 열려 있다.

### MuRAG - 기반 논문

MuRAG(Chen et al., 2022)는 최초의 multimodal RAG였다. multimodal KB에서 image + text를 검색하고 답변을 생성했다. VLM 물결 이전에 가능성을 보였다. 현대 시스템(REACT, VisRAG, M3DocRAG)은 그 위에 구축된다.

### 프로덕션 여행 플래너 예시

Query: "자연광이 드는 조용한 비건 브런치 장소를 찾아줘."

Pipeline:

1. Query 분해. "quiet" → audio/review keyword; "vegan brunch" → menu item; "natural light" → image feature.
2. modality별 검색:
   - Review에 대한 text retrieval: "vegan brunch, quiet ambiance."
   - Restaurant photo에 대한 image retrieval: "natural light, airy."
   - Ambient-sound clip에 대한 audio retrieval: "low decibel, no music."
3. Score fusion. 각 restaurant가 composite score를 가진다.
4. Top-k restaurant → 모든 evidence와 함께 VLM generator → citation이 포함된 answer.

이는 text-RAG를 훨씬 넘어선다. 각 modality는 텍스트만으로는 놓치는 신호를 추가한다.

### 에이전트형 멀티모달 RAG

Multi-hop: 첫 검색이 high-confidence answer를 반환하지 않으면 LLM이 재구성하고 다시 검색한다. Phase 14의 agentic RAG 패턴이 여기에도 적용된다. 예:

- 초기 top-10 검색 → LLM이 "너무 시끄러움, <40 dB로 filter" 요청 → 다시 검색.
- 이미지 검색 → LLM이 한 이미지에 menu가 있음을 봄 → menu text 검색 → 답변.

복잡성은 늘지만 single-shot retrieval이 처리하지 못하는 query를 다룬다.

### 평가

Cross-modal evaluation은 아직 미성숙하다. 일반적인 proxy:

- modality별 Recall@k.
- Fused top-k accuracy.
- 사람 평가 end-to-end satisfaction.
- 작업별 지표(예약 완료, 구매 완료).

모든 modality를 아우르는 표준 benchmark는 없다. 대부분의 논문은 도메인별 작업에서 평가한다.

## 활용하기

`code/main.py`:

- restaurant의 공유 corpus에서 동작하는 세 mock retriever(text, image, audio).
- modality score를 configurable weight로 결합하는 score fusion.
- citation과 함께 최종 answer를 내보내는 generator stub.
- confidence가 낮으면 query를 재구성하는 단순 agentic loop.

## 산출물

이 lesson은 `outputs/skill-multimodal-rag-designer.md`를 만든다. multimodal query flow가 있는 product spec이 주어지면 retriever, fusion, generator, evaluation을 설계한다.

## 연습 문제

1. 의료 triage multimodal RAG를 제안하라. query = 부상 사진 + 텍스트 증상. 어떤 modality가 어떤 KB에서 검색하는가?

2. Score fusion은 단순 weighted sum이다. MoE fusion이 피하는 어떤 failure mode가 있는가?

3. Abootorabi et al.의 taxonomy(Section 3)를 읽어라. 세 가지 대표 하위 문제는 무엇이며, 선택한 product에 어떻게 매핑되는가?

4. 여행 플래너 multimodal RAG의 eval spec을 설계하라. image recall, audio recall, composite correctness를 어떤 metric이 다루는가?

5. Agentic multi-hop RAG는 왕복마다 latency tax가 있다. 어떤 query 난이도에서 정확도 향상이 latency를 정당화하는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Query one modality, retrieve another" | 텍스트 query로 이미지를 검색하고 이미지 query로 텍스트를 검색함; 공유 공간 또는 translator 필요 |
| Score fusion | "Combine scores" | modality별 retrieval score의 weighted sum; 가장 단순한 fusion |
| MoE fusion | "Modality-routed experts" | query별로 어떤 modality score를 신뢰할지 gating network가 선택 |
| Grounded generation | "Cite your sources" | 답변의 각 claim에 source index를 tag함 |
| MuRAG | "First multimodal RAG" | multimodal RAG 패턴을 확립한 2022년 논문 |
| Agentic multi-hop | "Reformulate and retry" | 첫 pass confidence가 낮을 때 LLM이 retriever에 다시 query함 |

## 더 읽을거리

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
