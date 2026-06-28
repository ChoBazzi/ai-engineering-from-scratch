# 문서와 다이어그램 이해

> 문서는 사진이 아니다. PDF, 과학 논문, 송장, 손글씨 양식에는 레이아웃, 표, 다이어그램, 각주, 헤더, 의미 구조가 있으며, 단순 이미지 이해만으로는 이를 포착할 수 없다. VLM 이전의 stack은 Tesseract OCR + LayoutLMv3 + 표 추출 휴리스틱으로 이루어진 pipeline이었다. VLM 물결은 이를 OCR-free 모델, 즉 구조화된 markup을 직접 내보내는 Donut(2022), Nougat(2023), DocLLM(2023)으로 대체했다. 2026년 frontier는 그냥 "페이지 이미지를 Claude Opus 4.7에 2576px native로 넣기"이며, 구조화된 markup 출력은 덤으로 따라온다. 이 lesson은 document AI의 세 시대 흐름을 읽는다.

**Type:** Build
**Languages:** Python (stdlib, layout-aware document parser skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 5 (NLP)
**Time:** ~180 minutes

## 학습 목표

- document AI의 세 시대를 설명한다: OCR pipeline, OCR-free, VLM-native.
- LayoutLMv3의 세 입력 스트림을 설명한다: 텍스트, 레이아웃(bbox), 이미지 patch, 그리고 unified masking.
- Donut(OCR-free, 이미지 → markup), Nougat(과학 논문 → LaTeX), DocLLM(layout-aware generative), PaliGemma 2(VLM-native)를 비교한다.
- 새로운 작업(송장, 과학 논문, 손글씨 양식, 중국어 영수증)에 맞는 문서 모델을 고른다.

## 문제

"이 PDF를 이해하라"는 deceptively hard하다. 정보는 다음에 들어 있다.

- 텍스트 내용(신호의 90%).
- 레이아웃(헤더, 각주, 사이드바, 2단 형식).
- 표(행, 열, 병합 셀).
- 그림과 다이어그램.
- 손글씨 주석.
- 글꼴과 typography(제목 vs 본문).

원시 OCR은 텍스트를 덤프하고 나머지를 잃는다. 송장을 다루는 시스템은 "Total: $1,245"가 각주가 아니라 오른쪽 아래에서 나왔다는 사실을 알아야 한다.

## 개념

### 시대 1 - OCR pipeline(2021년 이전)

고전적인 stack:

1. PDF → 페이지별 이미지.
2. Tesseract(또는 상용 OCR)가 단어별 bounding box와 함께 텍스트를 추출한다.
3. 레이아웃 분석기가 블록(헤더, 표, 문단)을 식별한다.
4. 표 구조 인식기가 표를 파싱한다.
5. 도메인 규칙 + regex가 필드를 추출한다.

깨끗한 인쇄 텍스트에는 잘 작동한다. 손글씨, 기울어진 스캔, 복잡한 표, 비영어 스크립트에서는 깨진다. 모든 실패 모드에는 custom exception path가 필요하다.

### TrOCR(2021)

TrOCR(Li et al., arXiv:2109.10282)는 Tesseract의 고전적인 CNN-CTC를 synthetic + real 텍스트 이미지로 학습한 transformer encoder-decoder로 대체했다. 손글씨와 다국어 텍스트에서 명확한 승리였다. 여전히 pipeline(검출기, TrOCR, 레이아웃)이지만 OCR 단계가 크게 개선되었다.

### 시대 2 - OCR-free(2022-2023)

첫 OCR-free 모델들은 이렇게 말했다. 검출을 완전히 건너뛰고 이미지 픽셀을 구조화된 출력으로 직접 매핑하라.

Donut(Kim et al., arXiv:2111.15664):
- Encoder-decoder transformer이며 encoder는 Swin-B다.
- 출력은 양식 이해를 위한 JSON, 요약을 위한 markdown, 또는 작업별 schema가 될 수 있다.
- OCR도, 레이아웃도, 검출도 없다.

Nougat(Blecher et al., arXiv:2308.13418):
- 과학 논문에 특화되어 학습되었다.
- 출력은 LaTeX / markdown이다.
- 수식, 다단 레이아웃, 그림을 처리한다.
- 모든 arXiv parser가 호출하는 모델이다.

이들은 generalist가 아니라 specialist다. Donut은 과학 논문에서 실패하고, Nougat은 송장에서 실패한다.

### LayoutLMv3(2022)

다른 흐름도 있다. LayoutLMv3(Huang et al., arXiv:2204.08387)는 OCR을 유지하되 레이아웃 이해를 추가한다.

- 세 입력 스트림: OCR 텍스트 토큰, 토큰별 2D bounding box, 이미지 patch.
- 세 modality 전반에 걸친 masked training objective(masked text, masked patches, masked layout).
- Downstream: 분류, entity extraction, table QA.

LayoutLMv3는 OCR 기반 문서 이해의 정점이다. 양식과 송장에 강하다. 상류에 OCR이 필요하다. 표준화된 문서 benchmark에서 VLM 이전 최고 정확도를 냈다.

### DocLLM(2023)

DocLLM(Wang et al., arXiv:2401.00908)은 LayoutLM의 generative sibling이다. 레이아웃 토큰을 조건으로 자유 형식 답변을 생성한다. 문서 QA에 더 낫지만 여전히 OCR 입력에 의존한다.

### 시대 3 - VLM-native(2024+)

2024년 VLM은 pipeline 전체를 대체할 만큼 좋아졌다. 전체 페이지 이미지를 고해상도로 VLM에 넣고, 질문을 던지고, 답을 받는다.

- LLaVA-NeXT 336-tile AnyRes는 작은 문서에 작동한다.
- Qwen2.5-VL dynamic-resolution은 2048+ 픽셀을 native로 처리한다.
- Claude Opus 4.7은 2576px 문서를 지원한다.
- PaliGemma 2(2025년 4월)는 문서 + 손글씨에 특화해 학습한다.

VLM-native와 OCR-pipeline 사이의 격차는 빠르게 닫혔다. 2026년에는 VLM-native가 다음에서 이긴다.

- Scene text(손글씨 + 인쇄, 혼합 스크립트).
- 병합 셀이 있는 복잡한 표.
- 텍스트에 포함된 수학 수식.
- 텍스트 주석이 있는 그림.

OCR pipeline이 여전히 이기는 곳:

- 페이지당 지연 시간이 중요한 대규모 순수 스캔 workload.
- Pipeline reliability(결정적 실패 vs VLM hallucination).
- 감사 가능한 OCR 출력을 요구하는 규제 환경.

### Claude 4.7 / GPT-5 프런티어

2576픽셀 native 입력에서 frontier VLM은 거의 인간 수준 정확도로 문서를 이해한다. 2026년 초 benchmark 수치는 다음과 같다.

- DocVQA: Claude 4.7 약 95.1, PaliGemma 2 약 88.4, Nougat 약 77.3, pipelined LayoutLMv3 약 83.
- ChartQA: Claude 4.7 약 92.2, GPT-4V 약 78.
- VisualMRC: Claude 4.7 약 94.

폐쇄형 모델의 격차는 대부분 해상도와 base LLM 규모에서 나온다. 공개 7B 모델은 몇 점 뒤처져 있지만 따라잡고 있다.

### 수학 수식과 LaTeX 출력

과학 논문에는 수식의 정확한 LaTeX 출력이 필요하다. Nougat은 이를 위해 학습되었다. LaTeX target으로 학습한 VLM(Qwen2.5-VL-Math, Nougat 파생 모델)은 사용할 만한 LaTeX를 만든다. 명시적인 LaTeX 학습이 없으면 VLM은 읽을 수는 있지만 부정확한 transcription을 만든다.

2026년 과학 논문 pipeline: PDF에 Nougat을 먼저 실행한 다음, 까다로운 페이지에는 VLM을 사용한다.

### 손글씨

여전히 가장 어려운 하위 작업이다. 인쇄 + 손글씨가 섞인 경우(의사 노트, 작성된 양식)는 비용 측면에서 OCR pipeline이 여전히 VLM보다 낫다. 손글씨 전용 VLM은 개선 중이다(Claude 4.7, PaliGemma 2).

### 2026년 recipe

새 document-AI 프로젝트라면:

- 대규모 순수 인쇄 송장: LayoutLMv3 + 규칙, 비용 효율적.
- 혼합 문서(과학 + 손글씨 + 양식): VLM-native(PaliGemma 2 또는 Qwen2.5-VL).
- 전체 arXiv ingestion: 수식에는 Nougat, 그림에는 VLM.
- 규제: OCR pipeline + 교차 검사용 VLM validator.

## 활용하기

`code/main.py`:

- 장난감 layout-aware tokenizer: (text, bbox) 쌍이 주어지면 LayoutLMv3 스타일 입력을 만든다.
- Donut 스타일 작업 schema 생성기: 양식을 위한 JSON template.
- OCR-pipeline, Donut, Nougat, VLM-native 사이에서 페이지당 토큰 예산을 비교한다.

## 산출물

이 lesson은 `outputs/skill-document-ai-stack-picker.md`를 만든다. document-AI 프로젝트(domain, scale, quality, regulatory)가 주어지면 OCR pipeline, OCR-free specialist, VLM-native 중에서 고른다.

## 연습 문제

1. 프로젝트가 하루 1000만 개 송장을 처리한다. 정확도를 잃지 않으면서 페이지당 비용을 최소화하는 stack은 무엇인가?

2. 왜 LayoutLMv3는 form QA에서 pure-CLIP-VLM보다 뛰어나지만 scene-text에서는 뒤처지는가? bbox stream은 무엇을 포기하는가?

3. Nougat은 LaTeX를 생성한다. VLM-native 출력이 LaTeX fidelity에서 Nougat을 이기는 테스트 사례와 Nougat이 이기는 사례를 제안하라.

4. PaliGemma 2 논문(Google, 2024)을 읽어라. PaliGemma 1 대비 문서 정확도를 끌어올린 핵심 학습 데이터 추가는 무엇이었는가?

5. 규제에 안전한 hybrid를 설계하라. OCR pipeline을 primary로, VLM을 secondary cross-check로 둔다. 불일치는 어떻게 해결하는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| OCR pipeline | "Tesseract-style" | detect -> OCR -> layout -> rules로 이어지는 단계적 stack; 결정적이지만 취약함 |
| OCR-free | "Donut-style" | 명시적 OCR을 건너뛰는 image-to-output transformer; 단일 모델 |
| Layout-aware | "LayoutLM" | 입력에 토큰별 bbox 좌표가 포함되며 modality 전반에 unified masking 적용 |
| VLM-native | "Frontier VLM" | 페이지 이미지를 고해상도로 Claude/GPT/Qwen VLM에 직접 넣음; pipeline 없음 |
| DocVQA | "Doc benchmark" | Document VQA 표준이며 가장 많이 인용되는 점수 |
| Markup output | "LaTeX / MD" | 자유 형식 텍스트 대신 구조화된 출력 형식; downstream 자동화를 가능하게 함 |

## 더 읽을거리

- [Li et al. — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher et al. — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang et al. — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim et al. — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang et al. — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
