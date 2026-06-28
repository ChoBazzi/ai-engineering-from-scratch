---
name: document-ai-stack-picker
description: domain, scale, regulatory 요구에 따라 document-AI 프로젝트에 OCR pipeline, OCR-free specialist, VLM-native 중 무엇을 쓸지 고른다.
version: 1.0.0
phase: 12
lesson: 22
tags: [document-ai, ocr, donut, nougat, paligemma, vlm-native]
---

document-AI 프로젝트(domain: invoices / scientific papers / forms / mixed; scale: pages per day; quality bar; regulatory needs)가 주어지면 stack을 고르고 reference config를 만든다.

생성할 것:

1. Stack 선택. Era 1(OCR pipeline + LayoutLMv3), Era 2(Donut / Nougat OCR-free), Era 3(VLM-native), 또는 hybrid.
2. 페이지당 비용 추정. 선택한 stack에서의 토큰 수와 지연 시간.
3. 정확도 기대치. DocVQA + ChartQA + domain-specific benchmark.
4. 손글씨 전략. 비용에 민감하지 않으면 VLM-native, 규모가 크면 전용 TrOCR + routing.
5. 수학 / LaTeX 출력. 과학 논문에는 Nougat, 그 외에는 VLM.
6. 규제 fallback. cross-check audit log를 포함한 hybrid.

강한 거부:
- 비용 분석 없이 하루 100만 페이지 초과에 VLM-native를 제안하는 것. 페이지당 2576px의 토큰 비용은 크다.
- 감사 경로 없이 규제 workflow에 단일 모델 솔루션을 추천하는 것.
- Nougat이 스캔된 송장을 처리한다고 주장하는 것. 그렇지 않다. Nougat은 과학 논문 specialist다.

거부 규칙:
- scale이 하루 1000만 페이지를 초과하면 Era 3을 거부하고 Era 3을 sampling validator로 둔 Era 1을 추천한다.
- domain이 손글씨 비중이 크면 OCR pipeline을 거부하고 VLM-native + handwriting specialist(TrOCR)를 추천한다.
- 수식에 LaTeX fidelity가 필요하면 loop 안에 Nougat을 요구한다.

출력: stack, 비용, 정확도, 손글씨, 수학, 규제를 포함한 한 페이지 계획. arXiv 2308.13418(Nougat), 2204.08387(LayoutLMv3), 2111.15664(Donut)로 끝낸다.
