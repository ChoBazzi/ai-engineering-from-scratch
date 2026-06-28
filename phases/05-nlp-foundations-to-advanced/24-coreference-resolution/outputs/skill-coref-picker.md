---
name: coref-picker
description: coreference approach, evaluation plan, integration strategy를 고른다.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

use case(single-doc / multi-doc, domain, language)가 주어지면 다음을 출력하라.

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

sliding-window merge 없이 2,000 tokens를 넘는 document에 LLM-only coref를 쓰는 것은 거부하라. mention-level precision-recall report 없이 coref를 실행하는 pipeline은 거부하라. demographically diverse text에 배포된 gender-heuristic system은 flag하라.
