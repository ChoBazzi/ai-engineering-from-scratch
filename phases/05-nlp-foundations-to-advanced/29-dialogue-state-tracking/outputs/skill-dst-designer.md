---
name: dst-designer
description: dialogue state tracker를 설계합니다 — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

use case(domain, languages, vocab openness, compliance needs)가 주어지면 다음을 출력하세요.

1. Schema. domain list, domain별 slot, slot별 open vs closed vocabulary.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. 이유.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. held-out dialogue set에서 Joint Goal Accuracy, slot-level precision/recall, 가장 어려운 slot의 confusion.
5. Confirmation flow. 언제 사용자에게 명시적으로 confirm을 요청할지(destructive actions, low-confidence extractions).

rule-based secondary check가 없는 compliance-sensitive slot의 LLM-only DST는 거부하세요. user correction에서 slot을 roll back할 수 없는 DST는 거부하세요. version tag가 없는 schema를 flag하세요.
