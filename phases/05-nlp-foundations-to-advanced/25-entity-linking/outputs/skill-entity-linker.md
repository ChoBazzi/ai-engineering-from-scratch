---
name: entity-linker
description: Entity linking pipeline을 설계합니다 - KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Use case(domain KB, language, volume, latency budget)가 주어지면 다음을 출력하세요:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

mention-recall baseline이 없는 EL pipeline은 거부하세요(candidate gen이 올바른 entity를 surfacing했는지 모르면 disambiguator를 평가할 수 없습니다). 유효한 KB id로 constrained output하지 않는 LLM-prompted EL pipeline도 거부하세요. popularity bias가 minority entity(예: name-clashes)에 영향을 주는데 domain fine-tuning이 없는 system은 표시하세요.
