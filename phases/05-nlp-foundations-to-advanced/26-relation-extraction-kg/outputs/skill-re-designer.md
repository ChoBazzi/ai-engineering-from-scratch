---
name: re-designer
description: provenance와 canonicalization을 포함한 relation extraction pipeline을 설계합니다.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Corpus(domain, language, volume)와 downstream use(KG-RAG, analytics, compliance)가 주어지면 다음을 출력하세요:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Precision vs recall target에 연결된 reason.
2. Ontology. Closed property list(Wikidata / domain) 또는 canonicalization pass가 있는 open IE.
3. Provenance. 모든 triple은 source char-span + doc id를 포함합니다. Audit에는 타협할 수 없습니다.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. 손으로 label한 triple 200개에서 precision / recall + LLM-extracted sample의 hallucination-rate.

span verification(source provenance)이 없는 LLM-based RE pipeline은 거부하세요. canonicalization 없이 production graph로 흘러가는 open-IE output도 거부하세요. time-bounded relation(employer, spouse, position)에 temporal qualifier가 없는 pipeline은 표시하세요.
