---
name: long-context-eval
description: 주어진 model과 use case를 위한 long-context evaluation battery를 설계한다.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

target model, target context length, use case가 주어지면 다음을 output하라:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. 각 length에서 depth 0, 0.25, 0.5, 0.75, 1.0.
3. Metrics. retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. effective retrieval length(90% pass)와 effective reasoning length(70% pass). 둘 다 report하라.
5. Regression. fixed harness, model upgrade마다 rerun, delta를 surface하라.

model card만 보고 context window를 믿는 것을 거부하라. multi-hop workload에 대해 NIAH-only evaluation을 거부하라. vendor self-reported long-context score를 독립적 evidence로 받아들이는 것을 거부하라.
