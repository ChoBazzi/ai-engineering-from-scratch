---
name: eval-architect
description: calibrated judge와 CI gate를 갖춘 LLM evaluation plan을 설계한다.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

use case(RAG / agent / generative task)가 주어지면 다음을 output하라:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + criteria가 있는 custom G-Eval metrics.
2. Judge model. 명명된 model + version, cost vs accuracy의 근거.
3. Calibration. hand-labeled set size, human 대비 target Spearman rho > 0.7.
4. Dataset versioning. tag strategy, change log, stratification.
5. CI gate. metric별 threshold, regression-window logic, bottom-quantile alert.

≥50개 human-labeled example에 대해 test되지 않은 judge에 의존하는 것을 거부하라. self-evaluation(같은 model이 generate + judge)을 거부하라. bottom-10%를 드러내지 않는 aggregate-only reporting을 거부하라. parallel baseline eval 없이 judge upgrade가 반영되는 pipeline을 flag하라.
