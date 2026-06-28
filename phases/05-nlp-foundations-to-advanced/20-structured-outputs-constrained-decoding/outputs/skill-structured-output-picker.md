---
name: structured-output-picker
description: structured output approach, schema design, validation plan을 선택합니다.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

use case(provider, latency budget, schema complexity, failure tolerance)가 주어지면 다음을 출력합니다.

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, 또는 XGrammar CFG. 한 문장 이유.
2. Schema design. Field order(reasoning first, answer last), "unknown"을 위한 nullable fields, enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate(target 100%), semantic validity(LLM-judge), field-coverage rate, latency p50/p99.

`answer` 또는 `decision`을 reasoning fields보다 앞에 두는 design을 거부하세요. schema 없는 bare JSON mode 사용을 거부하세요. recursive schema가 FSM-only library 뒤에 있으면 flag하세요.
