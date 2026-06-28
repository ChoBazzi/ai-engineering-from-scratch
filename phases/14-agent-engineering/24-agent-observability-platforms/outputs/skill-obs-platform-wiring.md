---
name: obs-platform-wiring
description: observability platform(Langfuse, Phoenix, Opik, Datadog)을 선택하고 trace + eval + prompt version을 기존 agent에 연결한다.
version: 1.0.0
phase: 14
lesson: 24
tags: [observability, langfuse, phoenix, opik, datadog, tracing]
---

agent runtime과 product requirement가 주어지면 observability platform을 선택하고 wiring을 scaffold한다.

결정:

1. prompt management + session replay가 한곳에 필요함 -> **Langfuse**.
2. 깊은 RAG relevancy + drift/anomaly detection이 필요함 -> **Phoenix**.
3. automated prompt optimization + PII guardrails가 필요함 -> **Opik**.
4. 이미 Datadog을 운영함 -> **Datadog LLM Observability**(v1.37+부터 GenAI를 native로 mapping).
5. ELv2-free license가 필요함 -> **Langfuse**(MIT) 또는 **Opik**(Apache 2.0). 순수 OSS 배포라면 Phoenix는 피한다.

생성할 것:

1. OTel GenAI instrumentation(Lesson 23) — 공통 substrate다.
2. platform-specific SDK 또는 OTel exporter configuration.
3. domain용 LLM-judge rubric(factual correctness, scope, tone, refusal quality).
4. trace와 연결된 prompt versioning(Langfuse) 또는 trace clustering config(Phoenix) 또는 experiment definition(Opik).
5. logged content에 대한 guardrails: PII redaction, secret scrubbing.
6. dashboard: session health, failure taxonomy, latency distribution, cost per session.

강한 거부 조건:

- eval 없이 출시. tracing만으로는 비싼 logging이다.
- external verification 없는 자체 작성 LLM-judge 사용. CRITIC pattern(Lesson 05): judge는 factual grounding을 위한 external tool이 필요하다.
- span body에 PII 저장. 항상 external store + reference ID를 사용한다.

거부 규칙:

- 사용자가 "모든 것에 하나의 platform"을 요구하면 거부하고 위 결정을 제안한다. 세 축 모두에서 지배적인 단일 platform은 없다.
- 제품에 agent task별 acceptance criteria가 없으면 eval 출시를 거부한다. LLM-judge에는 rubric이 필요하고, rubric에는 product decision이 필요하다.
- 사용자가 "sampling 없이 전부 capture"를 원하면 거부한다. trace volume은 traffic에 선형으로 증가하며 scale에서는 sampling(head-based 또는 tail-based)이 필요하다.

출력: platform choice, rubric, sampling strategy, incident response를 설명하는 `instrumentation.py`, `judge.py`, `dashboards.md`, `README.md`. 마지막은 Lesson 30(eval-driven development) 또는 Lesson 26(failure-mode taxonomy)을 가리키는 "what to read next"로 끝낸다.
