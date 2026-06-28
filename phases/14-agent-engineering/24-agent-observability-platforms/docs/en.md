# 에이전트 Observability: Langfuse, Phoenix, Opik

> 2026년에는 세 open-source agent observability platform이 두드러진다. Langfuse(MIT) — 월 600만+ 설치, tracing + prompt management + evals + session replay. Arize Phoenix(Elastic 2.0) — 깊은 agent-specific evals, RAG relevancy, OpenInference auto-instrumentation. Comet Opik(Apache 2.0) — automated prompt optimization, guardrails, LLM-judge hallucination detection.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 minutes

## Learning Objectives

- 주요 open-source agent observability platform 세 가지와 license를 말한다.
- 각각의 강점을 구분한다: Langfuse(prompt mgmt + sessions), Phoenix(RAG + auto-instrumentation), Opik(optimization + guardrails).
- 2026년에 조직의 89%가 agent observability를 갖췄다고 보고하는 이유를 설명한다.
- LLM-judge evaluation을 포함한 stdlib trace-to-dashboard pipeline을 구현한다.

## The Problem

OTel GenAI(Lesson 23)는 schema를 제공한다. 그래도 span을 ingest하고, evaluation을 실행하고, prompt version을 저장하고, regression을 드러내는 platform이 필요하다. 세 후보는 lifecycle의 서로 다른 부분을 강조한다.

## The Concept

### Langfuse (MIT)

- 월 600만+ SDK 설치, GitHub star 19k+.
- 기능: tracing, versioning + playground가 있는 prompt management, evaluations(LLM-as-judge, user feedback, custom), session replay.
- 2025년 6월: 기존 commercial module(LLM-as-a-judge, annotation queues, prompt experiments, Playground)이 MIT로 open-source화됨.
- 강점: 촘촘한 prompt-management loop를 갖춘 end-to-end observability.

### Arize Phoenix (Elastic License 2.0)

- 더 깊은 agent-specific evaluation: trace clustering, anomaly detection, RAG용 retrieval relevancy.
- native OpenInference auto-instrumentation.
- production에서는 managed Arize AX와 함께 사용한다.
- prompt versioning 없음 — 더 넓은 platform 옆에서 drift/behavioral-regression tool로 포지셔닝된다.
- 강점: RAG relevancy, behavioral drift, anomaly detection.

### Comet Opik (Apache 2.0)

- A/B experiment를 통한 automated prompt optimization.
- Guardrails(PII redaction, topical constraints).
- LLM-judge hallucination detection.
- Comet 자체 측정 benchmark: Opik logs + evals 23.44s vs Langfuse 327.15s(약 14배 차이) — vendor benchmark는 방향성으로만 받아들인다.
- 강점: optimization loop, automated experimentation, guardrail enforcement.

### Industry data

Maxim(2026 field analysis)에 따르면 조직의 89%가 agent observability를 갖추고 있으며, quality issue가 최상위 production barrier다(응답자 32%가 언급).

### Picking one

| Need | Pick |
|------|------|
| prompt management가 있는 all-in-one | Langfuse |
| 깊은 RAG evaluation + drift | Phoenix |
| automated optimization + guardrails | Opik |
| open license, ELv2 제외 | Langfuse(MIT) 또는 Opik(Apache 2.0) |
| Datadog / New Relic integration | 아무거나 — 모두 OTel을 export한다 |

### Where this pattern goes wrong

- **eval strategy가 없음.** evaluation 없는 tracing은 비싼 logging일 뿐이다.
- **grounding 없는 자체 LLM-judge.** CRITIC pattern(Lesson 05)이 적용된다. judge는 factual verification을 위한 external tool이 필요하다.
- **prompt version이 trace와 연결되지 않음.** prod가 regress하면 원인 prompt를 bisect할 수 없다.

## Build It

`code/main.py`는 stdlib trace collector + LLM-judge evaluator를 구현한다.

- GenAI 형태의 span을 ingest한다.
- session별로 group하고 failed run(guardrail trip, low-confidence eval)을 tag한다.
- rubric으로 agent response를 score하는 scripted LLM-judge.
- dashboard 같은 summary: failure rate, top failure reasons, eval score distribution.

실행:

```bash
python3 code/main.py
```

출력: Langfuse/Phoenix/Opik이 보여 줄 내용과 맞는 session별 eval score와 failure categorization.

## Use It

- **Langfuse** self-hosted 또는 cloud. OTel 또는 자체 SDK로 연결한다.
- **Arize Phoenix** self-hosted. OpenInference를 auto-instrument한다.
- **Comet Opik** self-hosted 또는 cloud. automated optimization loop.
- **Datadog LLM Observability** 이미 Datadog을 운영하는 ops+ML 혼합 팀에 적합하다.

## Ship It

`outputs/skill-obs-platform-wiring.md`는 platform을 선택하고 trace + eval + prompt version을 기존 agent에 연결한다.

## Exercises

1. 일주일치 OTel trace를 Langfuse cloud(free tier)에 export한다. 어떤 session이 실패했는가? 왜인가?
2. domain용 LLM-judge rubric(factual correctness, tone, scope adherence)을 작성한다. 50개 trace에서 테스트한다.
3. Langfuse prompt versioning과 Phoenix trace clustering을 비교한다. 무엇이 망가졌는지 더 빨리 알려 주는 쪽은 어느 것인가?
4. Opik의 guardrail 문서를 읽는다. agent run 하나에 PII redaction guardrail을 연결한다.
5. 세 platform을 자신의 corpus에서 benchmark한다. vendor가 공개한 숫자는 무시하고 직접 측정한다.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Tracing | "Spans collector" | OTel / SDK span을 ingest하고 session별로 index |
| Prompt management | "Prompt CMS" | trace와 연결된 versioned prompt |
| LLM-as-judge | "Automated eval" | 별도 LLM이 rubric에 따라 agent output을 score |
| Session replay | "Trace playback" | debugging을 위해 과거 run을 단계별로 재생 |
| RAG relevancy | "Retrieval quality" | retrieved context가 query와 맞는지 |
| Trace clustering | "Behavioral grouping" | drift detection을 위해 비슷한 run을 cluster |
| Guardrail enforcement | "Policy at log time" | logged content의 PII/toxicity/scope check |

## Further Reading

- [Langfuse docs](https://langfuse.com/) — tracing, evals, prompt mgmt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — auto-instrumentation, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 세 platform이 모두 소비하는 schema
