# OpenTelemetry GenAI Semantic Conventions

> OpenTelemetry의 GenAI SIG(2024년 4월 출범)는 에이전트 telemetry의 표준 schema를 정의한다. span 이름, attribute, content-capture 규칙이 vendor 전반에서 수렴하므로 agent trace가 Datadog, Grafana, Jaeger, Honeycomb에서 같은 의미를 갖는다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 minutes

## Learning Objectives

- GenAI span category(model/client, agent, tool)를 말한다.
- `invoke_agent` CLIENT span과 INTERNAL span을 구분하고 각각 언제 적용되는지 설명한다.
- top-level GenAI attribute(provider name, request model, data-source ID)를 나열한다.
- content-capture 계약을 설명한다: opt-in, `OTEL_SEMCONV_STABILITY_OPT_IN`, external-reference 권장.

## The Problem

모든 vendor가 자체 span 이름을 만든다. 운영팀은 framework별 dashboard를 만들게 된다. OpenTelemetry의 GenAI SIG는 전체 생태계가 겨냥하는 하나의 표준을 정의해 이 문제를 해결한다.

## The Concept

### Span categories

1. **Model / client spans.** raw LLM call을 포괄한다. provider SDK(Anthropic, OpenAI, Bedrock)와 framework model adapter가 emit한다.
2. **Agent spans.** `create_agent`(agent가 생성될 때)와 `invoke_agent`(agent가 실행될 때).
3. **Tool spans.** tool invocation마다 하나씩 있으며 parent-child 관계로 agent span에 연결된다.

### Agent span naming

- Span name: 이름이 있으면 `invoke_agent {gen_ai.agent.name}`; fallback은 `invoke_agent`.
- Span kind:
  - **CLIENT** — remote agent service(OpenAI Assistants API, Bedrock Agents)에 사용.
  - **INTERNAL** — in-process agent framework(LangChain, CrewAI, local ReAct)에 사용.

### Key attributes

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — model ID.
- `gen_ai.response.model` — resolve된 model(routing 때문에 request와 다를 수 있음).
- `gen_ai.agent.name` — agent identifier.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — RAG에서 어떤 corpus 또는 store를 참조했는지.

Anthropic, Azure AI Inference, AWS Bedrock, OpenAI에 대한 기술별 convention도 존재한다.

### Content capture

기본 규칙: instrumentation은 기본적으로 input/output을 capture해서는 안 된다(SHOULD NOT). Capture는 다음 attribute를 통한 opt-in이다.

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

권장 production pattern: content를 외부(S3, 자체 log store)에 저장하고 span에는 reference(pointer ID, prose가 아님)를 기록한다. 이것은 Lesson 27의 content-poisoning 방어를 observability에 연결한 것이다.

### Stability

대부분의 convention은 2026년 3월 기준 experimental이다. stable preview에는 다음으로 opt in한다.

```bash
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+는 GenAI attribute를 자체 LLM Observability schema에 native로 mapping한다. 다른 backend(Grafana, Honeycomb, Jaeger)는 raw attribute를 지원한다.

### Where this pattern goes wrong

- **span에 전체 prompt를 capture함.** 운영자가 읽을 수 있는 trace에 PII, secret, customer data가 들어간다. 외부에 저장하라.
- **`gen_ai.provider.name`이 없음.** attribution이 없으면 multi-provider dashboard가 깨진다.
- **parent link 없는 span.** orphaned tool span이 생긴다. 항상 context를 propagate한다.
- **stability opt-in을 설정하지 않음.** backend upgrade 때 attribute 이름이 바뀔 수 있다.

## Build It

`code/main.py`는 GenAI convention에 맞춘 stdlib span emitter를 구현한다.

- GenAI attribute schema를 가진 `Span`.
- `start_span`과 nested context를 가진 `Tracer`.
- `create_agent`, `invoke_agent`(INTERNAL), tool별 span, LLM call용 `chat` span을 emit하는 scripted agent run.
- prompt를 외부에 저장하고 span에 ID를 기록하는 content-capture mode.

실행:

```bash
python3 code/main.py
```

출력: 필요한 모든 GenAI attribute를 가진 span tree와 opt-in content reference를 보여 주는 "external store".

## Use It

- **Datadog LLM Observability**(v1.37+)는 attribute를 native로 mapping한다.
- **Langfuse / Phoenix / Opik**(Lesson 24) — 생태계를 auto-instrument한다.
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel trace; GenAI attribute로 dashboard를 만든다.
- **Self-hosted** — GenAI processor와 함께 OTel Collector를 실행한다.

## Ship It

`outputs/skill-otel-genai.md`는 content-capture default와 external-reference storage를 갖춘 OTel GenAI span을 기존 agent에 연결한다.

## Exercises

1. Lesson 01 ReAct loop에 `invoke_agent`(INTERNAL) + tool별 span을 instrument한다. Jaeger instance로 보낸다.
2. "references only" mode로 content capture를 추가한다. prompt는 SQLite에 저장하고 span attribute에는 row ID만 담는다.
3. `gen_ai.data_source.id` spec을 읽는다. Lesson 09 Mem0 search에 연결한다.
4. `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`을 설정하고 collector가 attribute 이름을 바꾸지 않는지 확인한다.
5. GenAI attribute만으로 "어떤 tool error가 어떤 model과 상관되는가" dashboard를 만든다.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | schema를 정의하는 OTel working group |
| invoke_agent | "Agent span" | agent run을 나타내는 span의 이름 |
| CLIENT span | "Remote call" | remote agent service 호출용 span |
| INTERNAL span | "In-process" | in-process agent run용 span |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | retrieval이 hit한 corpus/store |
| Content capture | "Prompt logging" | message의 opt-in capture; production에서는 외부 저장 |
| Stability opt-in | "Preview mode" | experimental convention을 고정하는 env var |

## Further Reading

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 기본 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 내장 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
