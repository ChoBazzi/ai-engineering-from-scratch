# OpenTelemetry GenAI — Tool Call End-to-End 추적

> agent가 다섯 tool, 세 MCP server, 두 sub-agent를 호출합니다. 이 전체에 걸친 하나의 trace가 필요합니다. OpenTelemetry GenAI semantic conventions(v1.37 이상에서 stable attribute)는 2026년 표준이며 Datadog, Langfuse, Arize Phoenix, OpenLLMetry, AgentOps가 native로 지원합니다. 이 lesson은 필요한 attribute를 이름 붙이고, span hierarchy(agent → LLM → tool)를 따라가며, 어떤 OTel exporter에도 꽂을 수 있는 stdlib span emitter를 제공합니다.

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 학습 목표

- LLM span과 tool-execution span에 필요한 OTel GenAI attribute를 이름 붙입니다.
- agent loop, LLM call, tool call, MCP client dispatch를 포괄하는 trace hierarchy를 만듭니다.
- 어떤 content를 capture할지(opt-in), 무엇을 redact할지(default)를 결정합니다.
- tool code를 다시 작성하지 않고 local collector(Jaeger, Langfuse)로 span을 emit합니다.

## 문제

2026년 2월의 debug 사례입니다. 사용자가 "내 agent는 때로 응답에 30초가 걸리고, 다른 때는 3초가 걸린다"고 보고합니다. trace가 없습니다. log에는 LLM call만 보이고, tool dispatch도, MCP server round-trip도, sub-agent도 보이지 않습니다. 추측할 수밖에 없습니다. 결국 한 MCP server가 cold-start에서 가끔 hang된다는 것을 찾습니다.

end-to-end tracing 없이는 이것을 찾을 수 없습니다. OTel GenAI가 이를 해결합니다.

convention은 2025-2026년에 OpenTelemetry semantic-conventions group 아래에서 안정화되었습니다. Datadog, Langfuse, Phoenix, OpenLLMetry, AgentOps가 같은 span을 parse하도록 stable attribute name을 정의합니다. 한 번 instrument하고 어떤 backend로든 보낼 수 있습니다.

## 개념

### Span hierarchy

```text
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

전체가 하나의 trace id 아래에 중첩됩니다. Span id는 parent-child relationship을 연결합니다.

### Required attributes

2025-2026 semconv 기준:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — 요청한 model string(예: `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — 실제로 served된 model.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — correlation용 provider response id.

Tool span:

- `gen_ai.tool.name` — tool identifier.
- `gen_ai.tool.call.id` — 특정 call id.
- `gen_ai.tool.description` — tool description(optional).

Agent span:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span kinds

- process boundary를 넘는 call(LLM provider, MCP server)에는 `SpanKind.CLIENT`.
- agent 자체 loop step과 tool execution에는 `SpanKind.INTERNAL`.

### Opt-in content capture

기본적으로 span은 metric과 timing만 담습니다. prompt나 completion은 담지 않습니다. 큰 payload와 PII는 기본 off입니다. content를 포함하려면 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 및 특정 content-capture env var를 설정하세요. prod에서 켜기 전에 신중히 검토하세요.

### Events on spans

Token-level event는 span event로 추가할 수 있습니다.

- `gen_ai.content.prompt` — input messages.
- `gen_ai.content.completion` — output messages.
- `gen_ai.content.tool_call` — 기록된 tool call.

event는 span 안에서 시간순으로 정렬되어 세밀한 replay를 지원합니다.

### Exporters

OTel span은 다음으로 export됩니다.

- **Jaeger / Tempo.** OSS, on-prem.
- **Langfuse.** LLM-observability-specific. token usage를 시각화합니다.
- **Arize Phoenix.** eval과 tracing 결합.
- **Datadog.** commercial. `gen_ai.*` attribute를 native로 parse합니다.
- **Honeycomb.** column-oriented. query-friendly.

모두 wire format인 OTLP를 말합니다. code는 backend를 신경 쓰지 않습니다.

### Propagation across MCP

MCP client가 server를 호출할 때 W3C traceparent header를 request에 inject하세요. Streamable HTTP는 표준 header를 지원합니다. Stdio는 HTTP header를 native로 운반하지 않습니다. spec의 2026 roadmap은 JSON-RPC call에 `_meta.traceparent` field를 추가하는 방안을 논의합니다.

그 전까지는 모든 request의 `_meta`에 traceparent를 수동으로 포함하세요. server는 trace id를 log합니다.

### Metrics

span과 함께 GenAI semconv는 metric도 정의합니다.

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

call별 detail이 필요 없는 dashboard에는 이것을 사용하세요.

### AgentOps layer

AgentOps(2024년 설립)는 GenAI observability에 특화되어 있습니다. popular framework(LangGraph, Pydantic AI, CrewAI)를 wrap해 OTel span을 자동으로 emit합니다. stack이 지원 framework를 사용한다면 유용합니다. 그렇지 않으면 manual instrumentation을 사용하세요.

## 사용하기

`code/main.py`는 LLM을 호출하고 두 tool을 dispatch하며 하나의 MCP round-trip을 수행하는 agent에 대해 OTel 형태의 span을 stdout으로 emit합니다(OTLP-JSON-like format). 실제 exporter는 없습니다. 이 lesson은 span shape와 attribute set에 집중합니다. output을 OTLP-compatible viewer에 붙여 넣거나 그냥 읽으세요.

볼 부분:

- trace id가 모든 span에서 공유됩니다.
- Parent-child link는 `parentSpanId`로 encode됩니다.
- 필요한 `gen_ai.*` attribute가 채워집니다.
- content capture는 기본 off이고, 한 scenario에서 env var로 켭니다.

## 산출물

이 lesson은 `outputs/skill-otel-genai-instrumentation.md`를 만듭니다. agent codebase가 주어지면 이 skill은 instrumentation plan을 만듭니다. 어디에 span을 추가할지, 어떤 attribute를 채울지, 어떤 exporter를 target할지 포함합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. span 수를 세고 어느 것이 CLIENT인지 INTERNAL인지 식별하세요.

2. content capture(env var)를 켜고 `gen_ai.content.prompt`와 `gen_ai.content.completion` event가 나타나는지 확인하세요. PII에 대한 함의를 기록하세요.

3. tool-execution metric `gen_ai.tool.execution.duration`을 추가하고 call마다 histogram sample로 emit하세요.

4. parent agent span에서 MCP request의 `_meta.traceparent` field로 traceparent를 propagate하세요. MCP server가 같은 trace id를 볼 수 있는지 확인하세요.

5. OTel GenAI semconv spec을 읽으세요. 이 lesson의 code가 emit하지 않는 semconv attribute 하나를 식별하고 추가하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| OTel | "OpenTelemetry" | trace, metric, log를 위한 open standard |
| GenAI semconv | "GenAI semantic conventions" | LLM / tool / agent span을 위한 stable attribute name |
| `gen_ai.*` | "The attribute namespace" | 모든 GenAI attribute가 공유하는 prefix |
| Span | "Timed operation" | 시작, 종료, attribute가 있는 작업 단위 |
| Trace | "Cross-span ancestry" | trace id를 공유하는 span tree |
| SpanKind | "CLIENT / SERVER / INTERNAL" | span direction에 대한 hint |
| OTLP | "OpenTelemetry Line Protocol" | exporter용 wire format |
| Opt-in content | "Prompt / completion capture" | 기본 off. env var로 활성화 |
| traceparent | "W3C header" | 서비스 사이에서 trace context를 propagate |
| Exporter | "Backend-specific shipper" | span을 Jaeger / Datadog 등으로 보내는 component |

## 더 읽을거리

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span, metric, event의 canonical convention
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 및 tool-execution span attribute list
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent-level `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-hosted source of truth
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — production integration walk-through
