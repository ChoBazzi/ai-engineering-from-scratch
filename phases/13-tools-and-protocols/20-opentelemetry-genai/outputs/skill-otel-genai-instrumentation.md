---
name: otel-genai-instrumentation
description: agent codebase가 OTel GenAI span을 end-to-end로 emit하도록 instrumentation 계획을 작성합니다.
version: 1.0.0
phase: 13
lesson: 19
tags: [otel, observability, gen-ai, tracing]
---

agent codebase(LLM call, tool dispatch, MCP client, sub-agent)가 주어지면 OTel GenAI instrumentation 계획을 작성하세요.

다음을 산출하세요.

1. Span hierarchy. Root `agent.invoke_agent`(INTERNAL)와 child: `llm.chat`(CLIENT), `tool.execute`(INTERNAL), `mcp.call`(CLIENT), `subagent.invoke`(INTERNAL).
2. span별 attribute checklist. `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`, `gen_ai.tool.name`, `gen_ai.agent.name`.
3. Propagation rule. 모든 remote call에 W3C traceparent를 inject하세요. MCP stdio에는 임시 필드로 `_meta.traceparent`를 사용하세요.
4. Content capture policy. 기본값은 off입니다. 어떤 env var로 활성화하는지 문서화하고 PII risk를 명시하세요.
5. Exporter choice. Jaeger / Tempo / Langfuse / Phoenix / Datadog / Honeycomb 중 선택하고 wire format은 OTLP로 둡니다.

강한 거부 조건:
- MCP 또는 sub-agent boundary를 가로지르는 trace propagation이 빠진 계획.
- content capture가 기본으로 켜진 계획. prompt와 PII가 누출됩니다.
- `gen_ai.` 또는 명시적 vendor prefix 없이 임의의 custom attribute를 emit하는 계획.

거부 규칙:
- codebase가 built-in OTel auto-instrumentation이 있는 framework(Pydantic AI, LangGraph, AgentOps)를 사용한다면 framework hook을 먼저 권장하세요.
- exporter backend가 on-prem인데 팀에 SRE support가 없다면 managed backend를 권장하세요.
- 사용자가 prod debugging을 위해 content capture를 요청하면 typed consent policy와 PII redaction pipeline 없이는 거절하세요.

산출물: span hierarchy, span별 attribute checklist, propagation rule, content capture policy, exporter choice를 담은 한 페이지 계획. 마지막에는 alert해야 할 최상위 metric을 적으세요(일반적으로 p95 `gen_ai.client.operation.duration`).
