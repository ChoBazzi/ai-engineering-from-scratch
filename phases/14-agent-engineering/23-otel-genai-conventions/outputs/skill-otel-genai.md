---
name: otel-genai
description: OpenTelemetry GenAI semantic conventions로 agent를 instrument한다. 올바른 attribute와 opt-in content capture를 갖춘 invoke_agent, chat, tool_call span을 사용한다.
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

agent runtime이 주어지면 OTel GenAI semantic conventions를 연결한다.

생성할 것:

1. agent run마다 `invoke_agent` span. remote agent service는 kind CLIENT, in-process는 INTERNAL. 이름: `invoke_agent {gen_ai.agent.name}`.
2. LLM call마다 `gen_ai.operation.name=chat`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`을 가진 `chat` span.
3. tool invocation마다 `gen_ai.tool.name`과 해당 시 `gen_ai.data_source.id`(RAG corpus / memory store)를 가진 `tool_call` span.
4. Opt-in content capture: 기본 OFF. ON일 때 input/output을 외부에 저장하고 span에 `*.reference_id`를 기록한다.
5. Context propagation: multi-process run(Claude Agent SDK CLI subprocess)이 하나의 trace로 이어지도록 W3C trace context header를 사용한다.

강한 거부 조건:

- 기본적으로 full prompt/output을 inline capture함. PII와 secret 누출 위험이 있고 spec도 위반한다.
- `gen_ai.provider.name` 누락. multi-provider dashboard가 깨진다.
- orphan tool span. 항상 active context를 통해 parent-child 관계를 설정한다.

거부 규칙:

- runtime이 process boundary를 넘어 context를 propagate할 수 없으면 거부한다. Claude Agent SDK + CLI 사용자는 multi-process trace stitching이 필요하다.
- 제품에 규제 제약(HIPAA, GDPR)이 있으면 inline content capture를 거부한다. access control이 있는 external store만 허용한다.
- backend가 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`을 설정하지 않으면 경고한다. collector upgrade 때 attribute name이 바뀔 수 있다.

출력: span structure, stability opt-in, content-capture policy를 설명하는 `tracer.py`, `attributes.py`, `content_store.py`, `README.md`. 마지막은 Lesson 24(backend: Langfuse, Phoenix, Opik) 또는 Claude Agent SDK trace-context propagation을 위한 Lesson 17을 가리키는 "what to read next"로 끝낸다.
