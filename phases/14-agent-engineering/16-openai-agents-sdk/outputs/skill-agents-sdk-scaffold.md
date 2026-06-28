---
name: agents-sdk-scaffold
description: triage agent, handoff, input/output/tool guardrail, session store, trace processor를 갖춘 OpenAI Agents SDK 앱을 scaffold한다.
version: 1.0.0
phase: 14
lesson: 16
tags: [openai, agents-sdk, handoffs, guardrails, tracing, session]
---

product domain과 specialist agent 목록이 주어지면 OpenAI Agents SDK 앱을 scaffold한다.

다음을 만든다.

1. specialist마다 `Agent` 하나와, handoff만 가진 `triage` agent 하나(domain tool 없음).
2. typed input schema, 명확한 description(모델에 언제 사용할지 알려 줌), execution sandbox를 가진 domain tool별 `FunctionTool`.
3. triage에서 각 specialist로 가는 `Handoff`. tool name이 `transfer_to_<agent>` convention을 따르는지 확인한다.
4. PII, policy, scope를 위한 `InputGuardrail`. guardrail LLM이 main model에 비해 크지 않다면 기본은 parallel mode다. 크면 blocking을 사용한다.
5. length, PII, policy를 위한 `OutputGuardrail`. safety-critical output의 prod에서는 항상 blocking.
6. network나 filesystem을 건드리는 function tool의 per-tool guardrail.
7. `Session` store(SQLite 기본, prod는 Redis).
8. span을 OpenAI trace UI와 함께 자체 backend로 보내는 `add_trace_processor` wiring.

Hard reject:

- domain tool을 가진 triage agent. triage는 handoff만 한다. 섞으면 router decision이 흐려진다.
- input/output을 mutate하는 guardrail. guardrail은 승인하거나 거부한다. 다시 쓰지 않는다.
- 조용한 handoff loop. hop counter가 필요하다(default max 3).

거부 규칙:

- 사용자가 "가드레일 없이 빠르게만"을 원하면, paying user나 PII를 다루는 모든 product에서는 거부한다.
- product에 specialist가 2명뿐이면 triage+handoff 대신 direct classifier(Lesson 12)를 가진 `Agents` routing을 제안한다. token cost가 더 낮다.
- prod에서 tracing이 꺼져 있으면 ship을 거부한다. multi-step failure는 trace 없이는 debug할 수 없다.

출력: `agents.py`, `tools.py`, `guardrails.py`, `app.py`, 그리고 triage-agent 근거, guardrail mode, trace processor, session backend를 설명하는 `README.md`. 마지막에는 Lesson 23(OTel GenAI), Lesson 24(observability backend), 또는 Claude Agent SDK translation을 위한 Lesson 17을 가리키는 "what to read next"로 끝낸다.
