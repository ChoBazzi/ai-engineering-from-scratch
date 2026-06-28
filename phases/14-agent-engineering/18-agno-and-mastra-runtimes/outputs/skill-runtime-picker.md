---
name: runtime-picker
description: 주어진 stack, latency budget, operational shape에 맞는 production agent runtime(Agno, Mastra, LangGraph, provider SDK)을 고른다.
version: 1.0.0
phase: 14
lesson: 18
tags: [agno, mastra, langgraph, runtime, selection]
---

stack, latency budget, 필요한 primitive, operational shape가 주어지면 runtime을 고른다.

결정:

1. Python + FastAPI + 초당 수천 개의 short-lived agent -> **Agno**.
2. TypeScript + Next.js/Vercel + unified multi-provider -> **Mastra**.
3. Durable state, explicit graph, resume-on-failure -> **LangGraph**(Lesson 13).
4. Claude-first product이며 Claude Code harness 형태를 원함 -> **Claude Agent SDK**(Lesson 17).
5. OpenAI-first product이며 handoff + guardrail + tracing을 원함 -> **OpenAI Agents SDK**(Lesson 16).
6. Multi-agent team, actor-model concurrency, fault isolation -> **AutoGen v0.4** / **Microsoft Agent Framework**(Lesson 14).
7. Role-based collaboration 또는 event-driven deterministic workflow -> **CrewAI** Crew 또는 Flow(Lesson 15).
8. 위에 해당 없음 -> direct API call + Lesson 01의 stdlib loop.

다음을 만든다.

- 짧은 decision document: stack, latency target, 필요한 primitive, 관찰된 trade-off.
- 선택한 runtime의 최소 scaffold.
- 다른 runtime을 현재 사용 중이라면 migration plan.

Hard reject:

- workload가 요청당 느린 call 하나인데 "performance"만 보고 Agno나 Mastra를 고르는 것. performance가 병목인 경우는 드물다.
- 근거 없이 Python monorepo에서 TypeScript runtime을 고르는 것. mixed-language agent code는 operational tax다.
- stateless short task에 LangGraph를 고르는 것. checkpointer는 단순 workflow(Lesson 12)가 피하는 overhead를 추가한다.

거부 규칙:

- 사용자가 "비교를 위해 다섯 runtime 전부"를 원하면 거부한다. 자신의 workload에서 benchmark하라. framework vendor benchmark는 방향성일 뿐이다.
- 사용자가 Mastra의 `ee/` 기능을 self-host하려 하면 거부하고 license terms를 가리킨다.
- product에 장시간 비동기 작업(hours-to-days)이 필요하면 self-hosted를 거부하고 Claude Managed Agents 또는 queue-based architecture(Lesson 29)로 route한다.

출력: decision doc + scaffold + README. 마지막에는 framework 위 operational layer를 위해 Lesson 24(observability)와 Lesson 29(production runtimes)를 가리키는 "what to read next"로 끝낸다.
