---
name: a2a-agent-spec
description: A2A로 호출 가능해야 하는 agent의 Agent Card와 skills schema를 작성합니다.
version: 1.0.0
phase: 13
lesson: 18
tags: [a2a, agent-card, task-lifecycle, delegation]
---

agent의 capability와 의도한 collaborator가 주어지면 A2A Agent Card와 skill 정의를 작성하세요.

다음을 산출하세요.

1. Agent Card. `name`, `description`, `url`, `version`, `schemaVersion`, `capabilities`(streaming, pushNotifications), `skills[]`.
2. Skills 목록. 각 항목에는 `id`, `name`, `description`, `inputModes`, `outputModes`를 포함하세요. description에는 "Use when X. Do not use for Y." 패턴을 사용하세요.
3. Task-state 계획. 각 skill의 예상 state transition과 `input_required` path를 적으세요.
4. Signing 계획. AP2로 card에 서명할지 여부를 정하세요(외부에서 호출 가능한 agent에는 권장).
5. Transport. JSON-RPC over HTTP(기본값) 또는 gRPC. v1.0과의 backward-compat을 명시하세요.

강한 거부 조건:
- stable URL이 없는 Agent Card. discovery가 깨집니다.
- input/output mode가 선언되지 않은 skill. caller가 compatibility를 판단할 수 없습니다.
- AP2 signing 계획이 없는 externally-callable agent. impersonation vector입니다.

거부 규칙:
- agent의 use case가 단일 tool call이면 A2A scaffold를 거절하고 MCP를 권장하세요.
- agent가 노출하면 안 되는 internals(tool call trace, chain-of-thought)를 노출한다면 거절하고 opacity를 요구하세요.
- agent가 payment(AP2 use case)를 위해 A2A가 필요하다면 AP2 extension version을 확인하고 AP2가 core A2A와 별개임을 표시하세요.

산출물: 한 페이지 Agent Card JSON, 작업별 skills schema, state-transition 계획, signing 및 transport 선택. 마지막에는 agent가 약속하는 최소 v1.0 backward-compat 보장을 적으세요.
