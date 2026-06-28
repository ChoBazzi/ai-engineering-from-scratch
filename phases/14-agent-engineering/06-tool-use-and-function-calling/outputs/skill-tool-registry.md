---
name: tool-registry
description: JSON Schema validation, parallel dispatch, observability를 갖춘 production tool catalog와 registry를 만듭니다.
version: 1.0.0
phase: 14
lesson: 06
tags: [function-calling, tools, schema, validation, bfcl, parallel-tools]
---

task domain이 주어지면 BFCL V4 축(agentic, multi-turn, live, non-live, hallucination) 전반에서 agent가 reliable하게 사용할 수 있는 tool catalog를 생성하세요.

생성할 것:

1. Tool definitions. 각 tool에 대해 `name`(snake_case), `description`(model에게 언제 사용할지와 언제 사용하지 말아야 할지 알려 줌), typed properties가 있는 JSON Schema input, required fields, 가능한 경우 enums, numeric minimum/maximum, per-tool timeout, per-tool sandbox policy(fs surface, network, memory cap)를 제공합니다.
2. Description quality check. 각 description을 "이 tool을 다른 tool 대신 언제 골라야 하는지 model에게 알려 주는가?"로 점검하세요. 두 tool의 description이 overlap하면 거부하고 rewrite하세요.
3. Parallel-dispatch plan. realistic task마다 어떤 tool call이 independent(parallel 가능)하고 어떤 것이 sequential이어야 하는지 식별하세요. expected dispatch graph를 emit하세요.
4. Validation policy. enum check, type coercion rule(예: "accept int-as-string, reject float-as-string"), required-field enforcement. 모든 failure는 structured observation string을 반환해야 하며 loop로 raise하면 안 됩니다.
5. Observability. 각 tool은 attribute `gen_ai.tool.name`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`를 가진 OpenTelemetry GenAI `tool_call` span을 emit합니다(content policy가 요구하면 result는 inline이 아니라 reference).

강한 거부:

- generic shell/command-exec tool. 거부하고 specific verb(`git_status`, `fs_read`, `npm_test`)로 나누세요.
- parameter에 closed set of values가 있는데 enum이 없음. Enum validation은 drift를 잡는 가장 싼 방법입니다.
- 서로 다른 두 tool에 같은 description. model이 reliable하게 고를 수 없습니다.
- tool 이름만 말하는 `description`("Adds two numbers"). 대안보다 언제 골라야 하는지를 포함하세요.
- timeout 없음. 모든 tool call에는 ceiling이 있어야 합니다.

거부 규칙:

- tool list가 단일 agent에 대해 30개를 넘으면 거부하고 subagent delegation(Lesson 17)을 권하세요.
- 어떤 tool이 confirmation gate 없이 destructive action을 수행하면 거부하고 Lesson 09(permissions, sandboxing)를 가리키세요.
- task가 computer use(click, type, screenshot)이면 거부하고 Lesson 21을 가리키세요. 이는 vision-based action이 있는 별도의 tool shape입니다.

출력: Anthropic / OpenAI / Gemini SDK call에 붙여 넣을 수 있는 JSON tool catalog, dispatch-graph diagram, validation-policy document, registry가 통과해야 할 BFCL-style mini-eval.

마지막에는 Lesson 09(sandboxing), Lesson 23(OTel GenAI spans), Lesson 30(eval-driven) 중 하나를 가리키는 "what to read next" pointer를 붙이세요.
