---
name: agent-loop
description: tools, stop condition, turn budget을 갖춘 올바르고 최소한의 ReAct agent loop를 어떤 target language/runtime에서든 작성합니다.
version: 1.0.0
phase: 14
lesson: 01
tags: [react, agent-loop, tools, observability, stop-condition]
---

target runtime(Python async, Python sync, Node, Rust async, Go)과 tool list(name, input schema, callable)가 주어지면, 첫 시도에 올바른 ReAct agent loop를 생성하세요.

생성할 것:

1. role {user, assistant, tool, final}을 가진 message-buffer type과 target provider가 기대하는 schema(Anthropic `tool_use` / `tool_result` block, OpenAI function-calling message, Responses API reasoning channel). provider 사이에서 schema를 조용히 바꾸지 마세요.
2. name -> callable dispatch, input validation, typed result를 갖춘 tool registry. error는 반드시 catch되어 observation string으로 바뀌어야 하며, loop로 raise되면 안 됩니다.
3. explicit `finish` action, assistant turn에 tool call 없음, max turns, max total tokens, guardrail trip 중 하나가 될 때까지 실행되는 loop. primary stop은 정확히 하나만 고르고, 나머지는 safety belt로 두세요.
4. task class에 맞게 조정된 turn budget. short task 10, computer-use 200, deep research 400. 선택 이유를 명시하세요.
5. 모든 thought, action, observation, stop reason을 logging하는 trace record. runtime에 OTel SDK가 있으면 OpenTelemetry GenAI span(`invoke_agent`, `tool_call`)을 emit하세요.

강한 거부:

- turn cap 없이 loop하기. 이는 optimization 문제가 아니라 reliability 문제입니다.
- tool error를 빈 observation으로 삼키기. model이 correction할 수 있도록 failure text를 봐야 합니다.
- retrieved content를 trusted instruction으로 취급하기. 모든 tool output은 untrusted input입니다. permission은 user message만 가집니다(OpenAI CUA docs 참고).
- schema-translation layer 없이 provider를 섞기. Anthropic과 OpenAI는 tool schema와 message shape가 다릅니다.

거부 규칙:

- target이 "no framework, bash only"라면 거부하고 최소한 typed message schema를 권하세요. agent loop는 untyped shell glue로 만들기에는 error-prone합니다.
- 사용자가 "model에 feedback하지 않는 failed tool call auto-retry"를 요청하면 거부하세요. retry는 model을 거치거나(CRITIC/Self-Refine, Lesson 05), tool 자체의 idempotency contract에 속해야 합니다.
- tool list에 human-in-the-loop confirmation 없는 destructive tool이 있으면 거부하고 Lesson 09(permissions + sandboxing)를 가리키세요.

출력: language target마다 file 하나와, stop-condition 선택, turn budget 근거, step별 thought-action-observation을 보여 주는 worked trace 하나를 설명하는 `README.md`. 마지막에는 "what to read next"로 long-horizon task라면 Lesson 02(ReWOO planning), repeat-of-previous task라면 Lesson 03(Reflexion), tool이 untrusted content를 다룬다면 Lesson 27(prompt injection)을 가리키세요.
