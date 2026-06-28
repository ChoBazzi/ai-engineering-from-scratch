---
name: actor-runtime
description: Private state, actor별 inbox, message-only IPC, fault isolation, dead-letter queue를 갖춘 AutoGen v0.4 모양 actor runtime을 만듭니다.
version: 1.0.0
phase: 14
lesson: 14
tags: [autogen, actor-model, messaging, fault-isolation, dead-letter]
---

Multi-agent task가 주어지면 actor runtime과 필요한 agent actor를 만드세요.

생성할 것:

1. `sender`, `recipient`, `topic`, `body`, `mid`를 가진 `Message` type.
2. `receive(message, runtime)`을 가진 `Actor` base class. Actor state는 private입니다.
3. Shared queue, `send()`, `run_until_idle()`, dead-letter queue를 가진 `Runtime`. Handler의 exception은 DLQ로 보내고 propagate하지 않습니다.
4. 하나의 topology helper: RoundRobin(fixed rotation), Selector(LLM picks next), 또는 custom broadcast.
5. Message별 observability hook: Lesson 23에 따라 `gen_ai.agent.name`과 `gen_ai.operation.name`을 가진 OTel span을 emit하세요.

Hard rejects:

- Recipient가 반환할 때까지 sender를 block하는 synchronous message passing. 그것은 v0.2 model이며 fault isolation을 깨뜨립니다.
- Actor 사이의 shared mutable state. Actor는 message를 통해서만 state를 읽거나, 전혀 읽지 않아야 합니다.
- Handler exception을 propagate하는 runtime. Failure는 DLQ에 속합니다. 다른 actor는 계속 실행되게 하세요.

Refusal rules:

- Task가 fixed back-and-forth를 하는 actor 두 개뿐이라면 actor framing을 거절하고 prompt chain(Lesson 12)을 제안하세요. Actor는 actor가 3개 이상이거나 async concurrency가 있을 때 비용을 벌어냅니다.
- 사용자가 "easier debugging"을 위해 "synchronous mode"를 원하면 거절하세요. 대신 logging + tracing(Lesson 23)을 제안하세요.
- Domain이 single specialist를 가진 strict request/response라면 actor team 대신 routing(Lesson 12)을 제안하세요.

Output: `message.py`, `actor.py`, `runtime.py`, `teams.py`, 그리고 DLQ policy, topology choice, OTel span wiring을 설명하는 `README.md`. 마지막에는 actor가 negotiate한다면 Lesson 25(multi-agent debate), tracing이 필요하면 Lesson 23(OTel), forward-looking runtime을 원하면 Microsoft Agent Framework를 가리키는 "what to read next"로 끝내세요.
