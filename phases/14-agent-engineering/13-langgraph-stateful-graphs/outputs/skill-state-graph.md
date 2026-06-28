---
name: state-graph
description: Typed state, conditional edge, per-node checkpointing, durable resume을 갖춘 LangGraph 모양 state machine을 만듭니다.
version: 1.0.0
phase: 14
lesson: 13
tags: [langgraph, state-machine, durable, checkpointing, human-in-the-loop]
---

Target runtime, state shape, node function set, checkpointer backend가 주어지면 stateful agent graph를 만드세요.

생성할 것:

1. Typed `State`(dict 또는 Pydantic). 모든 field를 문서화하세요. Node는 state를 읽고 update를 반환합니다.
2. `add_node`, `add_edge`, `add_conditional_edges`, `set_entry`, 그리고 `START`/`END` sentinel을 가진 `StateGraph`.
3. `save(session_id, node, state)`와 `load_latest(session_id)`를 가진 `Checkpointer` interface. Default는 SQLite로 하고 Postgres/Redis/custom을 허용하세요.
4. Graph를 step through하고, 모든 node 뒤에 state를 serialize하며, human-in-the-loop를 위해 `PausedAtNode`를 catch하고, optional `state_override`와 함께 `resume_from`을 지원하는 `Runner`.
5. 세 topology helper: supervisor(central router), swarm(shared-tool handoffs), hierarchical(subgraphs).

Hard rejects:

- 명시적인 random-seed 또는 wall-clock capture 없는 non-deterministic node. Resume은 input state가 주어졌을 때 node output이 reproducible하다고 가정합니다.
- "Summary" state만 저장하는 checkpointer. Full state를 serialize하지 않으면 resume이 깨집니다.
- 모든 edge가 conditional인 graph. 가끔 branch가 있는 linear chain을 선호하세요.

Refusal rules:

- 사용자가 persistence 없는 state graph를 요청하면 거절하세요. 핵심은 durable resume입니다. Resume이 필요 없다면 Lesson 12의 workflow pattern을 사용하세요.
- 사용자가 "success일 때만 checkpoint"를 요청하면 거절하세요. Failure에도 state가 필요합니다. Debugging은 거기서 시작합니다.
- Graph가 약 30개 node를 넘으면 flat layout을 거절하고 nested subgraph를 요구하세요. Flat 30-node graph는 review할 수 없습니다.

Output: `state.py`, `graph.py`, `checkpointer.py`, `runner.py`, 그리고 state schema, checkpointer choice, resume semantics를 설명하는 `README.md`. 마지막에는 actor-model alternative는 Lesson 14, handoff/guardrail layer는 Lesson 16, graph step의 OTel span은 Lesson 23을 가리키는 "what to read next"로 끝내세요.
