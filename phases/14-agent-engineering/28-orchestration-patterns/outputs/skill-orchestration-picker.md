---
name: orchestration-picker
description: 주어진 문제에 맞는 orchestration topology(supervisor, swarm, hierarchical, debate, or none)를 선택하고 최소로 구현한다.
version: 1.0.0
phase: 14
lesson: 28
tags: [orchestration, supervisor, swarm, hierarchical, debate]
---

product domain과 task class가 주어지면 minimal topology를 선택한다.

결정:

1. 1 agent + workflow patterns(Lesson 12)로 충분한가? -> topology를 전혀 쓰지 않는다.
2. distinct responsibility를 가진 specialist가 2-4개인가? -> **supervisor-worker**.
3. latency-critical이고 specialist가 깔끔하게 handoff할 수 있는가? -> **swarm**.
4. specialist가 10개 이상이고 supervisor context budget이 실패하는가? -> **hierarchical**.
5. cost보다 accuracy가 중요하고 multi-proposer + critique가 도움이 되는가? -> **debate**(Lesson 25).

생성할 것:

1. 선택한 topology scaffold.
2. swarm의 hop counter, hierarchical의 nesting depth limit, debate의 round cap.
3. handoff별 또는 step별 observability hook(OTel GenAI span, Lesson 23).
4. "why this, not that" README section.

강한 거부 조건:

- LLM call 3개를 순차 호출하고 "multi-agent"라고 부름. 그것은 prompt chain이다.
- hop counter 없는 swarm. bouncing은 확실하다.
- branch마다 specialist 1개로 끝나는 hierarchical. flatten하라.

거부 규칙:

- single ReAct loop가 처리할 task에 multi-agent를 원하면 거부하고 Lesson 01을 제안한다.
- 2-step task에 supervisor를 원하면 거부하고 prompt chaining(Lesson 12)을 제안한다.
- domain에 compliance / audit requirement가 있으면 swarm을 거부하고 supervisor 또는 hierarchical을 제안한다.

출력: topology scaffold + decision rationale이 담긴 README. 마지막은 supervisor implementation을 위한 Lesson 13(LangGraph), handoffs-as-tools를 위한 Lesson 16(OpenAI Agents SDK), 또는 debate 세부사항을 위한 Lesson 25를 가리키는 "what to read next"로 끝낸다.
