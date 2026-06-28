---
name: crew-or-flow
description: 주어진 task에 대해 CrewAI Crew 또는 Flow를 고르고 최소 구현을 scaffold한다.
version: 1.0.0
phase: 14
lesson: 15
tags: [crewai, crews, flows, multi-agent, role-based]
---

task description이 주어지면 Crew(autonomous) 또는 Flow(deterministic)를 고른 뒤 scaffold한다.

결정:

1. task에 SLA, compliance, deterministic replay 요구가 있는가? -> Flow.
2. task가 exploratory한가(research, first draft, brainstorm)? -> Crew.
3. task에 LLM이 순서를 고르는 specialist가 4명 이상 있는가? -> Hierarchical Crew.
4. task에 고정 순서의 specialist가 3명 이하인가? -> Sequential Crew 또는 Flow. Flow를 선호한다.

Crew의 경우 다음을 만든다.

1. Agent definition: role, goal, backstory(타이트하게, 200단어 이하), tools.
2. Task definition: description, expected_output, agent.
3. 올바른 Process(Sequential | Hierarchical)를 가진 Crew.
4. sample input에서 Crew를 실행하고 expected_output이 생성되는지 확인하는 test harness.

Flow의 경우 다음을 만든다.

1. `@start` entry function.
2. DAG를 이루는 `@listen(topic)` step.
3. 명시적 event topic. magical broadcast 없음.
4. replay harness: kickoff payload가 주어지면 deterministic하게 다시 실행한다.

Hard reject:

- backstory 없는 Crew. backstory는 load-bearing이다.
- 명시적 topic name 없는 Flow. "Implicit chaining"은 audit 목적을 무너뜨린다.
- specialist가 2명인 Hierarchical Crew. manager overhead가 비용을 정당화하지 못한다.

거부 규칙:

- 사용자가 prod-only compliance task에 Crew를 요구하면 거부하고 Flow로 migrate한다.
- 사용자가 open-ended research task에 Flow를 요구하면 거부하고 Crew로 migrate한다.
- backstory가 200단어를 넘으면 거부하고 줄이도록 요구한다. context budget은 유한하다.

출력: `agents.py`, `tasks.py`, `crew.py` 또는 `flow.py`, 그리고 결정 근거가 담긴 `README.md`. 마지막에는 observability를 위해 Lesson 24(Langfuse/AgentOps)를, Flow에 durable resume semantics가 필요하면 Lesson 13을 가리키는 "what to read next"로 끝낸다.
