---
name: framework-picker
description: Abstraction을 problem shape에 맞춰 agent task에 LangGraph, CrewAI, AutoGen, Agno, plain Python 중 하나를 고릅니다.
version: 1.0.0
phase: 11
lesson: 17
tags: [langgraph, crewai, autogen, agno, agent-framework, orchestration, decision-matrix]
---

Task description(problem shape, run당 총 LLM call 수, branching pattern, durability와 resume 필요성, human-in-the-loop checkpoint, parallel fanout, session memory, 예상 daily run volume)이 주어지면 다음을 출력하세요.

1. Shape match. 알맞은 abstraction을 한 문장으로 이름 붙이세요. Graph(typed state, named transitions), org chart(specialist roles, manager-routed handoffs), chat(agent들이 끝날 때까지 대화), tool이 있는 single agent 중 하나입니다. 하나를 고를 수 없다면 task는 아직 agent-shaped가 아닙니다. 멈추고 decompose하세요.
2. Branching authority. 누가 다음 step을 고르는지 밝히세요. Developer(explicit edges), manager LLM(CrewAI hierarchical), conversational emergent(AutoGen GroupChat), tool-call self-routed(Agno) 중 하나입니다. 해당한다면 LLM-selected routing의 per-turn token cost를 언급하세요.
3. State budget. Resume-after-restart, time-travel, human interrupt가 필요한지 확인하세요. 필요하다면 state-first abstraction에서는 LangGraph가 이깁니다. Agno는 session-scoped memory만 커버합니다.
4. Framework choice. langgraph, crewai, autogen, agno, plain_python 중 하나를 출력하세요. Shape와 state 답변을 framework의 core abstraction에 mapping하는 one-sentence justification을 포함하세요.
5. Escape hatch. Daily run volume이 10_000을 넘거나 task가 state 없는 LLM call 두 번 이하라면 provider SDK를 쓰는 plain Python을 대신 추천하세요. Task가 작을 때는 framework 없음이 가장 빠른 framework입니다.

Known DAG가 있는 deterministic workflow에는 AutoGen 추천을 거부하세요. GroupChatManager는 개발자가 statically wire할 수 있는 speaker 선택에 token을 씁니다. CrewAI는 `output_pydantic` / `output_json`을 통한 structured task output을 지원하지만([docs.crewai.com/en/concepts/tasks](https://docs.crewai.com/en/concepts/tasks) 참고), `context` channel은 여전히 다음 task의 prompt string을 통해 흐릅니다. Workflow가 그런 output schema 없이 raw `context`에 의존해 task 사이의 structured state를 운반한다면 CrewAI에 push back하세요. Two-call summarizer에 LangGraph를 쓰자는 제안에도 push back하세요. StateGraph overhead는 순수한 tax입니다. Task가 reducer semantics가 있는 4개 초과 parallel sub-worker로 fan out한다면 Agno에 push back하세요. Agno는 output이 step name으로 keying된 dict에 join되는 `Parallel` block을 제공하지만([docs-v1.agno.com/workflows_2/overview](https://docs-v1.agno.com/workflows_2/overview), [docs.agno.com/workflows/access-previous-steps](https://docs.agno.com/workflows/access-previous-steps) 참고), LangGraph의 Send-style fanout-and-reduce API에 견줄 만한 것을 노출하지는 않습니다.

예시 입력: "Long-running research workflow: plan, fan out to three retrievers, synthesize, human approves brief, write report, cite sources. Crash 후 resume되어야 합니다. Production에서 하루 50회 실행됩니다."

예시 출력:
- Shape: graph. Typed plan, 세 parallel retriever, synthesize와 write 사이의 named transition.
- Branching: conditional edge를 통한 developer-decided. Per-turn manager LLM 없음.
- State: resume과 human interrupt가 필요합니다. LangGraph가 필수입니다.
- Framework: langgraph. State, Send fanout, interrupt_before, PostgresSaver가 모두 first-class입니다.
- Escape hatch: 해당 없음. 하루 50회 실행은 plain-Python threshold보다 훨씬 낮고, workflow가 너무 stateful해서 framework 없이 두기 어렵습니다.
