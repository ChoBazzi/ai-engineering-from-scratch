# 에이전트 프레임워크 트레이드오프 — LangGraph vs CrewAI vs AutoGen vs Agno

> 모든 framework는 같은 demo를 팝니다(research agent가 report를 만듦). 그리고 같은 bug를 숨깁니다(state schema가 orchestration layer와 충돌함). 문제의 형태와 abstraction이 맞는 framework를 고르세요. 나머지는 두 번 쓰게 될 glue code입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 minutes

## 문제

LLM call이 둘 이상 필요한 task가 있습니다. Research workflow(plan, search, summarize, cite)일 수도 있습니다. Code-review pipeline(parse diff, critique, patch, validate)일 수도 있습니다. Flight를 예약하고, email을 쓰고, expense report를 제출하는 multi-turn assistant일 수도 있습니다. Framework를 고릅니다.

사흘 뒤 framework의 abstraction이 새어 나온다는 사실을 알게 됩니다. CrewAI는 role을 주지만 "researcher"가 structured plan을 "writer"에게 넘겨야 할 때 방해가 됩니다. AutoGen은 agent 간 chat을 주지만 first-class state가 없어 checkpoint가 conversation log pickle이 됩니다. LangGraph는 state graph를 주지만 agent가 무엇을 할지 알기 전에 모든 transition 이름을 붙이게 합니다. Agno는 single-agent abstraction을 주지만 세 concurrent worker로 fan out하려 하면 비명을 지릅니다.

해법은 "최고의 framework"를 고르는 것이 아닙니다. Framework의 core abstraction을 문제의 형태에 맞추는 것입니다. 이 lesson은 그 지도를 그립니다.

## 개념

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

2026년의 지형에서는 네 framework가 지배적입니다. 이들의 core abstraction은 같지 않습니다.

| 프레임워크 | 핵심 abstraction | 가장 적합한 경우 | 맞지 않는 경우 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — typed state, nodes, conditional edges, checkpointer. | 명시적 state와 human-in-the-loop interrupt가 있는 workflow, time-travel debugging이 필요한 production agent. | Topology가 알려지지 않은 느슨한 role-driven brainstorming. |
| **CrewAI** | `Crew` — roles(goal, backstory), tasks, process(sequential 또는 hierarchical). | 짧은 linear/hierarchical plan이 있는 role-playing 또는 persona-driven workflow. | Crew turn history를 넘어서는 stateful한 것, 복잡한 branching. |
| **AutoGen** | `ConversableAgent` pair — 둘 이상의 agent가 exit condition까지 turn을 주고받음. | Chat에서 thinking이 생겨나는 multi-agent *dialogue*(teacher-student, proposer-critic, actor-reviewer). | 알려진 DAG가 있는 deterministic workflow, restart를 넘는 durable state가 필요한 것. |
| **Agno** | `Agent` — 단일 LLM + tools + memory, team으로 compose 가능. | 빠르게 만들 수 있는 single agent와 lightweight team, 강한 multi-modality와 built-in storage driver. | Custom reducer가 있는 깊고 명시적으로 branch된 graph. |

### "Abstraction"이 실제로 뜻하는 것

Framework의 core abstraction은 architecture를 설명할 때 whiteboard에 그리는 것입니다.

- **LangGraph** → graph를 그립니다. Node는 step, edge는 transition, 모든 지점의 state object는 typed입니다. Mental model은 state machine입니다.
- **CrewAI** → org chart를 그립니다. 각 role에는 job description이 있고 manager가 task를 route합니다. Mental model은 소규모 specialist team입니다.
- **AutoGen** → Slack DM을 그립니다. 두 agent가 서로 message를 보내고, moderator가 필요하면 세 번째가 합류합니다. Mental model은 chat입니다.
- **Agno** → tool이 매달린 단일 box를 그립니다. Team이 필요하면 box를 나란히 둡니다. Mental model은 "batteries included agent"입니다.

### State 질문

State는 production에서 framework 선택이 가장 자주 무너지는 지점입니다.

- **LangGraph.** Typed state(`TypedDict` 또는 Pydantic model), field별 reducer, first-class checkpointer(SQLite/Postgres/Redis). Resume, interrupt, time-travel이 공짜입니다. *(Phase 11 · 16 참고.)*
- **CrewAI.** State는 `context` field를 통해 task 사이에 string으로 흐르거나, `output_pydantic`을 통해 structured하게 흐릅니다. 기본 durable per-crew store는 없습니다. Crew가 restart를 견뎌야 한다면 직접 붙여야 합니다.
- **AutoGen.** State는 chat history와 user-defined `context`입니다. Conversation transcript는 persist할 수 있지만, adapter를 직접 쓰지 않으면 arbitrary workflow state는 persist되지 않습니다.
- **Agno.** `storage=`를 통해 `Agent`에 붙는 built-in storage driver(SQLite, Postgres, Mongo, Redis, DynamoDB)가 있습니다. Conversation session과 user memory는 자동으로 persist됩니다. Full graph checkpointer가 아니라 session store입니다.

### Branching 질문

사소하지 않은 모든 agent는 branch합니다. 누가 branch를 결정하는지가 중요합니다.

- **LangGraph** — 개발자가 conditional edge로 결정합니다. Routing은 named branch를 가진 Python function입니다. Branch는 compiled graph에서 first-class이고, checkpointer는 어떤 branch가 선택됐는지 기록합니다.
- **CrewAI** — hierarchical mode에서는 manager가 결정합니다. Sequential mode에서는 build time에 개발자가 결정합니다. Routing은 task list에 implicit합니다. Manager prompt 바깥에는 first-class "if"가 없습니다.
- **AutoGen** — agent가 chat으로 결정합니다. Branching은 다음에 누가 말하는지에서 emergent하게 생깁니다. `GroupChatManager`가 next speaker를 선택합니다. `speaker_selection_method`를 직접 작성할 수 있지만 default는 LLM-driven입니다.
- **Agno** — agent가 다음에 호출할 tool을 통해 결정합니다. Team에는 coordinator/router/collaborator mode가 있습니다. 그 이상의 branching은 개발자 책임입니다.

### Observability 질문

- **LangGraph** — LangSmith 또는 어떤 OTel exporter든 OpenTelemetry로 연결됩니다. 모든 node transition은 trace span입니다. Checkpoint는 replayable trace 역할도 합니다. LangSmith가 first-party option이고 Langfuse/Phoenix도 adapter가 있습니다.
- **CrewAI** — 2025년 말부터 first-class OpenTelemetry가 있습니다. Langfuse, Phoenix, Opik, AgentOps와 통합됩니다.
- **AutoGen** — `autogen-core`를 통한 OpenTelemetry integration이 있습니다. AgentOps와 Opik connector도 있습니다. Tracing granularity는 per-node가 아니라 per-agent-message입니다.
- **Agno** — Built-in `monitoring=True` flag와 OpenTelemetry exporter가 있습니다. Session trace를 위한 Langfuse integration이 긴밀합니다.

### 비용과 latency

네 framework 모두 per-call overhead(framework logic, validation, serialization)를 더합니다. Overhead가 증가하는 대략적 순서는 Agno ≈ LangGraph < CrewAI ≈ AutoGen입니다. 차이는 framework가 추가 LLM routing을 얼마나 많이 하는지가 지배합니다. CrewAI의 hierarchical manager는 다음에 누가 갈지 결정하는 데 token을 씁니다. AutoGen의 `GroupChatManager`도 그렇습니다. LangGraph는 여러분이 `llm.invoke`를 작성한 곳에서만 token을 씁니다. Agno의 single-agent path는 얇습니다.

Run당 cost가 중요하다면 LLM-selected routing보다 explicit routing(LangGraph edge, AutoGen `speaker_selection_method`)을 선호하세요.

### 상호운용성

- **LangGraph** ↔ **LangChain** tools, retrievers, LLMs. First-class MCP adapter(MCP server에서 tool import).
- **CrewAI** ↔ tool은 `BaseTool`에서 상속합니다. LangChain tools, LlamaIndex tools, MCP tools가 모두 adapt됩니다. `allow_delegation=True`로 crew-to-crew delegation을 할 수 있습니다.
- **AutoGen** → `FunctionTool`이 어떤 Python callable도 감쌉니다. MCP adapter가 있습니다. Agent-to-agent pattern은 AG2 ecosystem에 강하게 결합되어 있습니다.
- **Agno** → `@tool` decorator 또는 BaseTool subclass를 씁니다. MCP adapter가 있습니다. Tool은 agent와 team 사이에서 공유될 수 있습니다.

## 스킬

> 주어진 agent problem에 특정 framework가 맞는 이유를 한 문장으로 설명할 수 있어야 합니다.

Build 전 체크리스트:

1. **Shape 그리기.** 이것은 graph입니까(typed state, named transitions)? Role play입니까(specialist가 work를 hand off)? Chat입니까(agent가 끝날 때까지 대화)? Tool이 있는 single agent입니까?
2. **누가 branch하는지 결정하기.** Developer-decided branching → LangGraph. Manager-agent-decided → CrewAI hierarchical. Chat-emergent → AutoGen. Tool-call-decided → Agno.
3. **State budget 확인하기.** Resume-from-checkpoint가 필요합니까? Time-travel이 필요합니까? Mid-run human interrupt가 필요합니까? 그렇다면 LangGraph가 default입니다. Agno session은 conversation-scoped state를 커버합니다.
4. **Cost budget 확인하기.** LLM-selected routing은 turn마다 추가 token을 씁니다. Agent가 하루 수천 번 실행된다면 explicit routing을 선호하세요.
5. **Framework overhead 예산 잡기.** 모든 framework는 또 하나의 dependency입니다. Task가 LLM call 두 번과 tool 하나라면 plain Python 30줄을 쓰세요. Framework 없음보다 싼 framework는 없습니다.

Graph, org chart, chat, agent box 중 하나를 그릴 수 있기 전에는 framework를 집어 들지 마세요. 실제로 필요한 것에 대해 state model과 싸우게 만드는 framework는 고르지 마세요.

## 의사결정 매트릭스

| 문제 형태 | 선호 프레임워크 | 이유 |
|---------------|---------------------|-----|
| Typed state, human approval, long-running이 있는 workflow DAG | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Distinct role이 있는 research / writing pipeline | CrewAI(sequential) 또는 LangGraph subgraphs | Role-per-task는 CrewAI에서 싸게 표현됩니다. Branching이 복잡해지면 LangGraph로 확장하세요. |
| Proposer-critic 또는 teacher-student dialogue | AutoGen | Two-agent chat이 native shape입니다. |
| Tool, session, memory가 있는 single agent | Agno | 가장 얇은 setup, built-in storage와 memory. |
| Reducer가 있는 수천 개 parallel fanout | LangGraph + `Send` | First-class parallel-dispatch API가 있는 유일한 선택지입니다. |
| Framework commitment 없는 빠른 prototype | Plain Python + provider SDK | Framework 없음이 가장 빠른 framework입니다. |

## 연습문제

1. **쉬움.** 같은 task, 즉 "Anthropic headquarters를 조사하고, 200-word brief를 작성하고, source를 cite하라"를 LangGraph(four nodes: plan, search, write, cite)와 CrewAI(three roles: researcher, writer, editor)로 구현하세요. Run당 token cost와 code line 수를 보고하세요.
2. **보통.** 같은 task를 AutoGen(researcher ↔ writer chat, `GroupChat`으로 editor 합류)과 Agno(`search_tools`, `write_tools`, session store가 있는 single agent)로 만드세요. 네 implementation을 (a) run당 cost, (b) crash 후 resume 능력, (c) write step 전 human approval 주입 능력으로 rank하세요.
3. **어려움.** 짧은 problem description(JSON: `{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`)을 받아 recommendation과 one-sentence justification을 반환하는 decision-tree script `pick_framework.py`를 만드세요. 직접 설계한 여섯 case로 검증하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Orchestration | "Agent들이 조율하는 방식" | 어떤 node/role/agent가 다음에 실행될지 결정하는 layer입니다. |
| Durable state | "Restart 후 resume" | Process death를 견디는 state입니다. Checkpoint 또는 session store에 붙어 있습니다. |
| LLM-selected routing | "모델에게 결정하게 하기" | Planner LLM이 매 turn 다음 step을 고릅니다. 유연하지만 모든 decision마다 token을 지불합니다. |
| Explicit routing | "개발자가 결정" | Python function 또는 static edge가 다음 step을 고릅니다. 싸고 auditable합니다. |
| Crew | "CrewAI team" | Roles + tasks + process(sequential 또는 hierarchical)가 하나의 runnable로 묶인 것입니다. |
| GroupChat | "AutoGen의 multi-agent chat" | Speaker selector가 있는 N agent 간 managed conversation입니다. |
| Team (Agno) | "Multi-agent Agno" | Agent set 위의 route / coordinate / collaborate mode입니다. |
| StateGraph | "LangGraph graph" | Typed-state, node, conditional-edge, checkpointer abstraction입니다. |

## 더 읽을거리

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, checkpointers, interrupts, time-travel입니다.
- [CrewAI documentation](https://docs.crewai.com/) — Crews, Flows, Agents, Tasks, Processes입니다.
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent, GroupChat, teams, tools입니다.
- [Agno documentation](https://docs.agno.com/) — Agent, Team, Workflow, storage, memory입니다.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — framework-agnostic pattern library(prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer)입니다.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 모든 framework가 꾸며 입히는 loop입니다.
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen의 design paper입니다.
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI-style persona stack이 기반으로 삼는 role-play foundation입니다.
- Phase 11 · 16 (LangGraph) — 이 lesson이 benchmark하는 framework입니다.
- Phase 11 · 19 (Reflexion) — LangGraph에는 자연스럽게 mapping되지만 CrewAI에는 어색하게 mapping되는 pattern입니다.
- Phase 11 · 22 (Production observability) — 어떤 framework를 고르든 instrument하는 방법입니다.
