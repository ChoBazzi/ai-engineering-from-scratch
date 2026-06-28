# LangGraph: Stateful Graph와 Durable Execution

> LangGraph는 low-level stateful orchestration의 2026년 reference입니다. Agent는 state machine이고, node는 function이며, edge는 transition이고, state는 immutable하며 모든 step 뒤에 checkpoint됩니다. 어떤 failure에서도 멈춘 바로 그 위치에서 resume합니다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- LangGraph의 core model을 설명할 수 있습니다. Immutable state, function node, conditional edge, post-step checkpoint를 가진 state machine입니다.
- Docs가 강조하는 네 capability(durable execution, streaming, human-in-the-loop, comprehensive memory)를 말할 수 있습니다.
- LangGraph가 지원하는 세 orchestration topology(supervisor, peer-to-peer(swarm), hierarchical(nested subgraphs))를 설명할 수 있습니다.
- Immutable state, conditional edge, checkpoint/resume cycle을 가진 stdlib state graph를 구현할 수 있습니다.

## 문제

Agent와 workflow는 같은 문제를 공유합니다. 40-step run이 step 38에서 실패하면 처음부터 다시 시작하는 것이 아니라 step 38에서 resume하고 싶습니다. Second-class state model은 fresh run을 가정하는 library 주변에 operator가 retry를 덧대게 만듭니다.

LangGraph의 설계 답변은 이렇습니다. State는 first-class typed object이고, mutation은 명시적이며, checkpoint는 모든 node 뒤에 persist됩니다. Resume은 `load_state(session_id)` call입니다.

## 개념

### Graph

Graph는 다음으로 정의됩니다.

- **State type.** 모든 node가 읽고 mutate하는 typed dict(또는 Pydantic model).
- **Nodes.** Pure function `(state) -> state_update`. Update는 return 후 state에 merge됩니다.
- **Edges.** Node 사이의 conditional 또는 direct transition.
- **Entry and exit.** `START`와 `END` sentinel node가 boundary를 표시합니다.

예: `classify`, `refund`, `bug`, `sales`, `done` node를 가진 agent는 routing workflow를 graph로 표현한 것입니다.

### Durable execution

각 node가 반환한 뒤 runtime은 state를 serialize하고 checkpointer(SQLite, Postgres, Redis, custom)에 씁니다. Step N에서 실패하면 runtime은 `resume(session_id)`로 정확한 state를 가지고 step N+1부터 이어갈 수 있습니다.

LangGraph docs는 이 점이 중요한 production user를 명시적으로 강조합니다. Klarna, Uber, J.P. Morgan입니다. Claim은 graph shape 자체가 아니라 graph shape와 checkpointing이 결합되어 recovery를 저렴하게 만든다는 것입니다.

### Streaming

모든 node는 partial output을 yield할 수 있습니다. Graph는 per-node-delta event를 caller에 stream하므로 UI가 graph 실행 중 update됩니다.

### Human-in-the-loop

Node 사이에서 state를 inspect하고 modify합니다. 구현 방식은 critical node 앞에서 pause하고, state를 human에게 surface하고, modification을 받아 resume하는 것입니다. Checkpointer가 이미 state를 serialize하고 있으므로 이것이 쉽습니다.

### Memory

Short-term(run 내부, state 안의 conversation history)과 long-term(run 사이, checkpointer와 별도 long-term store를 통해 persistent)입니다. LangGraph는 tool을 통해 external memory system(Mem0, custom)과 integrate합니다.

### 세 topology

1. **Supervisor.** Central router LLM이 specialist subagent로 dispatch합니다. `langgraph-supervisor`의 `create_supervisor()`입니다. 다만 LangChain team은 2026년에 더 많은 context control을 위해 tool call을 직접 사용하는 방식을 권장합니다.
2. **Swarm / peer-to-peer.** Agent들이 shared tool surface를 통해 직접 hand off합니다. Central router가 없습니다.
3. **Hierarchical.** Supervisor가 sub-supervisor를 관리하며 nested subgraph로 구현됩니다.

### 이 패턴이 잘못되는 지점

- **Checkpoint가 너무 작음.** Conversation turn만 checkpoint하면 tool state와 memory write를 recover할 수 없습니다. Full state가 serialize되어야 합니다.
- **Non-deterministic node.** Resume은 node input이 같은 state update를 만든다고 가정합니다. Random seed, wall-clock, external API는 capture되어야 합니다.
- **Conditional edge의 과사용.** 모든 edge가 conditional인 graph는 추론할 수 없는 state machine입니다. 가끔 branch가 있는 linear chain을 선호하세요.

## 직접 만들기

`code/main.py`는 stdlib stateful graph를 구현합니다.

- `State` — `messages`, `step`, `route`, `output`, `human_approval`을 가진 typed dict.
- `Node` — state를 받아 update dict를 반환하는 callable.
- `StateGraph` — nodes + edges + conditional edges + run + resume.
- `SQLiteCheckpointer`(in-memory fake) — 모든 node 뒤에 state를 serialize합니다. `load(session_id)`가 restore합니다.
- Demo graph: classify -> branch(refund / bug / sales) -> human gate -> send.

실행:

```bash
python3 code/main.py
```

Trace는 첫 run이 human gate에서 실패하고, persistence 후 resume하여 final output을 생성하는 과정을 보여 줍니다.

## 활용하기

- **LangGraph** — reference이며 production-ready입니다. `create_react_agent`, `create_supervisor`를 사용하거나 직접 graph를 만드세요.
- **AutoGen v0.4**(Lesson 14) — high-concurrency scenario를 위한 actor model alternative입니다.
- **Claude Agent SDK**(Lesson 17) — built-in session store가 있는 managed harness입니다.
- **Custom** — state shape 또는 checkpointer backend를 정확히 제어해야 할 때 사용합니다.

## 출시하기

`outputs/skill-state-graph.md`는 checkpointing과 resume이 연결된 LangGraph 모양 state graph를 target runtime 어디에나 생성합니다.

## 연습 문제

1. Classification confidence가 threshold 아래일 때 `classify`에서 `end`로 가는 conditional edge를 추가하세요. Human이 `route`를 수동 설정한 뒤 run을 resume하세요.
2. SQLite-like fake를 실제 SQLite checkpointer로 바꾸세요. Step별 serialization overhead를 측정하세요.
3. Parallel edge를 구현하세요. 두 node를 concurrent하게 실행하고 custom reducer로 merge합니다. Immutable state는 여기서 무엇을 제공하나요?
4. `langgraph-supervisor` reference를 읽으세요. Toy를 `create_supervisor`로 port하세요. Trace shape를 비교하세요.
5. Streaming을 추가하세요. 각 node가 실행 중 partial state를 yield하게 하세요. Delta가 도착하는 대로 print하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| State graph | "Agent as state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | 모든 node 뒤에 state를 serialize하며 resume을 가능하게 함 |
| Reducer | "State merger" | Current state와 node update를 결합하는 function |
| Conditional edge | "Branch" | State의 function으로 선택되는 edge |
| Subgraph | "Nested graph" | 다른 graph 안에서 node로 쓰이는 graph |
| Durable execution | "Resume from failure" | Last successful node에서 exact state로 restart함 |
| Supervisor | "Router LLM" | Specialist subagent를 위한 central dispatcher |
| Swarm | "P2P agents" | Central router 없이 shared tool을 통해 hand off하는 agent들 |

## 더 읽기

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — supervisor pattern API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store and subagents
