# LangGraph — State Machines for Agents

> 손으로 작성한 ReAct loop는 `while True`입니다. LangGraph로 작성한 ReAct loop는 checkpoint하고, interrupt하고, branch하고, time-travel할 수 있는 graph입니다. Agent 자체가 바뀐 것은 아닙니다. 그 주변 harness가 바뀐 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## 문제

Function-calling agent를 출시합니다. 세 turn은 잘 동작하다가 문제가 생깁니다. 모델이 500을 반환하는 tool을 시도하거나, 사용자가 작업 중간에 마음을 바꾸거나, agent가 human sign-off 없이 order를 refund하기로 결정합니다. `while True:` loop에는 hook이 없습니다. 멈출 수도, 되감을 수도, "모델이 다른 tool을 골랐다면 어땠을까"라는 branch를 만들 수도 없습니다. Demo를 넘겨 출시하는 순간 agent는 성공했거나 실패한 black box가 됩니다.

한번 보면 다음 단계는 분명합니다. Agent는 이미 state machine입니다. System prompt, message history, pending tool call, next action으로 이루어져 있습니다. State machine을 명시적으로 만드세요. "모델이 생각한다", "tool이 실행된다", "human이 승인한다"를 node로 만들고, 그 사이의 conditional transition을 edge로 만드세요. Graph가 명시되면 harness는 네 가지를 거의 공짜로 얻습니다. Checkpointing(step 사이 state 저장), interrupts(human을 위해 pause), streaming(token과 intermediate event stream), time-travel(이전 state로 되감아 다른 branch 시도)입니다.

LangGraph는 이 abstraction을 제공하는 library입니다. LangChain식 agent framework("여기 AgentExecutor가 있으니 알아서 잘해 보세요")가 아닙니다. First-class state, first-class persistence, first-class interrupt를 갖춘 graph runtime입니다. Agent loop는 손으로 쓰는 것이 아니라 그리는 것입니다.

## 개념

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph`에는 세 가지가 있습니다.

1. **State.** Graph를 따라 흐르는 typed dict(TypedDict 또는 Pydantic model)입니다. 모든 node는 full state를 받고 partial update를 반환합니다. LangGraph는 field마다 *reducer*를 사용해 이를 merge합니다. 누적해야 하는 list에는 `operator.add`, 기본값은 overwrite입니다.
2. **Nodes.** Python function `state -> partial_state`입니다. 각 node는 "model 호출", "tool 실행", "요약" 같은 discrete step입니다.
3. **Edges.** Node 사이의 transition입니다. Static edge는 한 곳으로 갑니다. Conditional edge는 router function `state -> next_node_name`을 받아 model output에 따라 graph가 branch할 수 있게 합니다.

Graph를 compile합니다. Compile은 topology를 고정하고, checkpointer를 붙이며(선택이지만 production에서는 필수), runnable을 반환합니다. Initial state와 `thread_id`로 invoke합니다. 실행의 모든 step은 `(thread_id, checkpoint_id)`를 key로 checkpoint를 persist합니다.

### 네 가지 superpower

**Checkpointing.** 모든 node transition은 새 state를 store에 기록합니다(test에는 in-memory, prod에는 Postgres/Redis/SQLite). 같은 `thread_id`로 graph를 다시 호출하면 resume합니다. Graph는 pause한 지점에서 이어집니다.

**Interrupts.** Node를 `interrupt_before=["human_review"]`로 표시하면 실행이 해당 node 전에 멈춥니다. State는 persist됩니다. API는 사용자에게 "approval 대기 중"이라고 응답합니다. 이후 같은 `thread_id`에 `Command(resume=...)`를 보내면 execution이 재개됩니다.

**Streaming.** `graph.stream(state, mode="updates")`는 발생하는 state delta를 yield합니다. `mode="messages"`는 model node 내부의 LLM token을 stream합니다. `mode="values"`는 full snapshot을 yield합니다. UI에 보여 줄 것을 고르면 됩니다.

**Time-travel.** `graph.get_state_history(thread_id)`는 전체 checkpoint log를 반환합니다. 이전 `checkpoint_id`를 `graph.invoke`에 넘기면 그 시점에서 fork합니다. Debugging("모델이 tool B를 골랐다면?")과 production trace를 replay하는 regression test에 좋습니다.

### Reducer가 핵심입니다

모든 state field에는 reducer가 있습니다. 대부분의 기본값은 괜찮습니다. 새 값이 옛 값을 덮어씁니다. 하지만 message list에는 `operator.add`가 필요합니다. 새 message가 기존 list를 대체하지 않고 append되어야 하기 때문입니다. Parallel edge도 reducer를 통해 update를 merge합니다. 두 node가 모두 `messages`를 update하는데 `Annotated[list, add_messages]`를 잊었다면 두 번째 update가 조용히 이기고 turn의 절반을 잃습니다. Reducer는 이 library에서 유일하게 미묘한 부분입니다. 제대로 설정하면 나머지는 자연스럽게 합성됩니다.

### 네 node로 만드는 ReAct graph

Production ReAct agent는 네 node와 두 edge입니다.

1. `agent` — 현재 message history로 LLM을 호출합니다. Assistant message를 반환합니다(tool_calls를 포함할 수 있음).
2. `tools` — 마지막 assistant message의 tool_calls를 실행하고, tool result를 tool message로 append합니다.
3. `agent`에서 나가는 conditional edge — 마지막 message에 tool_calls가 있으면 `tools`로, 아니면 `END`로 route합니다.
4. `tools`에서 `agent`로 돌아가는 static edge.

그게 전부입니다. 대략 40줄의 코드로 checkpointing, interrupts, streaming을 갖춘 전체 ReAct loop(Thought → Action → Observation → Thought → …)를 얻습니다.

### StateGraph vs Send(fanout)

`Send(node_name, state)`는 node가 parallel subgraph를 dispatch할 수 있게 합니다. 예를 들어 agent가 세 retriever를 동시에 query하기로 결정합니다. 각 `Send`는 target node의 parallel execution을 하나 spawn합니다. Output은 state reducer를 통해 merge됩니다. 이것이 LangGraph가 threading primitive 없이 orchestrator-workers pattern을 표현하는 방식입니다.

### Subgraph

Compiled graph는 다른 graph의 node가 될 수 있습니다. Outer graph는 단일 node를 봅니다. Inner graph는 자체 state와 자체 checkpoint를 가집니다. Team은 이 방식으로 supervisor-worker agent를 만듭니다. Supervisor graph가 user intent를 domain별 worker subgraph로 route합니다.

## 직접 만들기

### 1단계: state와 node

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages`는 message list가 overwrite되지 않고 accumulate되게 하는 reducer입니다. 이를 잊는 것이 가장 흔한 LangGraph bug입니다.

### 2단계: thread로 실행하기

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

모든 update는 `{node_name: state_delta}` dict입니다. Frontend는 이를 UI로 stream해 사용자가 "agent가 생각 중… search_web 호출 중… 결과 받음… 답변 중"을 보게 할 수 있습니다.

### 3단계: human-in-the-loop interrupt 추가

실행이 node 전에 pause되도록 표시합니다.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State, checkpoint, thread는 모두 interrupt 전후로 persist됩니다. Execution 중을 제외하면 memory에 남아 있지 않습니다.

### 4단계: debugging을 위한 time-travel

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Input으로 `None`을 넘기면 지정한 checkpoint에서 replay합니다. 값을 넘기면 그 checkpoint의 state에 update로 append한 뒤 resume합니다. 이렇게 하면 전체 conversation을 다시 실행하지 않고도 나쁜 agent run을 재현할 수 있습니다.

### 5단계: production용 checkpointer로 교체

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis, Postgres가 제공됩니다. `MemorySaver`는 test용입니다. Restart를 넘어 persist되어야 하는 것은 모두 실제 store가 필요합니다.

## 스킬

> Agent는 `while True` loop가 아니라 graph로 만듭니다.

LangGraph에 손을 뻗기 전에 60초 동안 설계하세요.

1. **Node 이름 붙이기.** 모든 discrete decision이나 side-effecting action은 node입니다. "Agent thinks", "tool runs", "reviewer approves", "response streams." 이를 나열할 수 없다면 아직 agent-shaped task가 아닙니다.
2. **State 선언하기.** 모든 list field에 reducer가 있는 최소 TypedDict를 만드세요. 모든 것을 `messages`에 밀어 넣지 마세요. Task-specific field(작업용 `plan`, `budget` counter, `retrieved_docs` list)는 top level로 올리세요.
3. **Edge 그리기.** 다음 step이 model output에 의존하지 않으면 static입니다. 모든 conditional edge에는 named branch를 가진 router function이 필요합니다.
4. **Checkpointer를 먼저 고르기.** Test에는 `MemorySaver`, 그 외에는 Postgres/Redis/SQLite를 쓰세요. Checkpointer 없이 ship하지 마세요. Checkpointer가 없으면 resume도, interrupt도, time-travel도 없습니다.
5. **Tool 실행 뒤가 아니라 실행 전에 interrupt를 결정하기.** Approval은 side-effecting node로 들어가는 edge에 둬야 피해 전에 cancel할 수 있습니다. Validation은 model 바깥 edge에 둬야 나쁜 call을 싸게 reject할 수 있습니다.
6. **기본적으로 stream하기.** UI에는 `mode="updates"`, model node 내부 token-level streaming에는 `mode="messages"`, eval 중 full snapshot에는 `mode="values"`를 쓰세요.

Checkpointer가 없는 LangGraph agent는 출시를 거부하세요. Side effect 뒤에 interrupt하는 agent도 거부하세요. `add_messages`를 reducer로 쓰지 않는 `messages` field도 거부하세요.

## 연습문제

1. **쉬움.** Calculator tool과 web-search tool로 위의 four-node ReAct graph를 구현하세요. Two-turn conversation에 대해 `list(app.get_state_history(config))`가 checkpoint를 최소 네 개 반환하는지 확인하세요.
2. **보통.** `agent` 전에 실행되어 structured `plan: list[str]`을 state에 쓰는 `planner` node를 추가하세요. `agent`가 plan step을 done으로 표시하게 하세요. Checkpoint resume을 지나며 `plan`이 사라지면(wrong reducer) test를 fail시키세요.
3. **어려움.** `Send`를 사용해 세 subgraph(`researcher`, `writer`, `reviewer`) 사이를 route하는 supervisor graph를 만드세요. 각 subgraph는 자체 state와 checkpointer를 가집니다. Outer graph에 `interrupt_before=["writer"]`를 추가해 human이 research brief를 승인하게 하세요. 이전 checkpoint에서 time-travel하면 fork된 branch만 다시 실행되는지 확인하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| StateGraph | "LangGraph graph" | Compile 전에 node와 edge를 추가하는 builder object입니다. |
| Reducer | "Field가 merge되는 방식" | Node가 해당 field에 대한 update를 반환할 때 적용되는 `(old, new) -> merged` function입니다. 기본값은 overwrite이고, `add_messages`는 append합니다. |
| Thread | "Conversation ID" | 한 session의 모든 checkpoint를 scope하는 `thread_id` string입니다. |
| Checkpoint | "Paused state" | Node transition 뒤의 full graph state를 persist한 snapshot입니다. `(thread_id, checkpoint_id)`로 keying됩니다. |
| Interrupt | "Human을 위해 pause" | `interrupt_before` / `interrupt_after`는 node boundary에서 execution을 멈춥니다. `Command(resume=...)`로 resume합니다. |
| Time-travel | "이전 step에서 fork" | `graph.invoke(None, config_with_old_checkpoint_id)`가 그 checkpoint부터 앞으로 replay합니다. |
| Send | "Parallel subgraph dispatch" | Node가 target node의 N개 parallel execution을 spawn하기 위해 반환할 수 있는 constructor입니다. |
| Subgraph | "Node로 쓰는 compiled graph" | 다른 graph의 node로 사용되는 compiled StateGraph입니다. 자체 state scope를 보존합니다. |

## 더 읽을거리

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph, reducers, checkpointers, interrupts의 canonical reference입니다.
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 이 lesson이 사용하는 mental model이며 source에서 직접 가져온 것입니다.
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — Postgres/SQLite/Redis store, checkpoint namespace, thread ID에 대한 세부 설명입니다.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)`, edit-state pattern을 다룹니다.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 모든 LangGraph agent가 구현하는 pattern입니다. Reasoning trace의 근거를 보려면 읽어 보세요.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 어떤 graph shape(chain, router, orchestrator-workers, evaluator-optimizer)를 언제 선호해야 하는지 설명합니다.
- Phase 11 · 09 (Function Calling) — 모든 LangGraph agent node가 재사용하는 tool-call primitive입니다.
- Phase 11 · 14 (Model Context Protocol) — MCP adapter를 통해 LangGraph `ToolNode`에 꽂히는 external tool discovery입니다.
- Phase 11 · 17 (Agent framework tradeoffs) — CrewAI, AutoGen, Agno 대신 LangGraph를 언제 골라야 하는지 설명합니다.
