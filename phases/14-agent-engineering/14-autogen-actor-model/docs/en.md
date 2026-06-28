# AutoGen v0.4: Actor Model과 Agent Framework

> AutoGen v0.4(Microsoft Research, 2025년 1월)는 agent orchestration을 actor model 중심으로 재설계했습니다. Async message exchange, event-driven agents, fault isolation, natural concurrency가 핵심입니다. Microsoft Agent Framework(public preview 2025년 10월)가 successor가 되면서 이 framework는 현재 maintenance mode입니다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 minutes

## Learning Objectives

- Actor model을 설명할 수 있습니다. Agent는 actor이고, message가 유일한 IPC이며, failure isolation은 actor별로 일어납니다.
- AutoGen v0.4의 세 API layer(Core, AgentChat, Extensions)와 각 용도를 말할 수 있습니다.
- Message delivery를 handling과 decouple하면 fault isolation과 natural concurrency가 생기는 이유를 설명할 수 있습니다.
- Python으로 stdlib actor runtime을 구현하고 two-agent code-review flow를 그 위에 port할 수 있습니다.

## 문제

대부분의 agent framework는 synchronous합니다. 한 agent가 produce하고, 다른 agent가 call stack 안에서 consume합니다. Failure는 stack을 crash시킵니다. Concurrency는 나중에 덧붙습니다. Distribution은 rewrite를 요구합니다.

AutoGen v0.4의 답은 actor model입니다. 각 agent는 private inbox를 가진 actor입니다. Message가 유일한 interaction입니다. Runtime은 delivery와 handling을 decouple합니다. Failure는 한 actor에 격리됩니다. Concurrency는 native입니다. Distribution은 transport만 다른 것입니다.

## 개념

### Actor

Actor는 다음을 가집니다.

- Private state(외부에서 직접 건드리지 못함).
- Inbox(message queue).
- Handler: `receive(message) -> effects`. Effect는 "reply", "send to other actor", "spawn new actor", "update state", "stop self"일 수 있습니다.

두 actor는 memory를 공유할 수 없습니다. Message를 보낼 수만 있습니다.

### AutoGen v0.4의 세 API layer

1. **Core.** Low-level actor framework입니다. `AgentRuntime`, `Agent`, `Message`, `Topic`. Async message exchange, event-driven입니다.
2. **AgentChat.** Task-driven high-level API입니다(v0.2의 ConversableAgent 대체). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Integration입니다. OpenAI, Anthropic, Azure, tools, memory.

### Decoupling이 중요한 이유

v0.2 model에서 `agent_a.chat(agent_b)`를 synchronous하게 호출하면 agent_b가 반환할 때까지 agent_a가 block됩니다. v0.4에서 `send(agent_b, msg)`는 message를 agent_b의 inbox에 넣고 반환합니다. Runtime이 나중에 deliver합니다. 세 결과가 있습니다.

- **Fault isolation.** Agent B가 crash해도 Agent A가 crash하지 않습니다. Runtime은 B의 handler failure를 catch하고 무엇을 할지 결정합니다(log, retry, dead-letter).
- **Natural concurrency.** 여러 message가 동시에 in flight입니다. Actor들은 inbox를 concurrent하게 처리합니다.
- **Distribution-ready.** Inbox + transport는 actor가 in-process이든 다른 host에 있든 같은 abstraction입니다.

### Topologies

- **RoundRobinGroupChat.** Agent들이 fixed rotation으로 turn을 가집니다.
- **SelectorGroupChat.** Selector agent가 conversation context를 바탕으로 다음 순서를 고릅니다.
- **Magentic-One.** Web browsing, code execution, file handling을 위한 reference multi-agent team입니다. AgentChat 위에 구축되었습니다.

### Observability

OpenTelemetry support가 built in입니다. 모든 message는 span을 emit합니다. Tool call은 2026년 OTel GenAI semantic conventions(Lesson 23)에 따라 `gen_ai.*` attribute를 가집니다.

### Status: maintenance mode

2026년 초: AutoGen v0.7.x는 research와 prototyping에 안정적입니다. Microsoft는 active development를 Microsoft Agent Framework로 옮겼습니다(public preview 2025년 10월 1일, 1.0 GA 목표는 2026년 1분기 말). AutoGen pattern은 forward port가 깔끔합니다. Actor model이 durable idea입니다.

## 직접 만들기

`code/main.py`는 stdlib actor runtime을 구현합니다.

- `Message` — `sender`, `recipient`, `topic`, `body`를 가진 typed payload.
- `Actor` — `receive(message, runtime)`을 가진 abstract.
- `Runtime` — shared queue, delivery, failure isolation을 갖춘 event loop.
- Two-actor demo: `ReviewerAgent`가 code를 review하고 `ChecklistAgent`가 checklist를 실행합니다. 둘은 consensus가 날 때까지 message를 교환합니다.

실행:

```bash
python3 code/main.py
```

Trace는 message delivery, 한 actor의 simulated failure가 다른 actor를 crash시키지 않는 모습, shared verdict로의 convergence를 보여 줍니다.

## 활용하기

- **AutoGen v0.4/v0.7**(maintenance) — research, prototyping, multi-agent pattern에 안정적입니다.
- **Microsoft Agent Framework**(public preview) — forward path입니다. 같은 actor-model idea가 refreshed API에 들어 있습니다.
- **LangGraph swarm topology**(Lesson 13) — shared-tool handoff를 통한 유사 pattern입니다.
- **Custom actor runtime** — 특정 transport(NATS, RabbitMQ, gRPC)가 필요할 때 사용합니다.

## 출시하기

`outputs/skill-actor-runtime.md`는 주어진 multi-agent task를 위한 minimal actor runtime과 team template(RoundRobin 또는 Selector)을 생성합니다.

## 연습 문제

1. Dead-letter queue를 추가하세요. Handler가 raise하면 failing message를 human inspection용으로 보관합니다. Toy에서 DLQ는 얼마나 자주 hit되나요?
2. `SelectorGroupChat`을 구현하세요. Selector actor가 conversation state를 바탕으로 next message를 누가 처리할지 고릅니다.
3. Distributed transport를 추가하세요. In-process queue를 JSON-over-HTTP server로 바꿔 actor가 별도 process에서 실행될 수 있게 하세요.
4. Message마다 OTel span(또는 no-op stand-in)을 연결하세요. Lesson 23에 따라 `gen_ai.agent.name`, `gen_ai.operation.name`을 emit하세요.
5. AutoGen v0.4 architecture post를 읽으세요. Toy를 실제 `autogen_core` API로 port하세요. Production에서 중요한 무엇을 생략했나요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| Actor | "Agent" | Private state + inbox + handler. Shared memory 없음 |
| Message | "Event" | Typed payload. Actor가 interact하는 유일한 방법 |
| Inbox | "Mailbox" | Actor별 pending message queue |
| Runtime | "Agent host" | Message를 route하고 failure를 isolate하는 event loop |
| Topic | "Channel" | Actor 사이의 named publish-subscribe route |
| Fault isolation | "Let it crash" | 한 actor failure가 다른 actor를 crash시키지 않음 |
| RoundRobinGroupChat | "Fixed-rotation team" | Agent가 순서대로 turn을 가짐 |
| SelectorGroupChat | "Context-routed team" | Selector가 다음 순서를 고름 |
| Magentic-One | "Reference team" | Web + code + file을 위한 multi-agent squad |

## 더 읽기

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — redesign post
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — AutoGen이 기본 emit하는 span
