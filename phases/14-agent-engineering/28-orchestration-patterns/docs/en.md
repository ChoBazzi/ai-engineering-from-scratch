# Orchestration Patterns: Supervisor, Swarm, Hierarchical

> 2026년 framework 전반에서 네 orchestration pattern이 반복된다: supervisor-worker, swarm / peer-to-peer, hierarchical, debate. Anthropic의 guidance: "It's about building the right system for your needs." 단순하게 시작하고, single agent와 다섯 workflow pattern만으로 부족할 때만 topology를 추가하라.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 minutes

## Learning Objectives

- 반복되는 네 orchestration pattern과 각각이 맞는 상황을 말한다.
- 2026 LangChain 권장 사항을 설명한다: tool-call-based supervision vs supervisor library.
- Anthropic의 "build the right system" rule과 그것이 topology choice를 어떻게 gate하는지 설명한다.
- 공통 scripted LLM을 대상으로 네 가지를 모두 stdlib로 구현한다.

## The Problem

팀은 필요하기 전에 "multi-agent"에 손을 뻗는다. framework 전반에서 네 pattern이 반복된다. 이름을 붙일 수 있으면 올바른 것을 고르거나 topology를 아예 생략할 수 있다.

## The Concept

### Supervisor-worker

- central routing LLM이 specialist agent로 dispatch한다.
- 결정: 자기 자신으로 loop back, specialist에게 handoff, terminate.
- specialist는 서로 대화하지 않는다. 모든 routing은 supervisor를 통한다.

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**2026 LangChain recommendation:** `create_supervisor`보다 direct tool call로 supervision을 수행한다. 더 세밀한 context engineering control을 제공한다. 각 specialist가 정확히 무엇을 볼지 직접 결정한다.

### Swarm / peer-to-peer

- agent가 shared tool surface를 통해 직접 handoff한다.
- central router가 없다.
- supervisor보다 latency가 낮다(hop이 적음).
- 추론하기 어렵다(single point of control이 없음).

Frameworks: LangGraph swarm topology, OpenAI Agents SDK handoffs (when all agents can hand off to all others).

### Hierarchical

- supervisor가 sub-supervisor를 관리하고 sub-supervisor가 worker를 관리한다.
- LangGraph에서는 nested subgraph, CrewAI에서는 nested crew로 구현된다.
- operational complexity를 대가로 큰 agent population까지 scale한다.

필요한 경우: single supervisor의 context budget이 모든 specialist description을 담을 수 없을 때.

### Debate

- parallel proposer + iterative cross-critique(Lesson 25).
- 사실상 orchestration이라기보다 verification에 가깝지만 framework에서는 topology choice로 등장한다.

### CrewAI Crew vs Flow

CrewAI는 두 deployment mode를 formalize한다.

- **Flow** — deterministic event-driven automation용(production의 recommended starting point).
- **Crew** — autonomous role-based collaboration용.

이는 위 네 pattern과 orthogonal하지만 topology에 mapping된다. Flow는 보통 supervisor 또는 hierarchical이고, Crew는 보통 LLM router가 있는 supervisor다.

### Anthropic's guidance

"Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

결정 순서:

1. Single agent + workflow patterns(Lesson 12) — 여기서 시작한다.
2. Supervisor-worker — 2-4 specialist가 있을 때.
3. Swarm — reasoning clarity보다 latency가 더 중요할 때.
4. Hierarchical — supervisor context budget이 실패할 때만.
5. Debate — cost보다 accuracy가 더 중요할 때.

### Where this pattern goes wrong

- **topology-first thinking.** multi-agent가 어떤 문제를 푸는지 식별하기 전에 "We need multi-agent"라고 말한다.
- **swarm의 bouncing handoff.** A -> B -> A -> B. hop counter를 사용한다.
- **fake hierarchy.** 실제 team은 둘뿐인데 "enterprise"라며 세 layer를 둔다. collapse하라.

## Build It

`code/main.py`는 scripted LLM을 대상으로 네 pattern을 모두 stdlib로 구현한다.

- `Supervisor` — central router.
- `Swarm` — direct handoff가 있는 peer-to-peer.
- `Hierarchical` — supervisor의 supervisor.
- `Debate` — parallel proposer + critique.

각 pattern은 같은 three-intent task(refund / bug / sales)를 처리한다. trace shape는 다르다.

실행:

```bash
python3 code/main.py
```

출력: pattern별 trace + op count. Supervisor가 가장 깔끔하고, swarm이 가장 짧고, hierarchical이 가장 깊고, debate가 가장 비싸다.

## Use It

- **LangGraph** — supervisor와 hierarchical(nested subgraph)에 사용한다.
- **OpenAI Agents SDK** — handoffs-as-tools(supervisor-shaped)에 사용한다.
- **CrewAI Flow** — production deterministic에 사용한다.
- **Custom** — debate 또는 정확한 제어를 원할 때 사용한다.

## Ship It

`outputs/skill-orchestration-picker.md`는 topology를 선택하고 구현한다.

## Exercises

1. router를 제거해 supervisor-worker를 swarm으로 변환한다. 무엇이 깨지는가? 무엇이 개선되는가?
2. swarm에 hop counter를 추가한다. handoff 3회 후 거부한다. A->B->A bouncing을 잡는가?
3. 12-specialist domain을 위한 two-level hierarchical system을 만든다. nesting 없이 context budget은 어디에서 실패하는가?
4. production-shaped workload에서 네 pattern을 profile한다. 어떤 metric(latency, cost, accuracy, debuggability)에서 어느 것이 이기는가?
5. Anthropic의 "Building Effective Agents" post를 읽는다. 자신의 production flow 각각을 네 가지 중 하나에 mapping한다. 깔끔히 mapping되지 않는 것이 있는가?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | central LLM이 specialist로 dispatch하며 specialist끼리는 대화하지 않음 |
| Swarm | "Peer-to-peer" | shared tool을 통한 direct handoff. central router 없음 |
| Hierarchical | "Supervisors of supervisors" | 큰 population을 위한 nested subgraph |
| Debate | "Proposer + critique" | parallel proposer, cross-critique(Lesson 25) |
| Tool-call-based supervision | "Supervisor without a library" | context control을 위해 supervisor를 direct tool call로 구현 |
| Crew | "Autonomous team" | CrewAI의 role-based collaboration mode |
| Flow | "Deterministic workflow" | CrewAI의 event-driven production mode |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
