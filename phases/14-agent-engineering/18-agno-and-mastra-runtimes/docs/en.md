# Agno와 Mastra: 프로덕션 런타임

> Agno(Python)와 Mastra(TypeScript)는 2026년의 프로덕션 런타임 조합이다. Agno는 마이크로초 단위의 agent instantiation과 stateless FastAPI backend를 목표로 한다. Mastra는 Vercel AI SDK 기반 위에서 agent, tool, workflow, unified model routing, composite storage를 제공한다.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## 학습 목표

- Agno의 성능 목표와 그것이 언제 중요한지 식별한다.
- Mastra의 세 가지 primitive인 Agents, Tools, Workflows와 지원되는 server adapter를 말한다.
- stateless session-scoped FastAPI backend가 Agno의 권장 프로덕션 경로인 이유를 설명한다.
- 주어진 stack에서 Agno와 Mastra 중 하나를 고른다(Python 우선 vs TypeScript 우선).

## 문제

LangGraph, AutoGen, CrewAI는 프레임워크 성격이 강하다. "내 런타임 안에서 빠르게, agent loop만" 원하는 팀은 Agno(Python)나 Mastra(TypeScript)를 선택한다. 둘 다 일부 framework-owned primitive를 raw speed와 주변 stack에 더 잘 맞는 형태와 맞바꾼다.

## 개념

### Agno

- Python runtime이며, 예전 이름은 Phi-data다.
- "No graphs, chains, or convoluted patterns — just pure python."
- 문서상의 성능 목표: agent instantiation 약 2μs, agent당 메모리 약 3.75 KiB, 약 23개 model provider.
- 프로덕션 경로: stateless session-scoped FastAPI backend. 각 요청은 새 agent를 시작하고, session state는 DB에 둔다.
- native multimodal(text, image, audio, video, file)과 agentic RAG.

이 속도 목표는 초당 수천 개의 short-lived agent가 있을 때 중요하다(chat fan-in, evaluation pipeline). 하나의 agent가 10분 동안 실행되는 경우에는 덜 중요하다.

### Mastra

- TypeScript이며 Vercel AI SDK 위에 구축되었다.
- 세 가지 primitive: **Agents**, **Tools**(Zod typed), **Workflows**.
- Unified Model Router - 94개 provider의 3,300개 이상 model(2026년 3월).
- Composite storage: memory, workflow, observability를 서로 다른 backend로 보낼 수 있다. 규모 있는 observability에는 ClickHouse를 권장한다.
- Apache 2.0이며, `ee/` 디렉터리는 source-available enterprise license다.
- Express, Hono, Fastify, Koa용 server adapter를 제공한다. Next.js와 Astro 통합은 first-class다.
- 디버깅을 위한 Mastra Studio(localhost:4111)를 제공한다.
- 1.0 기준 GitHub star 22k+, 주간 npm download 300k+(2026년 1월).

### 포지셔닝

둘 다 LangGraph가 되려는 것은 아니다. 이들이 경쟁하는 지점은 다음과 같다.

- **언어 적합성.** Python 우선 팀에는 Agno, TypeScript 우선 팀에는 Mastra.
- **런타임 ergonomics.** Agno = 거의 0에 가까운 overhead. Mastra = Vercel ecosystem과 통합.
- **관측성.** 둘 다 Langfuse/Phoenix/Opik(Lesson 24)과 통합되지만 Mastra Studio는 first-party다.

### 무엇을 고를 것인가

- **Agno** - Python backend, 많은 short-lived agent, 강한 성능 요구, FastAPI 중심 팀.
- **Mastra** - TypeScript backend, Next.js / Vercel 배포, unified multi-provider model routing, Zod typed tool.
- **LangGraph**(Lesson 13) - raw speed보다 durable state와 명시적 graph reasoning이 중요할 때.
- **OpenAI / Claude Agent SDK** - provider가 제품화한 형태를 원할 때(Lessons 16-17).

### 이 패턴이 잘못되는 지점

- **성능을 위한 성능.** workload가 요청당 느린 agent call 하나인데 "2μs"가 좋아 보여 Agno를 고른다. overhead가 병목이 아니다.
- **Ecosystem lock-in.** Mastra의 Vercel 스타일 통합은 Vercel에서는 장점이지만 다른 곳에서는 단점이다.
- **Enterprise license 혼동.** Mastra의 `ee/` 디렉터리는 source-available이며 Apache 2.0이 아니다. fork를 계획한다면 license를 읽어라.

## 직접 만들기

이 lesson은 주로 비교형이다. 두 프레임워크를 제대로 보여 주는 단일 code artifact는 적절하지 않다. `code/main.py`에는 "agent를 실행하고, 출력을 stream하고, session을 persist한다"는 최소 flow를 Agno 형태와 Mastra 형태로 각각 구현한 side-by-side toy가 있다.

실행:

```bash
python3 code/main.py
```

구조는 다르지만 기능적으로는 같은 두 trace가 나온다.

## 활용하기

- **Agno** - 속도와 FastAPI 형태가 필요한 Python backend.
- **Mastra** - 많은 provider와 workflow primitive가 필요한 TypeScript backend.
- 둘 다 first-party observability hook을 제공한다. 둘 다 Langfuse와 통합된다.

## 출시하기

`outputs/skill-runtime-picker.md`는 stack, latency budget, operational shape를 기준으로 Agno, Mastra, LangGraph, provider SDK 중 하나를 고른다.

## 연습

1. Agno 문서를 읽어라. stdlib ReAct loop(Lesson 01)를 Agno로 이식하라. 무엇이 사라졌고 무엇이 남았는가?
2. Mastra 문서를 읽어라. 같은 loop를 Mastra로 이식하라. 도구 typing에서 무엇이 바뀌는가(Zod vs 없음)?
3. Benchmark: 자신의 stack에서 agent instantiation latency를 측정하라. Agno의 2μs가 workload에 중요한가?
4. migration을 설계하라. Python에서 CrewAI를 운영해 왔다면 Agno로 옮길 때 무엇이 깨지는가?
5. Mastra의 `ee/` license terms를 읽어라. open-source fork에 영향을 줄 제한은 무엇인가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agno | "Fast Python agents" | Stateless session-scoped agent runtime |
| Mastra | "TypeScript agents on Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Multi-provider access" | 94개 provider의 3,300개 이상 model을 위한 단일 client |
| Composite storage | "Multiple backends" | memory/workflow/observability 각각을 다른 store로 보낸다 |
| Mastra Studio | "Local debugger" | agent를 introspect하는 localhost:4111 UI |
| Source-available | "Not OSS" | source reading은 허용하지만 commercial use를 제한하는 license |

## 더 읽을거리

- [Agno Agent Framework docs](https://www.agno.com/agent-framework) - 성능 목표, FastAPI 통합
- [Mastra docs](https://mastra.ai/docs) - primitive, server adapter, Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - stateful graph 대안
- [Comet Opik](https://www.comet.com/site/products/opik/) - Mastra 통합이 인용하는 observability 비교
