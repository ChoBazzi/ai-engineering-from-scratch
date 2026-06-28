# Production Runtimes: Queue, Event, Cron

> production agent는 여섯 runtime shape에서 실행된다: request-response, streaming, durable execution, queue-based background, event-driven, scheduled. framework를 고르기 전에 shape를 먼저 골라라. 모든 shape에서 observability는 load-bearing이다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 minutes

## Learning Objectives

- 여섯 production runtime shape를 말하고 각각을 framework / product pattern에 matching한다.
- durable execution(LangGraph)이 long-horizon task에 중요한 이유를 설명한다.
- event-driven runtime과 Claude Managed Agents가 맞는 상황을 설명한다.
- multi-step agent에서 observability-as-load-bearing 주장을 설명한다.

## The Problem

production agent는 Jupyter notebook이 드러내지 않는 방식으로 실패한다. step 37의 network timeout, voice call 중간의 user hangup, machine reboot로 죽는 cron job, memory가 고갈되는 background worker. runtime shape가 어떤 failure가 survivable한지 결정한다.

## The Concept

### Request-response

- synchronous HTTP. user가 completion을 기다린다.
- short task(<30s)에만 viable하다.
- stack: Agno(Python + FastAPI), Mastra(TypeScript + Express/Hono/Fastify/Koa).
- observability: standard HTTP access log + OTel span.

### Streaming

- progressive output을 위한 SSE 또는 WebSocket.
- LiveKit은 이를 voice/video용 WebRTC로 확장한다(Lesson 22).
- stack: streaming support가 있는 framework + SSE/WS를 처리하는 frontend.
- observability: chunk별 timing, first-token latency, tail latency.

### Durable execution

- 모든 step 후 state가 checkpoint되고 failure 시 auto-resume한다.
- AutoGen v0.4 actor model은 failure를 한 agent로 isolate한다(Lesson 14).
- LangGraph의 core differentiator(Lesson 13).
- step count를 알 수 없고 recovery cost가 높을 때 필수다.

### Queue-based / background

- job이 queue에 들어가고 worker가 가져가며 result는 webhook 또는 pub/sub으로 돌아온다.
- long-horizon agent에 필수다(Anthropic의 computer use announcement에 따르면 task당 dozens-to-hundreds of steps).
- stack: Celery(Python), BullMQ(Node), SQS + Lambda(AWS), custom.
- observability: queue depth, job별 latency distribution, DLQ size.

### Event-driven

- agent가 trigger를 subscribe한다: new email, PR opened, cron fire.
- Claude Managed Agents는 이를 out of the box로 다룬다(Lesson 17).
- CrewAI Flows(Lesson 15)는 event-driven deterministic workflow를 구조화한다.
- observability: trigger source, event-to-start latency, agent latency.

### Scheduled

- 주기적으로 실행되는 cron-shaped agent.
- failing nightly run이 다음 tick에서 resume하도록 durable execution과 결합한다.
- stack: Kubernetes CronJob + durable framework, hosted(Render cron, Vercel cron).

### 2026 deployment patterns

- **CrewAI Flows** — event-driven production용.
- **Agno** — Python microservice용 stateless FastAPI.
- **Mastra** — embedding용 server adapter(Express, Hono, Fastify, Koa).
- **Pipecat Cloud / LiveKit Cloud** — managed voice용(Lesson 22).
- **Claude Managed Agents** — hosted long-running async용.

### Observability is load-bearing

OpenTelemetry GenAI span(Lesson 23)과 Langfuse/Phoenix/Opik backend(Lesson 24)가 없으면 step 40에서 실패한 multi-step agent를 debug할 수 없다. production에서는 선택 사항이 아니다. "we debug fast"와 "we replay from scratch with more logging"의 차이다.

### Where production runtimes fail

- **wrong shape choice.** 5분짜리 task에 request-response를 고른다. user는 hang up하고 worker는 쌓이며 retry가 compound된다.
- **DLQ 없음.** dead-letter 없는 queue worker. failed job이 사라진다.
- **opaque background work.** background agent가 trace export 없이 실행된다. user가 보고하기 전까지 failure가 보이지 않는다.
- **durable state 생략.** restart를 감당할 수 없는 30초 초과 run에는 durable execution이 필요하다.

## Build It

`code/main.py`는 stdlib multi-shape demo다.

- request-response endpoint(plain function).
- streaming handler(generator).
- DLQ가 있는 queue-based worker.
- event trigger registry.
- cron-shaped scheduler.

Run it:

```bash
python3 code/main.py
```

출력: 같은 task에서 각 shape의 behavior를 보여 주는 trace 다섯 개. 같은 agent logic, 다른 outer shell이다. durable execution(여섯 번째 shape)은 LangGraph checkpointing과 함께 Lesson 13에서 의도적으로 다룬다.

## Use It

- **Request-response** — chat-style UX에 사용한다.
- **Streaming** — progressive response에 사용한다.
- **Durable** — long-horizon task에 사용한다.
- **Queue** — batch / async / long-running에 사용한다.
- **Event** — agent reactivity에 사용한다.
- **Cron** — housekeeping(memory consolidation, evals, cost reports)에 사용한다.

## Ship It

`outputs/skill-runtime-shape.md`는 task에 맞는 runtime shape를 고르고 observability requirement를 연결한다.

## Exercises

1. Lesson 01 ReAct loop를 자신의 stack에서 여섯 shape 모두로 port한다. 어떤 shape가 어떤 product surface에 맞는가?
2. queue-based demo에 DLQ를 추가한다. 10% job failure를 simulate하고 DLQ size를 드러낸다.
3. 그날의 top 20 trace를 대상으로 nightly 실행되는 cron-triggered eval agent를 작성한다.
4. backpressure가 있는 streaming을 구현한다. client가 느리면 agent를 pause한다. 이것은 turn budget과 어떻게 상호작용하는가?
5. Claude Managed Agents 문서를 읽는다. self-hosted long-horizon agent를 managed로 옮길 때는 언제인가?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Request-response | "Synchronous" | user가 기다림. short task 전용 |
| Streaming | "SSE / WS" | progressive output. 더 나은 UX. chunk별 latency 관측 가능 |
| Durable execution | "Resume from failure" | checkpointed state. 마지막 step에서 restart |
| Queue-based | "Background jobs" | producer / worker pool / DLQ |
| Event-driven | "Trigger-based" | agent가 external event에 반응 |
| DLQ | "Dead-letter queue" | failed job을 위한 parking lot |
| Claude Managed Agents | "Hosted harness" | caching + compaction을 갖춘 Anthropic-hosted long-running async |

## Further Reading

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — durable execution detail
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "dozens-to-hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
