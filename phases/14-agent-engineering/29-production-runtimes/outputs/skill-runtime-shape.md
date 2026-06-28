---
name: runtime-shape
description: production runtime shape(request-response, streaming, queue, event, cron, durable)를 선택하고 observability를 연결한다.
version: 1.0.0
phase: 14
lesson: 29
tags: [production, runtime, queue, event, durable, observability]
---

task class(expected duration, step count, trigger type, latency budget)가 주어지면 runtime shape를 선택한다.

결정:

1. < 30s, user waits -> **request-response**.
2. progressive UX 또는 voice -> **streaming**.
3. minutes to hours, user가 기다리지 않음 -> **queue-based**.
4. external event에 reactive -> **event-driven**.
5. periodic housekeeping -> **cron**.
6. 위 항목 중 restart cost가 높은 경우 -> **durable execution** 추가.

생성할 것:

1. 자신의 stack에 맞는 shape scaffold.
2. observability: OTel GenAI span(Lesson 23), backend wired(Lesson 24).
3. queue: DLQ + retry policy + queue depth metric.
4. event: explicit subscriber registry + replay path.
5. cron: overlapping run을 막는 lock file 또는 distributed lock.
6. durable: checkpointer backend + resume semantics.

강한 거부 조건:

- 5분짜리 task에 synchronous HTTP. user는 hang up하고 worker는 쌓인다.
- DLQ 없는 queue-based. failed job이 사라진다.
- trace export 없는 background work. user가 불평하기 전까지 failure가 보이지 않는다.
- "No durable state, we'll just retry." long horizon은 반드시 checkpoint해야 한다.

거부 규칙:

- product에 SLA + replay requirement가 있으면 swarm topology + non-durable runtime을 거부한다.
- task가 compliance-bound이면 audit trail 없는 event-driven을 거부한다.
- user가 cron + no lock을 원하면 거부한다. overlapping cron run은 최선의 경우 duplicate work이고 최악의 경우 data corruption이다.

출력: runtime scaffold + observability hook + SLA, retry policy, checkpointer choice가 담긴 README. 마지막은 Lesson 23(OTel), Lesson 24(observability), 또는 hosted long-running을 위한 Lesson 17(Managed Agents)을 가리키는 "what to read next"로 끝낸다.
