---
name: workflow-picker
description: 주어진 task에 맞는 pattern(prompt chain, router, parallel, orchestrator-workers, evaluator-optimizer, 또는 full agent)을 선택하고 minimal implementation을 생성합니다.
version: 1.0.0
phase: 14
lesson: 12
tags: [anthropic, workflows, agents, patterns, minimal]
---

Task description이 주어지면 맞는 minimal pattern을 선택하고 가장 작은 correct implementation을 만드세요.

Decision tree:

1. Step을 enumerate할 수 있나요? -> **prompt chain** 또는 **routing**.
2. Output이 independent run 전체의 aggregation을 필요로 하나요? -> **parallelization**(sectioning 또는 voting).
3. Task마다 membership이 달라지는 specialist pool이 필요한가요? -> **orchestrator-workers**.
4. Judge가 pass할 때까지 iterative refinement가 필요한가요? -> **evaluator-optimizer**(Self-Refine shape).
5. 위에 해당하지 않거나 step count가 intermediate result에 따라 달라지나요? -> **agent loop**(Lesson 01).

생성할 것:

- Workflow의 경우: LLM + tool call을 compose하는 pure function. Framework는 쓰지 않습니다.
- Agent의 경우: Lesson 01의 ReAct loop와 task가 요구하는 tool registry.
- Decision rationale, step count, expected token cost, observable success criterion을 담은 `README.md`.

Hard rejects:

- Task가 3-step prompt chain인데 framework(LangGraph, AutoGen, CrewAI)를 선택하는 것. Over-engineering은 실제 문제를 숨깁니다.
- 3-worker orchestrator-worker를 "multi-agent"라고 설명하는 것. Worker는 agent가 아니라 LLM call입니다. 명확성을 위해 "orchestrator-workers"를 사용하세요.
- Stop condition 없는 evaluator-optimizer. `max_iter`와 "fail-pass-through" fallback이 없으면 loop가 무기한 돌 수 있습니다.

Refusal rules:

- 사용자가 "multi-agent"를 요청했지만 task가 실제로 router라면 거절하고 이름을 바꾸세요. Multi-agent label은 routing에는 필요 없는 operational cost(coordination, debugging, evals)를 동반합니다.
- 사용자가 open-ended research task에 workflow를 원하면 거절하고 turn budget이 있는 agent를 제안하세요. Workflow는 predictable trajectory에 적합합니다.
- 사용자가 2-step task에 agent를 원하면 거절하고 prompt chaining을 제안하세요. Agent는 latency와 failure mode를 추가합니다. 필요할 때만 사용하세요.

Output: pattern choice + minimal code + README. 마지막에는 durable state가 중요하면 Lesson 13(LangGraph), handoff와 guardrail은 Lesson 16(OpenAI Agents SDK), 결국 agent를 고르는 중이라면 Lesson 01을 가리키는 "what to read next"로 끝내세요.
