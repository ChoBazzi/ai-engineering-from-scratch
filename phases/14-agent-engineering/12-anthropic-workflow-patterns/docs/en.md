# Anthropic의 Workflow Patterns: 복잡함보다 단순함

> Schluntz와 Zhang(Anthropic, 2024년 12월)은 workflow(predefined path)와 agent(dynamic tool-use)를 구분합니다. 다섯 workflow pattern이 대부분의 경우를 덮습니다. Direct API call에서 시작하세요. Step을 예측할 수 없을 때만 agent를 추가하세요.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- Anthropic의 다섯 workflow pattern(prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer)을 말할 수 있습니다.
- Agent-vs-workflow 구분과 각각의 engineering cost를 설명할 수 있습니다.
- Workflow를 agent보다 선택해야 하는 때와 그 반대의 때를 식별할 수 있습니다.
- Scripted LLM을 대상으로 다섯 pattern을 모두 stdlib로 구현할 수 있습니다.

## 문제

팀들은 단일 function call이 필요한 문제에도 multi-agent framework를 선택합니다. 비용은 실제입니다. Framework는 prompt를 흐리고, control flow를 숨기며, premature complexity를 유도하는 layer를 추가합니다. Schluntz와 Zhang의 2024년 12월 글은 업계에서 가장 많이 인용되는 반론입니다. 단순하게 시작하고, complexity가 비용을 벌어낼 때만 추가하세요.

## 개념

### Workflow vs agent

- **Workflow.** Predefined code path를 통해 orchestrate되는 LLM과 tool입니다. Engineer가 graph를 소유합니다.
- **Agent.** LLM이 자기 tool과 step을 동적으로 지시합니다. Model이 graph를 소유합니다.

둘 다 자리가 있습니다. Workflow는 더 저렴하고 빠르며 debug하기 쉽습니다. Agent는 open-ended problem을 열어 주지만 failure mode를 추론하기 어렵게 만듭니다.

### Augmented LLM

다섯 pattern의 foundation은 세 capability가 연결된 하나의 LLM입니다. Search(retrieval), tools(actions), memory(persistence). 어떤 API call도 이를 사용할 수 있습니다.

### 다섯 pattern

1. **Prompt chaining.** Call 1의 output이 call 2의 input입니다. Task가 깨끗한 linear decomposition을 가질 때 사용합니다. Step 사이에 optional programmatic gate를 둘 수 있습니다.

2. **Routing.** Classifier LLM이 어떤 downstream LLM 또는 tool을 invoke할지 고릅니다. Categorically different input이 서로 다른 처리를 필요로 할 때 사용합니다(tier-1 support vs refund vs bug vs sales).

3. **Parallelization.** N개의 LLM call을 concurrent하게 실행하고 result를 aggregate합니다. 두 형태가 있습니다. Sectioning(서로 다른 chunk)과 voting(같은 prompt를 N번 실행하고 majority/synthesis).

4. **Orchestrator-workers.** Orchestrator LLM이 어떤 worker(역시 LLM)를 실행할지 동적으로 결정하고 output을 synthesize합니다. Agent loop와 비슷하지만 orchestrator는 무한 loop를 돌지 않습니다.

5. **Evaluator-optimizer.** 한 LLM이 answer를 propose하고 다른 LLM이 evaluate합니다. Evaluator가 pass할 때까지 iterate합니다. 이것은 Self-Refine(Lesson 05)의 일반화입니다.

### Workflow가 agent를 이기는 곳

- **Predictable task.** Step을 enumerate할 수 있다면 그래야 합니다.
- **Cost-bound task.** Workflow는 step count가 bounded입니다. Agent는 spiral할 수 있습니다.
- **Compliance-bound task.** Auditor는 trajectory에서 추론하는 graph가 아니라 읽을 수 있는 graph를 원합니다.

### Agent가 workflow를 이기는 곳

- **Open-ended research.** 다음 step이 이전 step의 반환값에 따라 달라질 때.
- **Variable-length task.** Step count가 unknown인, 수분에서 수시간 작업.
- **Novel domain.** 아직 올바른 workflow를 모를 때입니다. 먼저 exploration하고 나중에 codify하세요.

### Context-engineering companion

"Effective context engineering for AI agents"(Anthropic 2025)는 인접 discipline을 공식화합니다. 200k window는 container가 아니라 budget입니다. 무엇을 포함할지, 언제 compact할지, 언제 context를 키울지의 문제입니다. 이 curriculum에서는 context compression에 대한 Phase 14 lesson(renumber 이전 Phase 14 earlier lesson 06)에서 자세히 다룹니다.

## 직접 만들기

`code/main.py`는 `ScriptedLLM`을 대상으로 다섯 workflow pattern을 모두 구현합니다.

- `prompt_chain(input, steps)` — sequential.
- `route(input, classifier, handlers)` — classification + dispatch.
- `parallel_vote(prompt, n, aggregator)` — N run, aggregate.
- `orchestrator_workers(task, workers)` — orchestrator가 worker를 고릅니다.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — pass할 때까지 loop.

실행:

```bash
python3 code/main.py
```

각 pattern은 trace를 출력합니다. Pattern당 총 code line은 약 10-15줄입니다. Framework 비용은 수천 줄로 측정됩니다.

## 활용하기

- 대부분의 task에는 direct API call을 사용하세요.
- Pattern이 durable state(LangGraph), actor-model concurrency(AutoGen v0.4), role templating(CrewAI)을 진짜로 필요로 할 때만 framework를 사용하세요.
- Claude Code harness shape를 재구축하지 않고 원한다면 Claude Agent SDK를 선택하세요.

## 출시하기

`outputs/skill-workflow-picker.md`는 주어진 task description에 맞는 pattern을 선택하고, decision rationale과 workflow가 부족해질 때 agent로 refactor하는 path를 포함합니다.

## 연습 문제

1. Confidence threshold를 가진 routing을 구현하세요. Threshold 아래에서는 human에게 escalate합니다. Tier-1 support use case에서는 threshold가 어디에 놓이나요?
2. `parallel_vote`에 timeout을 추가하세요. Call 하나가 hang되면 어떻게 되나요? Missing vote가 있을 때 어떻게 aggregate하나요?
3. `evaluator_optimizer`를 bandit으로 바꾸세요. 늦게 나온 good result가 늦게 나온 bad result에 덮어써지지 않도록 iteration 사이의 top-2 output을 유지하세요.
4. Prompt chaining과 routing을 결합하세요. Router가 세 chain 중 하나를 고르게 합니다. Token cost를 single big-prompt alternative와 비교하세요.
5. Production feature 하나를 고르세요. Workflow graph를 그리세요. Step 수를 세세요. 정말 agent가 더 나을까요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| Workflow | "Predefined flow" | Engineer-owned graph of LLM and tool calls |
| Agent | "Autonomous AI" | Model-owned graph; dynamic tool direction |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory; atomic unit |
| Prompt chaining | "Sequential calls" | Call N의 output이 call N+1의 input |
| Routing | "Classifier dispatch" | 어떤 chain/model이 input을 처리할지 고름 |
| Parallelization | "Fan out" | N concurrent call. Sectioning 또는 voting으로 aggregate |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM이 specialist LLM을 동적으로 고름 |
| Evaluator-optimizer | "Proposer + judge" | Evaluator가 pass할 때까지 iterate. Self-Refine 일반화 |

## 더 읽기

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — five workflow patterns
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — companion discipline
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — stateful graph가 비용을 벌어내는 때
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — productized orchestrator-workers pattern
