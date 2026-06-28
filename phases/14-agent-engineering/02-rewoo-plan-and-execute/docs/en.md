# ReWOO and Plan-and-Execute: Decoupled Planning

> ReAct는 thought와 action을 한 stream에 interleave합니다. ReWOO는 둘을 분리합니다. 앞에서 큰 plan 하나를 만들고, 그다음 실행합니다. token은 5배 적게 쓰고, HotpotQA accuracy는 +4%이며, planner를 7B model로 distill할 수 있습니다. Plan-and-Execute가 이를 일반화했고, Plan-and-Act는 web navigation으로 확장했습니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 minutes

## Learning Objectives

- ReWOO의 Planner / Worker / Solver 분리가 ReAct의 interleaved loop보다 token을 절약하고 robustness를 높이는 이유를 설명합니다.
- plan DAG, dependency-ordered executor, worker output을 조합하는 solver를 모두 stdlib로 구현합니다.
- 2026년 "five workflow patterns" framing(Anthropic)을 사용해 task를 plan-then-execute로 실행할지 interleaved ReAct로 실행할지 결정합니다.
- long-horizon web 또는 mobile task에 Plan-and-Act의 synthetic plan data가 필요한 때를 인식합니다.

## The Problem

ReAct의 interleaved thought-action-observation loop는 단순하고 유연하지만, 각 tool call이 이전 thought를 포함한 전체 prior context를 들고 다녀야 합니다. token 사용량은 depth에 대해 quadratic하게 증가합니다. 더 나쁜 점은 tool이 중간에 실패하면 모델이 error observation에서 전체 plan을 다시 유도해야 한다는 것입니다.

ReWOO(Xu et al., arXiv:2305.18323, May 2023)는 이 문제를 보고 베팅을 바꿨습니다. 전체를 먼저 계획하고, evidence를 병렬로 가져오고, 마지막에 answer를 조합합니다. planning을 위한 LLM call 하나, evidence를 위한 N개 tool call(병렬 가능), solve를 위한 LLM call 하나입니다. tradeoff는 flexibility가 줄어드는 대신(plan이 static) token efficiency와 failure mode의 명확성이 크게 좋아진다는 것입니다.

## The Concept

### 세 가지 역할

```text
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner는 DAG를 만듭니다. 각 node는 tool, arguments, 그리고 어떤 earlier node에 의존하는지(`#E1`, `#E2` 같은 reference)를 명시합니다. Workers는 topological order로 node를 실행합니다. Solver는 모든 것을 함께 엮습니다.

### token이 5배 적은 이유

ReAct는 step count에 따라 prompt length가 선형으로 커집니다. step 10에서 prompt에는 thought 1, action 1, observation 1, thought 2, action 2, observation 2 등이 모두 들어 있습니다. 각 intermediate step도 original prompt를 중복해서 포함합니다.

ReWOO는 planner prompt 하나(큼), 작은 worker prompt N개(각각 tool call뿐이며 chain 없음), solver prompt 하나를 지불합니다. HotpotQA에서 논문은 token을 약 5배 적게 쓰면서 absolute accuracy +4를 측정했습니다.

### 더 robust한 이유

ReAct에서 worker 3이 실패하면 loop는 error에서 mid-stream으로 빠져나오며 추론해야 합니다. ReWOO에서는 worker 3이 error string을 반환합니다. solver는 original plan과 함께 이를 context에서 보고 graceful하게 degrade할 수 있습니다. failure localization은 per-step이 아니라 per-node입니다.

### Planner distillation

논문의 두 번째 결과는 planner가 observation을 보지 않기 때문에, 175B teacher의 planner output으로 7B model을 fine-tune할 수 있다는 것입니다. 작은 model이 planning을 맡고 큰 model은 inference에 필요하지 않습니다. 이는 이제 표준입니다. 2026년 production agent 다수는 작은 planner와 큰 executor를 쓰거나 그 반대로 구성합니다.

### Plan-and-Execute (LangChain, 2023)

LangChain 팀의 2023년 8월 글은 ReWOO를 Plan-and-Execute라는 pattern name으로 일반화했습니다. up-front planner가 step list를 내고, executor가 각 step을 실행하며, optional replanner가 result를 본 뒤 수정할 수 있습니다. observation을 다시 planning으로 가져오므로 ReWOO보다 ReAct에 더 가깝지만 token saving은 유지합니다.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act는 이 pattern을 long-horizon web 및 mobile agent로 확장합니다. 핵심 기여는 synthetic plan data입니다. labeled trajectory generator가 plan이 명시된 training data를 만듭니다. WebArena류 task에서 단일 ReAct trajectory가 coherence를 잃는 30-50 step 이후에도 계속 작동하는 planner model을 fine-tune하는 데 사용됩니다.

### 무엇을 고를지

| Pattern | When |
|---------|------|
| ReAct | 짧은 task, unknown environment, reactive exception handling 필요 |
| ReWOO | tool이 알려진 structured task, token-sensitive, parallelizable evidence |
| Plan-and-Execute | ReWOO와 비슷하지만 partial execution 후 replanning 필요 |
| Plan-and-Act | long-horizon(>30 steps), web/mobile/computer-use |
| Tree of Thoughts | search에 비용을 낼 가치가 있음(Lesson 04) |

Anthropic의 2024년 12월 guidance는 가장 단순한 것부터 시작하라고 말합니다. task가 tool call 하나와 summary 하나라면 ReWOO를 만들지 마세요. task가 40-step research assignment라면 ReAct만으로 처리하지 마세요.

## Build It

`code/main.py`는 toy ReWOO를 구현합니다.

- `Planner` - prompt에서 plan DAG를 내는 scripted policy.
- `Worker` - registry를 통해 각 node의 tool call을 dispatch.
- `Solver` - evidence를 읽고 final answer를 생성하는 scripted composition.
- Dependency resolution - `#E1` 같은 reference를 earlier worker output으로 대체.

demo는 "프랑스 수도의 인구를 백만 단위로 반올림하면 얼마인가?"에 답합니다. 두 step plan을 사용합니다. (1) capital lookup, (2) population lookup, 그다음 solve.

실행:

```bash
python3 code/main.py
```

trace는 먼저 전체 plan을 보여 주고, 그다음 worker result, solver composition을 보여 줍니다. token count(rough character count를 출력함)를 ReAct-style interleaved run과 비교하세요. 이런 structured task에서는 ReWOO가 이깁니다.

## Use It

LangGraph는 Plan-and-Execute를 recipe로 제공합니다(ReAct에는 `create_react_agent`, plan-execute에는 custom graph). CrewAI의 Flows는 이 pattern을 직접 encode합니다. task를 upfront로 정의하고 Flow DAG가 실행합니다. Plan-and-Act의 synthetic data approach는 아직 대부분 research입니다. runtime pattern(explicit plan DAG)은 LangGraph와 CrewAI Flows를 통해 production에 배포됩니다.

## Ship It

`outputs/skill-rewoo-planner.md`는 tool catalog가 주어졌을 때 user request에서 ReWOO plan DAG를 생성합니다. executor에 넘기기 전에 plan을 검증합니다(acyclic, every reference resolved, every tool exists).

## Exercises

1. independent plan node의 worker execution을 parallelize하세요. 2개 parallel group을 가진 6-node DAG에서 무엇을 얻나요?
2. worker가 error를 반환하면 발동하는 replanner node를 추가하세요. ReWOO를 Plan-and-Execute로 만드는 가장 작은 변화는 무엇인가요?
3. `Planner`를 small model(7B class)로 바꾸고 `Solver`는 frontier model에 둡니다. end-to-end quality를 비교하세요. 어디에서 분리가 실패하나요?
4. ReWOO paper의 planner distillation에 관한 Section 4를 읽으세요. 175B -> 7B 결과를 개념적으로 재현해 보세요. 어떤 training data가 필요하며 plan quality는 어떻게 score하나요?
5. toy를 Plan-and-Act의 trajectory shape로 port하세요. plan은 DAG가 아니라 sequence입니다. 어떤 tradeoff가 달라지나요?

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| ReWOO | "Reasoning without observations" | 먼저 plan하고, evidence를 병렬로 가져온 뒤 solve - planning prompt에 observation 없음 |
| Plan-and-Execute | "LangChain's plan-execute pattern" | execution 후 optional replanner node가 있는 ReWOO |
| Plan-and-Act | "Scaled plan-execute" | long-horizon task를 위한 synthetic plan training data와 explicit planner/executor split |
| Evidence reference | "#E1, #E2, ..." | dispatch 시 prior worker output으로 대체되는 plan-node placeholder |
| Planner distillation | "Small planner, big executor" | large teacher의 planner trace로 small model fine-tune |
| Token efficiency | "Fewer round trips" | 논문에서 HotpotQA 기준 ReAct 대비 token 5배 감소 |
| DAG executor | "Topological dispatcher" | dependency order로 plan node를 실행하며 각 level에서 parallel 가능 |

## Further Reading

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) - 표준 논문
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) - synthetic plan을 사용하는 scaled planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) - framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 작동하는 가장 단순한 pattern 선택
