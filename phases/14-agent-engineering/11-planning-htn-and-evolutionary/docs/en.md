# HTN과 Evolutionary Search로 Planning하기

> Symbolic planning은 plan이 증명 가능하게 correct해야 하는 경우를 처리합니다. Evolutionary code search는 fitness function을 machine-check할 수 있는 경우를 처리합니다. ChatHTN(2025)과 AlphaEvolve(2025)는 각각이 LLM과 결합될 때 무엇을 열어 주는지 보여 줍니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 minutes

## Learning Objectives

- Hierarchical Task Network의 tasks, methods, operators, preconditions, effects를 설명할 수 있습니다.
- ChatHTN의 hybrid loop(symbolic search와 LLM fallback decomposition)를 설명할 수 있습니다.
- AlphaEvolve의 evolutionary loop와 그것이 programmatic evaluator가 있을 때만 작동하는 이유를 설명할 수 있습니다.
- Toy HTN planner와 toy evolutionary search를 stdlib로 구현할 수 있습니다.

## 문제

ReWOO(Lesson 02), Plan-and-Execute, ReAct는 대부분의 agent planning을 다룹니다. 하지만 두 경우는 잘 다루지 못합니다.

1. **Provable correctness가 필요한 plan.** Scheduling, flight pathing, compliance workflow에서는 plan이 구성상 sound해야 합니다. 가끔 step을 hallucinate하는 fluent LLM plan은 받아들일 수 없습니다.
2. **Machine-checkable fitness function이 있는 optimization.** Matrix multiplication, scheduling heuristic, compiler pass에서는 목표가 "correct plan"이 아니라 "best plan"입니다.

HTN planning과 AlphaEvolve는 서로 다른 두 문제를 해결합니다. 둘 다 LLM을 replacement가 아니라 amplifier로 사용합니다.

## 개념

### Hierarchical Task Networks

HTN은 다음으로 구성됩니다.

- **Tasks** — compound(분해 대상)와 primitive(직접 실행 가능).
- **Methods** — compound task를 subtask로 분해하는 방법이며 precondition을 가집니다.
- **Operators** — precondition과 effect를 가진 primitive action.
- **State** — fact의 집합.

Planning: goal task와 initial state가 주어졌을 때, precondition이 순서대로 만족되는 primitive operator로 분해를 찾습니다.

HTN은 LLM보다 오래되었고, 여전히 provably-correct plan의 reference입니다.

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN(arXiv:2505.11814)은 symbolic HTN과 LLM query를 interleave합니다.

1. Existing method로 현재 compound task를 분해해 봅니다.
2. 적용 가능한 method가 없으면 LLM에 묻습니다. "`task`를 state `s`에서 어떻게 분해하겠는가?"
3. LLM response를 candidate subtask로 translate합니다.
4. Operator schema를 기준으로 validate하고 invalid decomposition은 reject합니다.
5. Recurse합니다.

논문의 핵심 claim은 생성되는 모든 plan이 provably sound하다는 것입니다. LLM suggestion은 direct plan edit가 아니라 candidate decomposition으로만 들어오기 때문입니다. Symbolic layer가 correctness를 소유하고, LLM은 method library를 확장합니다.

Online method learning(OpenReview `gwYEDY9j2x`, 2025 follow-up)은 LLM-produced decomposition을 regression으로 generalize하는 learner를 추가합니다. LLM query frequency를 최대 75%까지 줄입니다.

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve(arXiv:2506.13131, DeepMind, 2025년 6월)는 다른 종류입니다. Gemini 2.0 Flash/Pro ensemble이 orchestrate하는 evolutionary code search입니다.

Loop:

1. Seed program + programmatic evaluator(fitness score 반환)로 시작합니다.
2. LLM ensemble이 mutation을 제안합니다.
3. Mutation을 evaluator로 실행합니다.
4. Best를 유지하고 다시 mutate합니다.

Published wins:

- 56년 만의 4x4 complex matrix multiplication에서 Strassen 개선(48 scalar multiplication).
- Borg scheduling heuristic을 통해 Google compute 0.7% 회수.
- Frontier workload에서 FlashAttention 32% speedup.

강한 제약: fitness function은 반드시 machine-checkable이어야 합니다. Prose answer 위의 evolutionary search는 수렴하지 않습니다.

### 무엇을 언제 쓸까

| Problem class | 사용 | 이유 |
|---------------|------|------|
| Hard constraint가 있는 scheduling | HTN + ChatHTN | Provable soundness |
| Compiler optimization | AlphaEvolve | Machine-checkable fitness |
| Multi-step task execution | ReAct / ReWOO | Formal guarantee 없이 LLM in the loop |
| Test가 있는 code improvement | AlphaEvolve | Test가 evaluator임 |
| Policy-bound automation | HTN | Preconditions가 policy를 encode함 |

### 이 패턴이 잘못되는 지점

- **Operator 없는 HTN.** Precondition/effect schema가 없으면 soundness claim이 무너집니다. ChatHTN의 "LLM suggests decomposition"은 invalid move를 reject할 schema가 필요합니다.
- **Real evaluator 없는 AlphaEvolve.** "LLM에게 code가 더 나은지 묻기"는 fitness function이 아닙니다. Evaluator는 deterministic하고 빨라야 합니다.
- **Over-engineering.** 대부분의 agent task에는 둘 다 필요 없습니다. 먼저 ReAct 또는 ReWOO를 선택하세요.

## 직접 만들기

`code/main.py`는 두 toy를 구현합니다.

- Operator, method, precondition, effect를 가진 stdlib HTN planner와, compound task에 맞는 method가 없을 때 작동하는 `LLMFallback`. 이 "LLM"은 offline으로 실행되도록 scripted decomposer입니다.
- Arithmetic program 위의 stdlib evolutionary search: test set에서 `|f(x) - target|`을 최소화하는 expression을 키웁니다. Evaluator는 deterministic합니다.

실행:

```bash
python3 code/main.py
```

Trace는 compound task를 분해하는 HTN planner(mid-plan LLM fallback 포함)와 target expression으로 수렴하는 evolutionary loop를 보여 줍니다.

## 활용하기

- **HTN planners** — `pyhop`, `SHOP3`, 또는 domain-specific policy enforcement를 위한 자체 구현.
- **ChatHTN** — research code입니다. Pattern(symbolic + LLM fallback)은 어떤 HTN planner에도 깨끗하게 port됩니다.
- **AlphaEvolve** — DeepMind paper입니다. Pattern(ensemble + evaluator)은 재현 가능합니다. OpenEvolve와 비슷한 open-source fork가 등장하고 있습니다.
- **Agent frameworks** — 아직 first-class HTN 또는 AlphaEvolve를 제공하지 않습니다. Subagent 또는 background worker로 만드세요.

## 출시하기

`outputs/skill-hybrid-planner.md`는 LLM의 역할을 명시적으로 제한한 hybrid planner scaffold(HTN 또는 evolutionary)를 생성합니다.

## 연습 문제

1. HTN planner에 backtracking을 추가하세요. Operator의 postcondition이 runtime에 실패하면 rollback하고 다음 method를 시도하세요.
2. ChatHTN에 LLM-method cache를 추가하세요. LLM이 state pattern `P`에서 task `T`를 분해하면 결과를 저장하세요. 다음 call에서 method library를 먼저 다시 확인하세요.
3. Evolutionary search evaluator를 실제 test suite로 바꾸세요. 20개 test case를 통과하는 sort function을 evolve하고 convergence까지 generation 수를 보고하세요.
4. AlphaEvolve의 evaluator design note를 읽으세요. 관심 있는 domain(SQL query optimization, test-suite minimization, deployment YAML)의 evaluator를 설계하세요.
5. 결합해 보세요. HTN으로 compound task를 subtask로 분해한 다음, 각 subtask의 primitive operator에 evolutionary search를 사용하세요. 어디에서 빛나고 어디에서 over-engineering이 되나요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| HTN | "Hierarchical planner" | Operator, precondition, effect를 가진 task decomposition |
| Method | "Decomposition rule" | Compound task를 subtask로 나누는 방법 |
| Operator | "Primitive action" | Precondition과 effect가 있는 concrete step |
| ChatHTN | "LLM + HTN" | Method가 없을 때 LLM에 묻는 symbolic planner |
| AlphaEvolve | "Evolutionary code search" | Ensemble LLM이 code를 mutate하고 deterministic evaluator가 select함 |
| Fitness function | "Evaluator" | Output에 대한 deterministic, machine-checkable score |
| Online method learning | "Cached LLM decomposition" | Query cost를 줄이기 위해 LLM plan을 저장하고 generalize함 |

## 더 읽기

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814) — symbolic + LLM hybrid planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — LLM mutation을 쓰는 evolutionary code search
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — planner와 simple loop 중 무엇을 선택할지
