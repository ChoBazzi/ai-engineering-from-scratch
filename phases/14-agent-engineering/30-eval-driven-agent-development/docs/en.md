# Eval-Driven Agent Development

> Anthropic의 guidance: "start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed." evaluation은 마지막 step이 아니다. Phase 14의 다른 모든 선택을 구동하는 outer loop다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 minutes

## Learning Objectives

- 세 evaluation layer(static benchmarks, custom offline, online production)와 각각의 용도를 말한다.
- evaluator-optimizer tight loop를 설명한다.
- 2026 best practice를 설명한다: eval은 code 옆에 있고 CI에서 실행되며 PR을 gate한다.
- Phase 14의 모든 lesson을 그것이 생성하는 eval case와 연결한다.

## The Problem

agent는 demo를 통과한다. 하지만 demo가 예측할 수 없는 방식으로 production에서 실패한다. benchmark는 "이 model이 넓게 capable한가?"에는 답하지만 "이 agent가 내 product에 올바른 patch를 shipping하는가?"에는 답하지 않는다. 해답은 세 layer의 evaluation을 지속적으로 실행하고, 모든 guardrail과 learned rule을 eval case에 mapping하는 것이다.

## The Concept

### Three evaluation layers

1. **Static benchmarks** — code용 SWE-bench Verified(Lesson 19), browsing / desktop용 WebArena/OSWorld(Lesson 20), generalist용 GAIA(Lesson 19), tool use용 BFCL V4(Lesson 06). cross-model comparison과 regression gating에 사용한다. contamination은 실제다. SWE-bench+는 32.67% solution leakage를 발견했다. 항상 Verified / +-audited score를 report한다.

2. **Custom offline evals** — 자신의 product shape:
   - LLM-as-judge(Langfuse, Phoenix, Opik — Lesson 24).
   - execution-based(patch를 실행하고 test 확인).
   - trajectory-based(action sequence를 gold와 비교. OSWorld-Human은 top agent가 gold 대비 1.4-2.7x임을 보인다).

3. **Online evals** — production:
   - session replay(Langfuse).
   - guardrail-triggered alert(Lesson 16, 21).
   - step별 cost / latency tracking(Lesson 23 OTel spans).

### Evaluator-optimizer (Anthropic)

tight loop:

1. proposer가 output을 생성한다.
2. evaluator가 judge한다.
3. evaluator가 pass할 때까지 refine한다.

이는 Self-Refine(Lesson 05)을 generalize한 것이다. 신뢰성이 중요한 모든 agent flow를 evaluator-optimizer로 감쌀 수 있다.

### 2026 best practice

- eval은 code 옆에 둔다.
- 모든 PR에서 CI로 실행한다.
- eval score로 merge를 gate한다(예: "main 대비 regression > 5% 없음").
- 모든 guardrail은 eval case에 mapping된다.
- 모든 learned rule(Reflexion, pro-workflow learn-rule)은 failure case에 mapping된다.

### Tying Phase 14 together

Phase 14의 모든 lesson은 eval case를 생성한다.

| Lesson | Eval case it generates |
|--------|------------------------|
| 01 Agent Loop | budget-exhausted, infinite-loop guard |
| 02 ReWOO | tool 실패 시 planner가 올바르게 replan |
| 03 Reflexion | retry에서 learned reflection 적용 |
| 05 Self-Refine/CRITIC | judge가 refined output을 pass |
| 06 Tool Use | argument coercion이 작동하고 unknown tool을 reject |
| 07-10 Memory | retrieval citation이 source와 맞고 stale fact가 invalidate |
| 12 Workflow Patterns | 각 pattern이 올바른 output 생성 |
| 13 LangGraph | resume이 state를 정확히 재현 |
| 14 AutoGen Actors | DLQ가 crashed handler를 잡음 |
| 16 OpenAI Agents SDK | guardrail이 올바른 input에서 trip |
| 17 Claude Agent SDK | subagent result가 orchestrator로 반환 |
| 19-20 Benchmarks | SWE-bench Verified score, WebArena success rate, OSWorld efficiency |
| 21 Computer Use | step별 safety가 injected DOM을 잡음 |
| 23 OTel | span이 required attribute를 emit |
| 26 Failure Modes | detector가 known failure를 tag |
| 27 Prompt Injection | PVE가 poisoned retrieval을 refuse |
| 28 Orchestration | supervisor가 올바른 specialist로 route |
| 29 Runtime Shapes | DLQ가 N% failure를 처리 |

eval suite에 각각의 case가 있으면 Phase 14를 포괄한 것이다.

### Where eval-driven development fails

- **baseline 없음.** last-known-good 없는 eval은 읽을 수 없다. baseline을 저장한다.
- **grounding 없는 LLM-judge.** judge도 hallucinate한다. CRITIC pattern(Lesson 05) — judge를 external tool에 ground한다.
- **eval에 over-fitting.** eval에 최적화하면 production usefulness에서 벗어난다. case를 rotate한다.
- **flaky evals.** non-deterministic case는 false alarm을 만든다. seed를 고정하고 state를 snapshot한다.

## Build It

`code/main.py`는 stdlib eval harness다.

- category(benchmark, custom, online)가 있는 case registry.
- test 대상 scripted agent.
- evaluator-optimizer loop: pass 또는 max round까지 propose, judge, refine.
- CI gate: aggregate pass rate + baseline 대비 regression.

실행:

```bash
python3 code/main.py
```

출력: case별 pass/fail, regression flag, CI gate verdict.

## Use It

- agent code와 같은 repo에 eval case를 작성한다.
- 모든 PR에서 CI로 실행한다.
- regression이 있으면 build를 fail한다.
- 시간에 따른 pass rate를 추적한다.
- 모든 production failure를 새 case에 연결한다.

## Ship It

`outputs/skill-eval-suite.md`는 CI gate와 regression tracking을 갖춘 agent product용 three-layer eval suite를 만든다.

## Exercises

1. production failure 하나를 고른다. 이를 재현하는 eval case를 작성한다. 이제 agent가 통과하는가?
2. domain용 LLM-judge rubric을 세 dimension(factual, tone, scope)으로 만든다. 50개 session을 score한다.
3. eval suite를 CI에 연결한다. >=5% regression에서 build를 fail한다.
4. trajectory-efficiency metric을 추가한다. gold trajectory와 비교해 agent가 몇 step을 사용했는가?
5. Phase 14의 모든 lesson을 suite 안의 eval case에 mapping한다. 빠진 것이 있는가? 그것이 닫아야 할 gap이다.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Static benchmark | "Off-the-shelf eval" | SWE-bench, GAIA, AgentBench, WebArena, OSWorld |
| Custom offline eval | "Domain eval" | product shape에 대한 LLM-as-judge / exec / trajectory |
| Online eval | "Production eval" | session replay, guardrail alert, cost/latency tracking |
| Evaluator-optimizer | "Propose-judge-refine" | judge가 pass할 때까지 iterate |
| CI gate | "Merge blocker" | eval regression에서 build fail |
| Baseline | "Last-known-good" | regression을 detect하기 위한 reference score |
| Trajectory efficiency | "Steps over gold" | agent step count를 human expert minimum으로 나눈 값 |

## Further Reading

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — "start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — curated benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) — tool-use benchmark
- [Langfuse docs](https://langfuse.com/) — 실제 evals + session replay
