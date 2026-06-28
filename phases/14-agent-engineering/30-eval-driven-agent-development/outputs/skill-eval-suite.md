---
name: eval-suite
description: evaluator-optimizer loop와 CI gate를 갖춘 three-layer eval suite(static benchmarks, custom offline, online production)를 만든다.
version: 1.0.0
phase: 14
lesson: 30
tags: [evaluation, ci, regression, benchmarks, llm-judge]
---

agent product가 주어지면 CI에 연결된 three-layer eval suite를 만든다.

생성할 것:

1. **Static benchmark layer** — 관련 benchmark를 하나 이상 포함한다(code용 SWE-bench Verified, tool use용 BFCL V4, web용 WebArena, desktop용 OSWorld, generalist용 GAIA). 항상 +-audited score를 함께 report한다.
2. **Custom offline layer** — domain-specific dimension(factual, tone, scope, refusal quality)으로 score되는 LLM-judge rubric을 하나 이상 포함한다. agent 실행 후 actual state를 probe하는 execution-based case를 하나 이상 포함한다. gold path가 있는 trajectory-based case를 하나 이상 포함한다.
3. **Online eval layer** — session replay, guardrail-triggered alert, OTel GenAI span(Lesson 23)을 통한 step별 cost/latency tracking.
4. **Evaluator-optimizer runner** — round cap이 있는 propose / judge / refine로 agent를 감싼다.
5. **CI gate** — baseline 대비 >=5% regression에서 build를 fail한다. baseline을 시간에 따라 track한다.
6. **Case mapping** — Phase 14 lesson의 모든 guardrail과 모든 learned rule은 case를 하나 이상 가진다.

강한 거부 조건:

- baseline 없는 eval suite. reference 없이는 regression을 detect할 수 없다.
- factual task에서 external grounding 없는 LLM-judge. CRITIC pattern(Lesson 05)이 필요하다.
- pinned seed 또는 snapshot state 없는 flaky case. false alarm은 eval에 대한 team의 trust를 깎는다.

거부 규칙:

- user가 "just the happy path"를 원하면 거부한다. 모든 failure mode(Lesson 26)는 case를 가져야 한다.
- user가 "no CI gate"를 원하면 paying user가 있는 product에서는 거부한다. 그렇지 않으면 eval drift가 보이지 않는다.
- user가 "all LLM-judges"를 원하면 factual 및 compliance task에서는 거부한다. 그곳에는 execution-based 또는 programmatic evaluator가 필요하다.

출력: rubric, baseline, Phase 14 mapping table을 설명하는 `cases/benchmarks/`, `cases/custom/`, `cases/online/`, `runner.py`, `ci_gate.py`, `README.md`. 마지막은 substrate를 위한 Lesson 24(observability), Lesson 26(failure modes), 또는 Lesson 23(OTel)을 가리키는 "what to read next"로 끝낸다.
