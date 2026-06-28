---
name: hybrid-planner
description: Provably-sound plan을 위한 ChatHTN, machine-checkable evaluator를 가진 code search를 위한 AlphaEvolve를 구축하고 문제에 맞는 쪽을 선택합니다.
version: 1.0.0
phase: 14
lesson: 11
tags: [planning, htn, chathtn, alphaevolve, evolutionary-search]
---

Problem class(policy-bound workflow vs code optimization vs open-ended task)가 주어지면 planner를 선택하고 correct scaffold를 만드세요.

Decision:

1. 문제에 hard precondition / policy / scheduling constraint가 있나요? -> HTN(ChatHTN).
2. 문제에 deterministic, machine-checkable fitness function이 있나요? -> Evolutionary(AlphaEvolve).
3. 둘 다 아니면? -> 대신 ReAct(Lesson 01) 또는 ReWOO(Lesson 02)를 사용하세요.

HTN의 경우 생성할 것:

1. `preconditions`, `effects_add`, `effects_remove`를 가진 `Operator` type.
2. `task`, `preconditions`, `subtasks`를 가진 `Method` type.
3. Method를 먼저 시도하고, LLM decomposition으로 fallback하며, 성공한 LLM decomposition을 cache하는 planner.
4. Unknown operator 또는 method를 참조하는 LLM decomposition을 reject하는 validation step.

Evolutionary의 경우 생성할 것:

1. Candidate program의 seed population.
2. Scalar fitness를 반환하는 deterministic evaluator.
3. Mutation operator(LLM-driven 또는 rule-based).
4. Early stopping을 갖춘 selection loop(keep top-k, mutate, repeat).

Hard rejects:

- LLM output을 operator-schema validation 없이 직접 적용하는 ChatHTN. Soundness claim이 실패합니다.
- Evaluator가 LLM judge를 호출하는 AlphaEvolve. Fitness는 deterministic해야 합니다. LLM judge는 loop가 회복할 수 없는 stochastic noise를 도입합니다.
- Open-ended task("blog post 쓰기")에 두 pattern 중 하나를 쓰는 것. Evaluator도 precondition도 없다면 ReAct를 사용하세요.

Refusal rules:

- Domain에 clear operator schema가 없다면 ChatHTN을 거절하세요. ReWOO 또는 plain ReAct를 제안하세요.
- Domain에 machine-checkable fitness가 없다면 AlphaEvolve를 거절하세요. Self-Refine(Lesson 05)을 제안하세요.
- 사용자가 "planner + LLM makes final call"을 원하면 거절하세요. Symbolic correctness와 LLM exploration의 분리는 load-bearing입니다.

Output: `operators.py`, `methods.py`, `planner.py`(HTN) 또는 `evaluator.py`, `mutator.py`, `loop.py`(evolutionary), 그리고 decision rationale을 담은 `README.md`. 마지막에는 debate-style verification이 맞으면 Lesson 25를, task가 사실 ReWOO-shaped라면 Lesson 02를 가리키는 "what to read next"로 끝내세요.
