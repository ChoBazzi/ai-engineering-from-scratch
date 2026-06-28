---
name: refine-loop
description: task, verifier availability, iteration budget이 주어졌을 때 evaluator-optimizer(Self-Refine / CRITIC) loop를 구성합니다.
version: 1.0.0
phase: 14
lesson: 05
tags: [self-refine, critic, evaluator-optimizer, guardrails, iteration]
---

task, iteration budget, available verifier(tool-grounded 또는 self-eval only)가 주어지면 evaluator-optimizer loop용 prompt와 stop policy를 emit하세요.

생성할 것:

1. Generator prompt. first output을 위한 deterministic producer. task, output format, constraint를 명시하세요.
2. Evaluator/verifier prompt. tool(search, code run, tests, calculator, type check)이 있으면 어떻게 호출하고 structured critique(JSON: pass/fail, violations[], suggested_fixes[])를 생성할지 명시하세요. self-eval만 가능하면 Self-Refine rubber-stamp risk를 명시적으로 flag하고, 구조적으로 다른 prompt style(예: adversarial "find at least one flaw")을 사용하세요.
3. Refiner prompt. prior outputs와 critiques(history)를 반드시 참조해야 합니다. "prior iteration에서 flag된 failure mode를 반복하지 말 것"이 mandatory임을 명시하세요.
4. Stop policy. conjunction: verifier passes OR (self-eval says fine AND iterations >= 2) OR iterations >= max_iterations. single-condition은 절대 쓰지 마세요.
5. Observability hooks. 전체 refine trajectory를 audit할 수 있도록 각 iteration을 Lesson 23에 따라 OpenTelemetry GenAI span(evaluate, optimize)으로 log하세요.

강한 거부:

- generator와 critic에 같은 prompt 사용. Rubber-stamp risk가 있습니다. model이 자기 자신에게 동의합니다.
- iteration cap 없음. infinite refine loop는 token을 태웁니다. 기본적으로 항상 4로 cap을 두세요.
- freeform prose feedback을 요청하는 verifier prompt. structured JSON만 사용하세요. pass/fail과 itemized violations가 필요합니다.
- refiner prompt에서 history 제거. 논문은 history 없이는 quality가 붕괴함을 보여 줍니다.

거부 규칙:

- task에 verifier가 없고 만들 방법도 없다면 CRITIC을 거부하고 Self-Refine이 가능한 약한 option임을 알리세요. rubber-stamp risk를 사용자에게 경고하세요.
- max_iterations >= 10이면 거부하고 task 재설계를 권하세요. 3-4 pass를 넘는 refine-to-convergence는 보통 generator prompt가 잘못됐다는 신호입니다.
- verifier가 destructive tool(shell, git write)을 호출하면 거부하고 sandbox boundary(Lesson 09)를 요구하세요.

출력: 모든 prompt, stop policy, tool list가 담긴 단일 configuration block과 deployment target에 따라 Lesson 16(OpenAI Agents SDK guardrails), Lesson 12(Anthropic evaluator-optimizer), Lesson 30(eval-driven agent development)을 가리키는 "what to read next" note.
