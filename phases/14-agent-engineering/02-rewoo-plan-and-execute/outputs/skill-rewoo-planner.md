---
name: rewoo-planner
description: user request와 tool catalog에서 검증된 ReWOO plan DAG를 생성합니다.
version: 1.0.0
phase: 14
lesson: 02
tags: [rewoo, plan-and-execute, planning, dag, distillation]
---

user request와 tool catalog(name, input schema, description)가 주어지면 tool call과 evidence reference(`#E1`, `#E2`, ...)가 있는 step DAG인 ReWOO plan을 생성하세요. executor에 넘기기 전에 plan을 검증하세요.

생성할 것:

1. plan DAG. 각 node는 id(`E1`, `E2`, ...), tool name, argument dict(string은 `#E<k>` reference를 포함할 수 있음), optional `parallel_group` label을 가집니다.
2. validation output. topological sort를 통한 acyclicity check, reference resolution check(모든 `#E<k>`에 preceding producer가 있음), tool existence check(모든 tool name이 catalog에 있음), arg schema check(각 argument가 tool의 input schema와 match).
3. parallelism hint. 각 topological level마다 concurrent하게 실행할 수 있는 node를 나열하세요.
4. planner/solver split recommendation. plan이 3 step 미만이면 ReAct를 권하세요. plan에 unbounded loop requirement(step마다 replanning)가 있으면 replanner가 있는 Plan-and-Execute를 권하세요. plan이 30 step을 넘거나 web/mobile을 target하면 synthetic plan data가 있는 Plan-and-Act를 권하세요.

강한 거부:

- cycle이 있는 plan. ReWOO는 DAG를 가정합니다. cycle은 ReAct 또는 LATS의 concern입니다.
- topological order에서 아직 존재하지 않는 `#E<k>`를 reference하는 plan. 실패한 edge를 구체적으로 emit하세요.
- catalog에 없는 tool을 호출하는 plan. plan을 성립시키려고 tool을 invent하지 마세요.
- reference의 argument type이 tool schema와 match하지 않는 plan(예: `#E1`은 string으로 substitute되지만 tool은 int를 기대).

거부 규칙:

- task가 open-ended exploration(필요한 tool도 step도 unknown)이면 거부하고 ReAct 또는 LATS(Lesson 04)를 권하세요.
- tool catalog에 gating approval tool 없는 destructive tool이 있으면 거부하고 Lesson 09(permissions, sandboxing)를 가리키세요.

출력: structured plan(JSON 또는 YAML), validation report, parallelism map, 그리고 executor(ReWOO Worker), replanner(Plan-and-Execute), 더 큰 trajectory-sampling loop(Plan-and-Act) 중 하나를 가리키는 follow-up action.

마지막에는 task class를 전에 시도한 적이 있으면 Lesson 03(Reflexion), plan이 search의 이점을 받을 수 있으면 Lesson 04(LATS)를 가리키는 "what to read next" note를 붙이세요.
