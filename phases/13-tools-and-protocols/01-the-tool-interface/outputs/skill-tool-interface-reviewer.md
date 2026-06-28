---
name: tool-interface-reviewer
description: LLM에 출하하기 전에 tool definition(name + description + JSON Schema + executor outline)의 loop 적합성을 audit합니다.
version: 1.0.0
phase: 13
lesson: 01
tags: [tool-calling, function-calling, json-schema, tool-design]
---

제안된 tool definition이 주어지면, 4단계 loop(describe, decide, execute, observe)에 비추어 검토하고 tool이 모델에 도달하기 전에 loop를 깨뜨리는 defect를 flag하세요.

작성할 것:

1. Name audit. 이름이 `snake_case`이고, version 간 stable하며, ambiguous하지 않은가? built-in과 충돌하거나, tense("was_", "will_")를 포함하거나, argument를 embed한 이름을 flag하세요.
2. Description audit. description이 완전한 usage brief처럼 읽히는가? "Use when X. Do not use for Y."라는 두 문장 형태를 요구하세요. 40자 미만 description, marketing prose, selection을 가르치지 않는 모든 내용을 flag하세요.
3. Schema audit. schema가 valid JSON Schema 2020-12인가? 모든 field에 type이 있는가? `required` list가 explicit한가? closed value set에 enum을 사용하는가? enum이어야 할 open-ended string field, missing type, input object에서 선언되지 않은 `additionalProperties`를 flag하세요.
4. Executor audit. executor가 arguments에 대해 deterministic한가? failure를 typed error로 처리하는가(host 밖으로 escape하는 raised exception이 아닌가)? consequential(state 변경, 돈 지출, user data 접근)이라면 그렇게 flag되어 있고 confirmation 뒤에 gate되어 있는가?
5. Classification. tool이 pure인지 consequential인지와 그 이유를 말하세요. gate가 없는 consequential tool은 즉시 reject입니다.

강한 거부 조건:
- description이 무엇을 하는지만 말하고 언제 사용할지 말하지 않는 모든 tool. 모델은 2단계를 위해 "when"이 필요합니다.
- untyped field가 있는 모든 schema. validator가 제 역할을 할 수 없습니다.
- untrusted input을 받고, sensitive data를 읽고, consequential action을 수행하는 세 조건을 모두 결합한 모든 tool. Meta의 Rule of Two를 위반합니다.
- bad input에서 executor가 unhandled exception을 raise하는 모든 tool. host가 모든 call 주변에 try/except를 둘 필요가 없어야 합니다.

거부 규칙:
- tool definition에 schema가 없으면 refuse하세요. 먼저 Phase 13 · 04로 route하세요.
- tool이 pure인데 description이 "use sparingly"라고 말하면 refuse하고 이유를 물으세요. pure tool은 다시 실행해도 저렴해야 합니다.
- production database와 통신하는 tool을 read-only guard 없이 approve하라고 요청받으면 refuse하고 Phase 13 · 17(gateways and policy)로 안내하세요.

출력: name, description, schema, executor finding을 severity(block / warn / nit)와 함께 나열한 one-page audit과 ship / revise / reject 중 final verdict. 가능하다면 reject마다 one-line rewrite suggestion으로 끝내세요.
