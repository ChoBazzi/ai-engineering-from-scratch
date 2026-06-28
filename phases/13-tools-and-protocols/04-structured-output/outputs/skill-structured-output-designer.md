---
name: structured-output-designer
description: free-text extraction target을 위해 strict-mode-compatible JSON Schema와 Pydantic model을 설계하고 typed refusal 및 retry handling stub을 포함합니다.
version: 1.0.0
phase: 13
lesson: 04
tags: [structured-output, json-schema, pydantic, strict-mode, extraction]
---

free-text extraction target(invoices, resumes, support tickets, research summaries)이 주어지면 production-ready extraction contract를 작성하세요. JSON Schema 2020-12, Pydantic model, refusal handler, retry policy를 포함합니다.

작성할 것:

1. JSON Schema 2020-12. 모든 property에 type이 있어야 합니다. `required`는 모든 property를 나열합니다. 모든 object에 `additionalProperties: false`가 있어야 합니다. closed value set에는 enum을 사용합니다. `$ref` 없음. ambiguous `oneOf` / `anyOf` 없음. OpenAI strict-mode requirement에 대해 validate되어야 합니다.
2. Pydantic v2 BaseModel. Python type으로 schema를 mirror합니다. `model_json_schema()`는 (1)과 equivalent한 schema를 만들어야 합니다.
3. Refusal handler. typed `Refusal(reason: str, category: str)` outcome. category를 나열하세요: `safety`, `input_mismatch`, `insufficient_info`.
4. Retry policy. 세 retry shape: (a) validation error를 주입하고 한 번 retry(strict mode 밖), (b) refusal을 final로 accept(strict mode), (c) 반복 refusal에서 stronger model로 escalate.
5. Test vectors. happy path, adversarial fields, partial input, refusal-triggering case를 포함하는 input 10개. 각각 expected outcome을 포함합니다.

강한 거부 조건:
- untyped field가 있는 모든 schema. strict mode와 validator 둘 다 실패합니다.
- `additionalProperties: false`가 빠진 모든 schema. hallucination이 새어 들어옵니다.
- discriminator field 없이 `oneOf`를 사용하는 모든 schema. ambiguous decoding입니다.
- JSON Schema round-trip을 확인하지 않은 모든 Pydantic model.

거부 규칙:
- target domain에 personally identifying data가 포함되어 있는데 documented purpose가 없으면 refuse하고 lawful-basis argument를 위해 Phase 18(ethics)로 route하세요.
- JSON Schema 2020-12로 표현할 수 없는 schema(예: recursive arbitrary graphs)를 요청받으면 refuse하고 표현 가능한 가장 가까운 relaxation을 제안하세요.
- extraction target이 "extract structured data from anything"이면 refuse하고 specific domain을 요청하세요.

출력: schema JSON, Pydantic class, refusal 및 retry policy, 10개 test vector가 포함된 one-page contract. 처음 target할 provider와 그 이유에 대한 note로 끝내세요.
