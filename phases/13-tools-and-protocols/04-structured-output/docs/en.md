# Structured Output - JSON Schema, Pydantic, Zod, Constrained Decoding

> "모델에게 JSON을 반환하라고 정중히 요청하기"는 frontier model에서도 5~15% 정도 실패합니다. structured output은 constrained decoding으로 그 간극을 닫습니다. 모델은 schema를 위반하는 token을 문자 그대로 출력할 수 없게 됩니다. OpenAI의 strict mode, Anthropic의 schema-typed tool use, Gemini의 `responseSchema`, Pydantic AI의 `output_type`, Zod의 `.parse`는 같은 아이디어의 다섯 가지 surface form입니다. 이 레슨은 학습자가 모든 production extraction pipeline에 사용할 schema validator와 strict-mode contract를 만듭니다.

**Type:** Build
**Languages:** Python (stdlib, JSON Schema 2020-12 subset)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 학습 목표

- 올바른 constraint(enum, min/max, required, pattern)를 사용해 extraction target에 대한 JSON Schema 2020-12를 작성합니다.
- strict mode와 constrained decoding이 "generation 이후 validate"와 어떻게 다른 보장을 주는지 설명합니다.
- 세 failure mode(parse error, schema violation, model refusal)를 구분합니다.
- typed repair와 typed refusal handling이 있는 extraction pipeline을 출하합니다.

## 문제

purchase-order email을 읽는 agent는 free text를 `{customer, line_items, total_usd}`로 바꿔야 합니다. 세 가지 접근이 있습니다.

**접근 1: JSON을 prompt로 요청.** "customer, line_items, total_usd field가 있는 JSON으로 답하세요." frontier model에서 85~95%는 동작합니다. 여섯 가지 방식으로 실패합니다. missing brace, trailing comma, wrong type, hallucinated field, token limit에서 truncation, "Here is your JSON:" 같은 leaked prose입니다.

**접근 2: generation 이후 validate.** 자유롭게 generate하고, parse하고, schema로 validate하고, 실패하면 retry합니다. 신뢰할 수 있지만 비쌉니다. retry마다 비용을 내고, truncation bug는 발생할 때마다 한 turn을 추가로 소모합니다.

**접근 3: constrained decoding.** provider가 decode time에 schema를 강제합니다. invalid token은 sampling distribution에서 mask됩니다. output은 parse된다는 보장과 validate된다는 보장을 받습니다. 실패는 하나의 mode로 수렴합니다. 모델이 입력이 schema에 맞지 않는다고 판단하는 refusal입니다.

2026년의 모든 frontier provider는 접근 3의 어떤 형태를 제공합니다.

- **OpenAI.** `response_format: {type: "json_schema", strict: true}`와, 모델이 거절할 경우 response의 `refusal`.
- **Anthropic.** `tool_use` input에 대한 schema enforcement. `stop_reason: "refusal"` 같은 것은 없지만 tool call 없이 `end_turn`이 나오면 signal입니다.
- **Gemini.** request level의 `responseSchema`. 2026년 Gemini는 selected type에 token-level grammar constraint를 제공합니다.
- **Pydantic AI.** `output_type=InvoiceModel`은 `InvoiceModel`로 typed된 structured `RunResult`를 출력합니다.
- **Zod (TypeScript).** provider output을 Zod schema로 validate하는 runtime parser입니다. OpenAI의 `beta.chat.completions.parse`와 함께 쓰입니다.

공통점은 하나입니다. schema를 한 번 선언하고 end to end로 강제합니다.

## 개념

### JSON Schema 2020-12 - lingua franca

모든 provider는 JSON Schema 2020-12를 받습니다. 가장 많이 쓰는 construct는 다음과 같습니다.

- `type`: `object`, `array`, `string`, `number`, `integer`, `boolean`, `null` 중 하나.
- `properties`: field name에서 subschema로 가는 map.
- `required`: 반드시 나타나야 하는 field name list.
- `enum`: 허용 값의 closed set.
- `minimum` / `maximum`(number), `minLength` / `maxLength` / `pattern`(string).
- `items`: 모든 array element에 적용되는 subschema.
- `additionalProperties`: `false`는 extra field를 금지합니다(mode에 따라 기본값이 다름).

OpenAI strict mode는 세 가지 requirement를 추가합니다. 모든 property는 `required`에 listed되어야 하고, 모든 곳에 `additionalProperties: false`가 있어야 하며, unresolved `$ref`가 없어야 합니다. 이를 어기면 API가 request time에 400을 반환합니다.

### Pydantic, Python binding

Pydantic v2는 `model_json_schema()`를 통해 dataclass-shaped model에서 JSON Schema를 생성합니다. Pydantic AI는 이를 감싸서 다음처럼 작성할 수 있게 합니다.

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

그러면 agent framework가 edge에서 schema를 OpenAI strict mode, Anthropic `input_schema`, Gemini `responseSchema`로 번역합니다. 모델 output은 typed `Invoice` instance로 돌아옵니다. validation error는 typed error path가 있는 `ValidationError`를 발생시킵니다.

### Zod, TypeScript binding

Zod(`z.object({customer: z.string(), ...})`)는 TS equivalent입니다. OpenAI의 Node SDK는 API의 JSON Schema payload로 번역되는 `zodResponseFormat(Invoice)`를 제공합니다.

### refusal

strict mode는 모델에게 답변을 강제할 수 없습니다. input이 schema에 맞을 수 없다면("email이 invoice가 아니라 poem이었다"), 모델은 이유가 담긴 `refusal` field를 출력합니다. 코드는 이를 failure가 아니라 first-class outcome으로 처리해야 합니다. refusal은 safety signal로도 유용합니다. protected-content email에서 credit card number를 추출하라는 요청을 받은 모델은 safety reason이 붙은 refusal을 반환합니다.

### open 환경의 constrained decoding

open-weights implementation은 세 가지 기법을 사용합니다.

1. **Grammar-based decoding**(`outlines`, `guidance`, `lm-format-enforcer`): schema에서 deterministic finite automaton을 만들고, 매 step마다 FSM을 위반할 token의 logit을 mask합니다.
2. **JSON parser를 사용한 logit masking**: streaming JSON parser를 모델과 lockstep으로 실행하고, 매 step마다 valid-next-token set을 계산합니다.
3. **verifier가 있는 speculative decoding**: 저렴한 draft model이 token을 제안하고, verifier가 schema를 강제합니다.

commercial provider는 이 중 하나를 behind the scenes에서 선택합니다. 2026년 state of the art는 짧은 structured output에서는 plain generation보다 빠르고, 긴 output에서는 거의 같은 속도입니다.

### 세 failure mode

1. **Parse error.** output이 valid JSON이 아닙니다. strict mode에서는 발생할 수 없습니다. non-strict provider에서는 여전히 발생할 수 있습니다.
2. **Schema violation.** output은 parse되지만 schema를 위반합니다. strict mode에서는 발생할 수 없습니다. 그 밖에서는 흔합니다.
3. **Refusal.** 모델이 거절합니다. typed outcome으로 처리해야 합니다.

### retry strategy

strict mode 바깥(Anthropic tool use, non-strict OpenAI, older Gemini)에 있을 때 recovery pattern은 다음과 같습니다.

```text
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

보통 retry 한 번이면 충분합니다. 세 번의 retry는 weak-model flake를 잡습니다. 세 번을 넘으면 bad schema의 신호입니다. 모델이 어떤 input에 대해 schema를 만족할 수 없으며 prompt나 schema를 고쳐야 합니다.

### small-model support

constrained decoding은 small model에서도 동작합니다. grammar enforcement가 있는 3B-parameter open model은 structured task에서 raw prompting을 사용하는 70B-parameter model보다 성능이 좋습니다. 이것이 structured output이 production에서 중요한 주된 이유입니다. reliability를 model size와 분리해 줍니다.

## 사용하기

`code/main.py`는 stdlib로 작성한 최소 JSON Schema 2020-12 validator(types, required, enum, min/max, pattern, items, additionalProperties)를 제공합니다. `Invoice` schema를 감싸고 fake LLM output을 validator에 통과시켜 parse error, schema violation, refusal path를 보여 줍니다. production에서는 fake output을 provider의 실제 response로 바꾸면 됩니다.

살펴볼 것:

- validator는 path와 message가 있는 typed `[ValidationError]` list를 반환합니다. 이것이 retry prompt에 노출하고 싶은 shape입니다.
- refusal branch는 retry하지 않습니다. log를 남기고 typed refusal을 반환합니다. Phase 14 · 09는 refusal을 safety signal로 사용합니다.
- `additionalProperties: false` check가 adversarial test input에서 발동하여, strict mode가 hallucinated field의 문을 닫는 이유를 보여 줍니다.

## 산출물

이 레슨은 `outputs/skill-structured-output-designer.md`를 만듭니다. free-text extraction target(invoices, support tickets, resumes 등)이 주어지면, 이 skill은 strict-mode-compatible JSON Schema 2020-12와 이를 mirror하는 Pydantic model을 만들고, typed refusal과 retry handling stub을 포함합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. `total_usd`가 negative number인 네 번째 test case를 추가하세요. validator가 `minimum` constraint path로 reject하는지 확인하세요.

2. discriminator가 있는 `oneOf`를 지원하도록 validator를 확장하세요. 흔한 case는 `line_item`이 `kind`로 tag된 product 또는 service인 경우입니다. strict mode에는 여기서 미묘한 규칙이 있습니다. OpenAI의 structured outputs guide를 확인하세요.

3. 같은 Invoice schema를 Pydantic BaseModel로 작성하고 `model_json_schema()` output을 hand-rolled schema와 비교하세요. Pydantic이 기본으로 설정하지만 hand-rolled version에는 빠져 있는 field 하나를 찾으세요.

4. refusal rate를 측정하세요. extract할 수 없어야 하는 input 10개(song lyric, math proof, blank email)를 구성하고 strict mode가 있는 real provider에 실행하세요. refusal과 hallucinated output을 세세요. 이것이 refusal-aware retry의 ground truth입니다.

5. OpenAI의 structured outputs guide를 처음부터 끝까지 읽으세요. plain JSON Schema는 허용하지만 strict mode에서 명시적으로 금지하는 construct 하나를 찾으세요. 그런 다음 그 금지 construct를 비필수적으로 사용하는 schema를 설계하고 strict-compatible하게 refactor하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| JSON Schema 2020-12 | "The schema spec" | 모든 최신 provider가 사용하는 IETF-draft schema dialect |
| Strict mode | "Guaranteed schema" | constrained decoding으로 schema를 강제하는 OpenAI flag |
| Constrained decoding | "Logit masking" | invalid next-token을 mask하는 decode-time enforcement |
| Refusal | "Model declines" | input이 schema에 맞을 수 없을 때의 typed outcome |
| Parse error | "Invalid JSON" | output이 JSON으로 parse되지 않음. strict에서는 불가능 |
| Schema violation | "Wrong shape" | parse는 되었지만 type / required / enum / range를 위반함 |
| `additionalProperties: false` | "No extras allowed" | unknown field를 금지함. OpenAI strict에서 필요 |
| Pydantic BaseModel | "Typed output" | JSON Schema를 emit하고 validate하는 Python class |
| Zod schema | "TypeScript output type" | provider output validation을 위한 TS runtime schema |
| Grammar enforcement | "Open-weights constrained decode" | outlines / guidance 같은 FSM-based logit masking |

## 더 읽을거리

- [OpenAI - Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) - strict mode, refusal, schema requirement
- [OpenAI - Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) - decoding guarantee를 설명하는 2024년 8월 launch post
- [Pydantic AI - Output](https://ai.pydantic.dev/output/) - 각 provider로 serialize되는 typed `output_type` binding
- [JSON Schema - 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) - canonical spec
- [Microsoft - Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) - enterprise deployment note와 strict-mode caveat
