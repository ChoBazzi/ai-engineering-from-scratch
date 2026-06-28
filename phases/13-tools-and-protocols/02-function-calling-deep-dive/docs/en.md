# Function Calling 심층 분석 - OpenAI, Anthropic, Gemini

> 세 frontier provider는 2024년에 같은 tool-call loop로 수렴한 뒤, 나머지 모든 것에서 갈라졌습니다. OpenAI는 `tools`와 `tool_calls`를 사용합니다. Anthropic은 `tool_use`와 `tool_result` block을 사용합니다. Gemini는 `functionDeclarations`와 unique-id correlation을 사용합니다. 이 레슨은 세 가지를 나란히 비교해, 한 provider에서 출시한 코드가 다른 provider로 port될 때 깨지지 않게 합니다.

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 minutes

## 학습 목표

- OpenAI, Anthropic, Gemini function-calling payload의 세 가지 shape 차이(declaration, call, result)를 말합니다.
- 하나의 tool declaration을 세 provider format으로 모두 번역하고, strict-mode constraint가 어디서 달라질지 예측합니다.
- 각 provider에서 `tool_choice`를 사용해 tool call을 force, forbid 또는 auto-pick합니다.
- provider별 hard limit(tool count, schema depth, argument length)과 limit 위반 시 각 provider가 내는 error signature를 압니다.

## 문제

function-calling request의 shape는 provider마다 다릅니다. 2026년 production stack에서 가져온 세 가지 구체적인 예입니다.

**OpenAI Chat Completions / Responses API.** `tools: [{type: "function", function: {name, description, parameters, strict}}]`를 전달합니다. 모델의 response에는 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`가 들어 있으며, 여기서 `arguments`는 직접 parse해야 하는 JSON string입니다. strict mode(`strict: true`)는 constrained decoding으로 schema compliance를 강제합니다.

**Anthropic Messages API.** `tools: [{name, description, input_schema}]`를 전달합니다. response는 `content: [{type: "text"}, {type: "tool_use", id, name, input}]`로 돌아옵니다. `input`은 이미 parse되어 있습니다(string이 아니라 object). 결과를 보낼 때는 `{type: "tool_result", tool_use_id, content}` block이 들어 있는 새 `user` message로 답합니다.

**Google Gemini API.** `tools: [{functionDeclarations: [{name, description, parameters}]}]`를 전달합니다(`functionDeclarations` 아래에 중첩). response는 `candidates[0].content.parts: [{functionCall: {name, args, id}}]`로 도착하며, `id`는 Gemini 3 이상에서 parallel-call correlation을 위해 unique합니다. 결과는 `{functionResponse: {name, id, response}}`로 답합니다.

같은 loop입니다. 하지만 field name, nesting, string-vs-object convention, correlation mechanism이 다릅니다. OpenAI에서 weather agent를 작성한 팀은 plumbing만으로 Anthropic port에 이틀, Gemini port에 하루를 더 쓰게 됩니다.

이 레슨은 세 format을 하나의 canonical tool declaration으로 통합하고 edge에서 route하는 translator를 만듭니다. Phase 13 · 17은 같은 패턴을 LLM gateway로 일반화합니다.

## 개념

### 공통 구조

모든 provider에는 다섯 가지가 필요합니다.

1. **Tool list.** tool별 name, description, input schema.
2. **Tool choice.** 특정 tool 강제, tool 금지, 또는 model에 결정 위임.
3. **Call emission.** tool과 arguments를 명명하는 structured output.
4. **Call id.** response를 올바른 call과 연결합니다(parallel에서 중요).
5. **Result injection.** result를 call에 다시 묶는 message 또는 block.

### field별 shape 차이

| 측면 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| 선언 envelope | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema field | `parameters` | `input_schema` | `parameters` |
| Response container | assistant message의 `tool_calls[]` | type이 `tool_use`인 `content[]` | type이 `functionCall`인 `parts[]` |
| Arguments type | stringified JSON | parsed object | parsed object |
| Id 형식 | `call_...`(OpenAI가 생성) | `toolu_...`(Anthropic) | UUID(Gemini 3+) |
| Result block | role `tool`, `tool_call_id` | `tool_result`, `tool_use_id`가 있는 `user` | matching `id`가 있는 `functionResponse` |
| 특정 tool 강제 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| Tool 금지 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | schema-is-schema(항상 강제) | request level의 `responseSchema` |

### 실제로 부딪히게 될 limit

- **OpenAI.** request당 tool 128개. schema depth 5. argument string <= 8192 bytes. strict mode는 `$ref` 금지, overlap이 있는 `oneOf`/`anyOf`/`allOf` 금지, 모든 property가 `required`에 있어야 함을 요구합니다.
- **Anthropic.** request당 tool 64개. schema depth는 사실상 unbounded지만 practical limit은 10입니다. strict-mode flag는 없습니다. schema는 contract이고 모델은 대체로 따릅니다.
- **Gemini.** request당 function 64개. schema type은 OpenAPI 3.0 subset(JSON Schema 2020-12와 약간 다름)입니다. Gemini 3부터 parallel call unique-id가 있습니다.

### `tool_choice` 동작

모두가 지원하지만 이름이 다른 세 mode가 있습니다.

- **Auto.** 모델이 tool 또는 text를 선택합니다. 기본값입니다.
- **Required / Any.** 모델은 적어도 하나의 tool을 호출해야 합니다.
- **None.** 모델은 tool을 호출하면 안 됩니다.

각 provider에 고유한 mode도 하나씩 있습니다.

- **OpenAI.** 이름으로 특정 tool을 force합니다.
- **Anthropic.** 이름으로 특정 tool을 force합니다. `disable_parallel_tool_use` flag가 single과 multi를 분리합니다.
- **Gemini.** `mode: "VALIDATED"`는 모델 의도와 관계없이 모든 response를 schema validator로 route합니다.

### parallel call

OpenAI의 `parallel_tool_calls: true`(기본값)는 한 assistant message에 여러 call을 출력합니다. 당신은 모두 실행하고, `tool_call_id`마다 하나의 entry가 있는 batched tool-role message로 답합니다. Anthropic은 역사적으로 single-call이었지만, `disable_parallel_tool_use: false`(Claude 3.5 기준 기본값)가 multi를 활성화합니다. Gemini 2는 parallel call을 허용했지만 stable id를 주지 않았습니다. Gemini 3는 UUID를 추가해 out-of-order response가 깔끔하게 correlate되도록 합니다.

### streaming

세 provider 모두 streamed tool call을 지원합니다. wire format은 다릅니다.

- **OpenAI.** `tool_calls[i].function.arguments`의 delta chunk가 incrementally 도착합니다. `finish_reason: "tool_calls"`까지 누적합니다.
- **Anthropic.** block-start / block-delta / block-stop event입니다. `input_json_delta` chunk가 partial arguments를 운반합니다.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3의 새 기능)가 `functionCallId`가 있는 chunk를 내보내 여러 parallel call이 interleave될 수 있게 합니다.

Phase 13 · 03은 parallel + streaming reassembly를 깊게 다룹니다. 이 레슨은 declaration과 single-call shape에 초점을 둡니다.

### error와 repair

invalid-argument error도 서로 다르게 보입니다.

- **OpenAI (non-strict).** 모델이 `arguments: "{bad json}"`를 반환하고, 당신의 JSON parse가 실패합니다. error message를 주입하고 다시 호출합니다.
- **OpenAI (strict).** validation은 decoding 중에 일어납니다. invalid JSON은 불가능하지만 `refusal`이 나타날 수 있습니다.
- **Anthropic.** `input`에 unexpected field가 들어 있을 수 있습니다. schema는 advisory입니다. server-side로 validate하세요.
- **Gemini.** OpenAPI 3.0 quirk: object field의 `enum`이 조용히 무시됩니다. 직접 validate하세요.

### translator pattern

코드의 canonical tool declaration은 이렇게 생겼습니다(형태는 당신이 정합니다).

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

세 개의 작은 function이 이를 세 provider shape로 번역합니다. `code/main.py`의 harness가 정확히 이 일을 한 뒤, fake tool call을 각 provider의 response shape로 round-trip합니다. network는 필요 없습니다. 이 레슨은 HTTP가 아니라 shape를 가르칩니다.

production team은 이 translator를 `AbstractToolset`(Pydantic AI), `UniversalToolNode`(LangGraph), `BaseTool`(LlamaIndex)로 감쌉니다. Phase 13 · 17은 세 provider 앞에서 OpenAI-shaped API를 노출하는 gateway를 제공합니다.

## 사용하기

`code/main.py`는 하나의 canonical `Tool` dataclass와 OpenAI, Anthropic, Gemini declaration JSON을 출력하는 세 translator를 정의합니다. 그런 다음 손으로 만든 각 provider response shape를 같은 canonical call object로 parse하여, 겉모습 아래의 의미론이 동일함을 보여 줍니다. 실행하고 세 declaration을 나란히 diff하세요.

살펴볼 것:

- 세 declaration block은 envelope와 field name만 다릅니다.
- 세 response block은 call이 있는 위치(top-level `tool_calls`, `content[]` block, `parts[]` entry)가 다릅니다.
- 하나의 `canonical_call()` function이 세 response shape 모두에서 `{id, name, args}`를 추출합니다.

## 산출물

이 레슨은 `outputs/skill-provider-portability-audit.md`를 만듭니다. 한 provider에 대한 function-calling integration이 주어지면, 이 skill은 portability audit을 만듭니다. 어떤 provider limit에 의존하는지, 어떤 field를 rename해야 하는지, 다른 provider로 port할 때 무엇이 깨지는지를 알려 줍니다.

## 연습 문제

1. `code/main.py`를 실행하고 세 provider declaration JSON이 모두 같은 underlying `Tool` object를 serialize하는지 확인하세요. canonical tool을 수정해 enum parameter를 추가하고, Gemini translator만 OpenAPI quirk를 처리해야 하는지 확인하세요.

2. 각 provider에 대해 `list_tools` 또는 discovery call 이후 모델이 반환하는 tool list를 추출하는 `ListToolsResponse` parser를 추가하세요. OpenAI에는 native로 이것이 없습니다. 이 asymmetry를 기록하세요.

3. `tool_choice` conversion을 구현하세요. canonical `ToolChoice(mode="force", tool_name="x")`를 세 provider shape로 모두 map하세요. 그런 다음 `mode="any"`와 `mode="none"`을 map하세요. 레슨의 diff table을 확인하세요.

4. 세 provider 중 하나를 골라 function-calling guide를 끝까지 읽으세요. 그 schema spec에서 다른 두 provider가 지원하지 않는 field 하나를 찾으세요. 후보: OpenAI `strict`, Anthropic `disable_parallel_tool_use`, Gemini `function_calling_config.allowed_function_names`.

5. test vector를 작성하세요. 선언된 schema를 위반하는 arguments를 가진 tool call입니다. 각 provider의 validator를 통해 실행하고(Lesson 01의 stdlib validator를 proxy로 써도 됩니다), 어떤 error가 발생하는지 기록하세요. strictness를 기준으로 production에서 어떤 provider를 쓸지 문서화하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Function calling | "Tool use" | structured tool-call emission을 위한 provider-level API |
| Tool declaration | "Tool spec" | Name + description + JSON Schema input payload |
| `tool_choice` | "Force / forbid" | Auto / required / none / specific-name mode |
| Strict mode | "Schema enforcement" | schema와 맞도록 decoding을 constrain하는 OpenAI flag |
| `tool_use` block | "Anthropic의 call shape" | id, name, input이 있는 inline content block |
| `functionCall` part | "Gemini의 call shape" | name, args, id를 담은 `parts[]` entry |
| Arguments-as-string | "Stringified JSON" | OpenAI는 args를 object가 아니라 JSON string으로 반환함 |
| Parallel tool calls | "한 turn의 fan-out" | 하나의 assistant message 안에 여러 tool call이 있음 |
| Refusal | "Model declines" | call 대신 strict-mode-only refusal block |
| OpenAPI 3.0 subset | "Gemini schema quirk" | Gemini가 사용하는, 작은 차이가 있는 JSON-Schema-like dialect |

## 더 읽을거리

- [OpenAI - Function calling guide](https://platform.openai.com/docs/guides/function-calling) - strict mode와 parallel call을 포함한 canonical reference
- [Anthropic - Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) - `tool_use`와 `tool_result` block semantics
- [Google - Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) - parallel call, unique id, OpenAPI subset
- [Vertex AI - Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) - Gemini의 enterprise surface
- [OpenAI - Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) - strict-mode schema enforcement detail
