# Tool Interface - 에이전트에 구조화된 I/O가 필요한 이유

> 언어 모델은 토큰을 생성합니다. 프로그램은 행동을 수행합니다. 이 둘 사이의 간극이 tool interface입니다. 모델이 어떤 행동을 요청하고 host가 그것을 실행할 수 있게 해 주는 계약입니다. 2026년의 모든 스택, 즉 OpenAI, Anthropic, Gemini의 function calling, MCP의 `tools/call`, A2A의 task part는 같은 4단계 loop를 서로 다르게 인코딩한 것입니다. 이 레슨은 그 loop에 이름을 붙이고, 이를 실행하는 데 필요한 최소 장치를 보여 줍니다.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 minutes

## 학습 목표

- 텍스트만 생성할 수 있는 LLM이 왜 스스로 실제 세계에 대해 행동할 수 없는지 설명합니다.
- 4단계 tool-call loop(describe -> decide -> execute -> observe)를 그리고 각 단계의 소유자를 말합니다.
- tool description을 name, JSON Schema input, deterministic executor function의 세 부분으로 작성합니다.
- pure tool과 side-effecting tool을 구분하고, 이 구분이 safety에 중요한 이유를 설명합니다.

## 문제

LLM은 다음 토큰에 대한 확률 분포를 출력합니다. 이것이 출력 표면의 전부입니다. chat model에게 "Bengaluru의 지금 날씨가 어때?"라고 물으면 그럴듯한 문장을 쓸 수는 있지만, weather API에 접속할 수는 없습니다. 그 문장은 우연히 맞을 수도 있고, 사흘 전 정보일 수도 있습니다.

그 간극을 닫는 것이 tool interface의 목적입니다. host program, 즉 agent runtime, Claude Desktop, ChatGPT, Cursor 또는 custom script가 호출 가능한 tool 목록을 모델에 광고합니다. 모델은 어떤 행동이 필요하다고 판단하면 tool 이름과 arguments를 담은 구조화된 payload를 출력합니다. host는 그 payload를 parse하고, 실제로 tool을 실행한 뒤, 결과를 다시 넣어 줍니다. 모델이 더 이상 호출이 필요 없다고 판단할 때까지 loop가 계속됩니다.

이 계약의 첫 버전은 2023년 6월 OpenAI의 "functions" parameter로 출시되었습니다. Anthropic은 Claude 2.1에서 `tool_use` block으로 뒤따랐습니다. Gemini는 몇 달 뒤 `functionDeclarations`를 추가했습니다. 이제 모든 provider가 같은 형태를 제공합니다. 입력에는 JSON-Schema로 타입이 지정된 tool list, 출력에는 JSON payload tool call입니다. Model Context Protocol(2024년 11월)은 이 계약을 일반화해 하나의 tool registry가 모든 모델을 담당하게 했습니다. A2A(2026년 4월, v1.0)는 agent-to-agent delegation을 위해 같은 primitive를 층으로 얹었습니다.

이 모든 것 아래에 있는 불변식이 4단계 loop입니다. Phase 13의 나머지는 모두 이 loop를 확장한 것입니다.

## 개념

### 1단계: describe

host는 각 tool을 세 필드로 선언합니다.

- **Name.** 안정적인 machine-readable identifier입니다. "weather thing"이 아니라 `get_weather`입니다.
- **Description.** 한 문단짜리 natural-language brief입니다. "사용자가 특정 도시의 현재 상태를 물을 때 사용한다. 과거 데이터에는 사용하지 않는다."
- **Input schema.** tool의 arguments를 설명하는 JSON Schema object(draft 2020-12)입니다.

모델은 이 목록을 받습니다. 최신 provider들은 provider-specific template을 사용해 이 선언들을 system prompt로 serialize하므로, caller인 당신은 structured form만 다루면 됩니다.

### 2단계: decide

사용자 메시지와 사용 가능한 tool이 주어지면 모델은 세 가지 행동 중 하나를 선택합니다.

1. 텍스트로 **직접 답변**합니다. tool call은 없습니다.
2. **하나 이상의 tool을 호출**합니다. structured call object를 출력합니다. `parallel_tool_calls: true`(OpenAI와 Gemini는 기본값, Anthropic은 opt-in)에서는 모델이 한 turn에 여러 call을 출력할 수 있습니다.
3. **거부**합니다. strict-mode structured output은 call 대신 typed `refusal` block을 만들 수 있습니다.

tool call payload에는 안정적인 세 필드가 있습니다. call `id`, tool `name`, JSON `arguments` object입니다. id는 host가 나중의 result를 특정 call과 연결할 수 있게 하려고 존재합니다. parallel call이 순서와 다르게 돌아올 때 특히 중요합니다.

### 3단계: execute

host는 call을 받고, 선언된 schema에 대해 arguments를 validate한 뒤 executor를 실행합니다. invalid arguments는 모델이 존재하지 않는 필드를 hallucinate했거나 잘못된 type을 사용했다는 뜻입니다. 약한 모델에서 매우 흔한 failure mode입니다. production host는 invalid arguments에 대해 보통 세 가지 중 하나를 합니다. 빠르게 실패하고 error를 모델에 노출하거나, constrained parser로 JSON을 repair하거나, validation error를 prompt에 포함해 모델을 retry합니다.

executor 자체는 평범한 코드입니다. Python, TypeScript, shell command, database query가 될 수 있습니다. executor는 result를 생성합니다. 보통 string이지만 어떤 JSON value나 structured content block(MCP에서는 text, image, resource reference)도 될 수 있습니다. result는 serializable해야 합니다.

### 4단계: observe

host는 tool result를 conversation에 추가하고(matching `id`가 있는 `tool` role message), 모델을 다시 호출합니다. 이제 모델은 context 안에 tool output을 가지고 있으므로 final answer를 만들거나 추가 call을 요청할 수 있습니다. 모델이 call 출력을 멈추거나 host가 iteration count safety limit에 도달할 때까지 이 과정이 계속됩니다.

### trust split

tool에는 safety에 중요한 두 종류가 있습니다.

- **Pure.** read-only, deterministic, side effect 없음. `get_weather`, `search_docs`, `get_current_time`. 추측성으로 호출해도 안전합니다.
- **Consequential.** state를 변경하거나, 돈을 쓰거나, 사용자 데이터를 건드립니다. `send_email`, `delete_file`, `execute_trade`. 반드시 gate가 필요합니다.

Meta의 2026년 agent security "Rule of Two"는 한 turn이 다음 셋 중 최대 둘까지만 결합할 수 있다고 말합니다. untrusted input, sensitive data, consequential action입니다. tool interface는 call을 거부하거나, 사용자 확인을 요구하거나, scope를 escalate하여 이 규칙을 강제하는 곳입니다. 전체 security 장은 Phase 13 · 15를, agent-level permission policy는 Phase 14 · 09를 참고하세요.

### loop가 사는 곳

| 맥락 | 설명 주체 | 결정 주체 | 실행 주체 |
|---------|---------------|-------------|--------------|
| 단일 turn function calling(OpenAI/Anthropic/Gemini) | app developer | LLM | app developer |
| MCP | MCP server | MCP client를 통한 LLM | MCP server |
| A2A | Agent Card 게시자 | 호출 agent | 호출받은 agent |
| Web browser(function-calling agent) | browser extension / WebMCP | LLM | browser runtime |

어디서나 같은 네 단계입니다. 열 이름은 바뀌지만 구조는 바뀌지 않습니다.

### 모델에게 JSON을 출력하라고 prompt하면 안 되는 이유

"JSON으로 답하라고 모델에게 요청하기"는 function calling 이전의 패턴이었습니다. frontier model에서도 5~15% 정도 실패하고, 작은 모델에서는 훨씬 더 많이 실패합니다. failure mode에는 빠진 중괄호, trailing comma, hallucinated field, wrong type이 포함됩니다. 그러면 JSON repair pass, retry 또는 constrained decoder가 필요합니다.

native function calling이 더 나은 이유는 세 가지입니다. 첫째, provider가 정확한 call shape에 대해 end-to-end로 모델을 훈련하므로 strict mode에서 valid-JSON rate가 98~99%까지 올라갑니다. 둘째, call payload가 free-text 안이 아니라 자체 protocol slot에 있으므로 tool call이 사용자에게 보이는 reply로 새지 않습니다. 셋째, provider가 constrained decoding(OpenAI의 strict mode, Anthropic의 `tool_use`, Gemini의 `responseSchema`)으로 schema compliance를 강제합니다. 출력은 validate된다는 보장을 받습니다.

Phase 13 · 02는 세 provider API를 나란히 살펴봅니다. Phase 13 · 04는 structured output을 깊게 다룹니다.

### circuit breaker

loop는 모델이 call 출력을 멈추거나 host가 maximum turn count에 도달하면 종료됩니다. production host는 이를 보통 5~20 turn 사이로 설정합니다. 그 이상이면 모델이 빠져나올 수 없는 loop에 들어갔을 가능성이 매우 큽니다. Claude Code의 기본값은 20, OpenAI Assistants는 10, Cursor의 agent mode는 25입니다.

대안인 unbounded loop는 6개월마다 "agent가 밤새 API call에 400달러를 썼다"는 post-mortem으로 나타납니다. bound 없이 배포하지 마세요.

Phase 14 · 12는 error recovery와 self-healing을 자세히 다룹니다. Phase 17은 production rate limit을 다룹니다.

### Phase 13의 다음 방향

- Lesson 02부터 05까지는 provider-level tool-call surface를 다듬습니다.
- Lesson 06부터 14까지는 loop를 MCP로 일반화합니다.
- Lesson 15부터 18까지는 적대적 server, adversarial user, 인증되지 않은 remote auth surface로부터 loop를 방어합니다.
- Lesson 19부터 22까지는 이 패턴을 agent-to-agent collaboration, observability, routing, packaging으로 확장합니다.
- Lesson 23은 모든 primitive를 사용하는 완전한 ecosystem을 출하합니다.

남은 모든 레슨은 이 4단계 loop의 확장입니다. 이를 불변식으로 기억하세요.

## 사용하기

`code/main.py`는 LLM 없이 4단계 loop를 실행합니다. 가짜 "decider" function이 사용자 메시지를 pattern-match해 모델을 simulation합니다. executor, schema validator, observe-step harness는 실제입니다. 실행해서 printable intermediate state와 함께 전체 request/response choreography를 확인한 뒤, 이후 레슨에서 fake decider를 실제 provider로 바꿔 보세요.

살펴볼 것:

- tool registry는 tool마다 name, description, schema, executor reference라는 세 필드를 보관합니다.
- validator는 stdlib만으로 작성한 최소 JSON Schema subset(types, required, enum, min/max)입니다. Phase 13 · 04는 더 완전한 validator를 제공합니다.
- loop는 iteration count를 5로 제한합니다. production agent에는 정확히 이런 circuit breaker가 필요합니다.

## 산출물

이 레슨은 `outputs/skill-tool-interface-reviewer.md`를 만듭니다. draft tool definition(name + description + schema + executor outline)이 주어지면, 이 skill은 loop 적합성을 audit합니다. name이 machine-stable한지, description이 완전한 usage brief인지, schema가 JSON Schema 2020-12를 올바르게 사용하는지, pure-vs-consequential classification이 명시적인지를 확인합니다.

## 연습 문제

1. `code/main.py`에 `get_stock_price(ticker)`라는 네 번째 tool을 추가하세요. description은 "Use when the user asks for a current stock price by ticker. Do not use for historical prices or market summaries."로 작성하세요. harness를 실행하고 ticker를 언급한 query가 fake decider에 의해 새 tool로 route되는지 확인하세요.

2. schema validator를 깨 보세요. `arguments` object에 required field가 빠진 call을 전달하고, host가 execution 전에 이를 reject하는지 확인하세요. 그런 다음 알 수 없는 extra field가 있는 call을 전달하세요. host는 reject해야 할까요, ignore해야 할까요? safety argument로 선택을 정당화하세요.

3. harness의 각 tool을 pure 또는 consequential로 분류하세요. 필요한 registry entry에 `consequential: true` flag를 추가하고, consequential tool이 선택될 때마다 loop가 "would confirm with user" line을 출력하도록 바꾸세요. 이것이 모든 production host에 필요한 confirmation gate의 형태입니다.

4. 위 provider-column table을 좋아하는 client(Claude Desktop, Cursor, ChatGPT 또는 custom stack)에 맞게 채운 뒤, 4단계 loop를 종이에 그려 보세요. Phase 13 · 06의 MCP-specific variant와 cross-reference하세요.

5. OpenAI의 function-calling guide를 처음부터 끝까지 읽으세요. 여기 제시된 4단계 loop에는 없지만 request에는 있는 필드 하나를 찾으세요. 그것이 무엇을 더하고, 왜 필수라기보다 편의 기능인지 설명하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Tool | "모델이 호출할 수 있는 것" | name + JSON-Schema-typed input + executor function의 삼중체 |
| Function calling | "Native tool use" | prose 대신 structured tool call을 출력하게 하는 provider-level API support |
| Tool call | "행동하라는 모델의 요청" | 모델이 출력하는 `id`, `name`, `arguments`가 있는 JSON payload |
| Tool result | "tool이 반환한 것" | matching id가 있는 `tool` role message로 감싼 executor output |
| Parallel tool calls | "여러 call을 한 번에" | 한 model turn 안의 여러 call object, id로 독립적이며 순서 지정 가능 |
| Strict mode | "Guaranteed JSON" | 모델 출력이 선언된 schema에 validate되도록 강제하는 constrained decoding |
| Pure tool | "Read-only tool" | side effect가 없고 다시 실행해도 안전함 |
| Consequential tool | "Action tool" | external state를 변경하며 gate, audit 또는 사용자 확인이 필요함 |
| Four-step loop | "tool-call cycle" | describe -> decide -> execute -> observe |
| Host | "Agent runtime" | tool registry를 보관하고, 모델을 호출하고, executor를 실행하는 프로그램 |

## 더 읽을거리

- [OpenAI - Function calling guide](https://platform.openai.com/docs/guides/function-calling) - OpenAI-style tool declaration과 call shape의 canonical reference
- [Anthropic - Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) - Claude의 `tool_use` / `tool_result` block format
- [Google - Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) - Gemini의 `functionDeclarations`와 parallel-call semantics
- [Model Context Protocol - Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) - tool interface의 provider-agnostic generalization
- [JSON Schema - 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) - 모든 최신 tool API가 사용하는 schema dialect
