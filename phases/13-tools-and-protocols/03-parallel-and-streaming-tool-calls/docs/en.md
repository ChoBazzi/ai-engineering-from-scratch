# Parallel Tool Calls와 Tool Streaming

> 독립적인 weather lookup 세 개를 직렬화하면 round trip이 세 번입니다. 병렬로 실행하면 전체 시간은 가장 느린 단일 call 수준으로 줄어듭니다. 모든 frontier provider는 이제 한 turn 안에서 여러 tool call을 출력합니다. 이득은 실제지만 plumbing은 미묘합니다. 이 레슨은 parallel fan-out과 streamed-argument reassembly 두 부분을 모두 다루며, id-correlation 함정에 특히 집중합니다.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## 학습 목표

- `parallel_tool_calls: true`가 왜 존재하며 언제 disable해야 하는지 설명합니다.
- parallel fan-out 중 streamed argument chunk를 올바른 tool-call id와 correlate합니다.
- partial `arguments` string을 너무 일찍 parse하지 않고 complete JSON으로 reassemble합니다.
- sequential latency와 parallel latency를 보여 주는 three-city weather benchmark를 실행합니다.

## 문제

parallel call이 없으면 "Bengaluru, Tokyo, Zurich의 날씨는 어때?"에 답하는 agent는 이렇게 동작합니다.

```text
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

LLM round trip이 세 번이고, 각각 executor latency도 냅니다. 이상적인 wall-clock time보다 대략 4배 걸립니다.

parallel call을 사용하면:

```text
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

LLM round trip은 한 번입니다. executor time은 세 call의 합이 아니라 최댓값입니다. OpenAI, Anthropic, Gemini의 production benchmark는 fan-out workload에서 wall-clock이 60~70% 감소함을 보여 줍니다.

대가는 correlation complexity입니다. 세 call이 순서와 다르게 완료되면, result가 matching `tool_call_id`를 가지고 있어야 모델이 줄을 맞출 수 있습니다. result가 stream될 때는 실행 전에 partial argument fragment를 complete JSON으로 조립해야 합니다. Gemini 3가 unique id를 추가한 이유 중 하나는 같은 tool에 대한 두 parallel call을 구분할 수 없었던 실제 문제를 해결하기 위해서였습니다.

## 개념

### parallel 활성화

- **OpenAI.** `parallel_tool_calls: true`가 기본값입니다. serial을 강제하려면 `false`로 설정합니다.
- **Anthropic.** `disable_parallel_tool_use: false`를 통해 parallel입니다(Claude 3.5 이상 기본값). serial은 `true`로 설정합니다.
- **Gemini.** 항상 parallel-capable입니다. `tool_config.function_calling_config.mode = "AUTO"`가 모델에게 결정하게 합니다.

tool에 ordering dependency가 있을 때(`create_file` 다음 `write_file`), 한 call의 output이 다른 call의 input에 영향을 줄 때, 또는 rate limiter가 fan-out을 감당할 수 없을 때 parallel을 disable하세요.

### id correlation

모델이 출력하는 모든 call에는 `id`가 있습니다. host가 반환하는 모든 result는 같은 id를 포함해야 합니다. 이것이 없으면 result가 모호해집니다.

- **OpenAI.** 각 tool-role message의 `tool_call_id`.
- **Anthropic.** 각 `tool_result` block의 `tool_use_id`.
- **Gemini.** 각 `functionResponse`의 `id`(Gemini 3 이상. Gemini 2는 name으로 match했는데 same-name parallel call에서 깨졌습니다).

### call을 concurrently 실행하기

host는 각 call의 executor를 별도 thread, coroutine 또는 remote worker에서 실행합니다. 가장 단순한 harness는 thread pool을 사용합니다. production은 `asyncio.gather` 또는 structured concurrency를 쓰는 asyncio를 사용합니다. 완료 순서는 예측할 수 없습니다. identifier는 id입니다.

흔한 bug 하나는 completion order가 아니라 call-list order로 result를 답하는 것입니다. 모델은 `tool_call_id`만 신경 쓰기 때문에 대체로 동작하지만, result가 누락되거나 중복되면 out-of-order submission이 debugging을 어렵게 합니다. explicit id와 함께 completion order로 답하는 편을 선호하세요.

### streaming tool call

모델이 stream할 때 `arguments`는 조각으로 도착합니다. 세 parallel call에 대한 세 개의 chunk stream이 wire에서 interleave됩니다. id마다 하나의 accumulator가 필요합니다.

provider별 shape:

- **OpenAI.** 각 chunk는 `choices[0].delta.tool_calls[i].function.arguments`(partial string)입니다. chunk에는 `index`(call list에서의 위치)가 있습니다. index별로 누적하고, 처음 나타날 때 `id`를 읽으며, `finish_reason = "tool_calls"`일 때 JSON을 parse합니다.
- **Anthropic.** stream event는 `message_start`, type이 `tool_use`인 block마다 하나의 `content_block_start`(id, name, 빈 input 포함), partial arguments를 담은 `input_json_delta` chunk가 있는 `content_block_delta`, 각 block을 닫는 `content_block_stop`입니다.
- **Gemini.** `streamFunctionCallArguments`(Gemini 3 이상)는 `functionCallId`가 있는 chunk를 내보내 call들이 깔끔하게 interleave되게 합니다. Gemini 3 이전에는 streaming이 한 번에 하나의 complete call을 반환했습니다.

### partial JSON과 parse-early 함정

`arguments`가 complete될 때까지 parse하면 안 됩니다. `{"city": "Beng` 같은 partial JSON은 valid하지 않으며 예외를 냅니다. 올바른 gate는 provider의 end-of-call signal입니다. OpenAI의 `finish_reason = "tool_calls"`, Anthropic의 `content_block_stop`, Gemini의 stream-end event가 그 신호입니다. 그때에만 `json.loads`를 시도하세요. 더 robust한 접근은 structure가 완성될 때 event를 yield하는 incremental JSON parser를 사용하는 것입니다. OpenAI의 streaming guide는 live "thinking" indicator를 보여 주는 UX에 이를 권장합니다. brace-counting은 quoted string이나 escaped content 안의 brace 때문에 false positive가 생겨 completeness test로 신뢰할 수 없고, informal debug heuristic으로만 써야 합니다.

### out-of-order completion

```text
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

host reply는 그래도 id를 인용해야 합니다.

```text
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

OpenAI나 Anthropic에서 correctness를 위해 reply order는 중요하지 않습니다. Gemini도 id가 match되기만 하면 어떤 순서든 받아들입니다.

### benchmark: sequential vs parallel

`code/main.py`의 harness는 400, 600, 800 ms latency를 가진 세 executor를 simulate합니다. sequential은 총 1800 ms에 실행됩니다. parallel은 max(400, 600, 800) = 800 ms에 실행됩니다. 차이는 비례가 아니라 상수에 가까우므로 tool count가 늘수록 savings가 커집니다.

현실의 caveat: parallel call은 downstream API를 압박합니다. rate-limited service에 대한 10-way fan-out은 실패합니다. Phase 13 · 17은 gateway-level backpressure를 다룹니다. retry semantics는 future phase에서 다룰 예정입니다.

### streaming fan-out wall-clock

모델 자체가 stream한다면 모든 call이 finalize될 때까지 기다리지 않고, 어떤 call의 arguments가 complete되는 즉시 실행을 시작할 수 있습니다. 이는 OpenAI가 문서화한 optimization이지만 모든 SDK가 노출하지는 않습니다. 이 레슨의 harness는 이를 구현합니다. simulated stream이 complete argument object를 yield하는 즉시 host가 해당 call을 시작합니다.

## 사용하기

`code/main.py`는 두 부분으로 되어 있습니다. 첫 번째는 `concurrent.futures.ThreadPoolExecutor`를 사용해 세 개의 simulated weather call을 sequential과 parallel로 실행하고 wall-clock time을 출력합니다. 두 번째는 fake streaming response, 즉 하나의 stream에서 interleave된 세 parallel call의 `arguments` chunk를 replay하고 `StreamAccumulator`로 id별 reassembly를 수행합니다. LLM도 network도 없고, reassembly logic만 있습니다.

살펴볼 것:

- sequential timer는 1.8초에 도달합니다. parallel timer는 같은 fake latency에서 0.8초에 도달합니다.
- accumulator는 id별로 buffer를 유지하고 각 call의 JSON이 complete될 때만 parse하여, out-of-order로 도착하는 chunk를 처리합니다.
- executor는 모든 stream이 끝난 뒤가 아니라 id의 arguments가 finalize되는 즉시 시작됩니다.

## 산출물

이 레슨은 `outputs/skill-parallel-call-safety-check.md`를 만듭니다. tool registry가 주어지면 이 skill은 어떤 tool이 parallelize하기에 안전한지, 어떤 tool에 ordering dependency가 있는지, 어떤 tool이 downstream rate limit을 압도할지 audit하고, per-tool `parallel_safe` flag가 있는 revised registry를 반환합니다.

## 연습 문제

1. `code/main.py`를 실행하고 simulated latency를 바꿔 보세요. parallel-to-sequential ratio가 대략 `max/sum`인지 확인하세요(real run은 thread scheduling, serialization, harness overhead 때문에 이상적인 값에서 조금 벗어납니다). 어떤 latency distribution에서 parallel이 더 이상 중요하지 않습니까?

2. "call이 stream 중간에 cancel됨" case를 처리하도록 accumulator를 확장하세요. buffer를 drop하고 `cancelled` event를 내보내면 됩니다. 어떤 provider가 이 case를 명시적으로 문서화합니까? Anthropic의 `content_block_stop` semantics와 OpenAI의 `finish_reason: "length"` behavior를 확인하세요.

3. thread pool을 `asyncio.gather`로 교체하세요. 둘 다 benchmark하세요. executor가 실제 I/O를 수행할 때만 async에서 context-switch cost가 낮아 작은 이득을 볼 것입니다.

4. parallelize하면 안 되는 tool 두 개를 고르세요(예: `create_file` 다음 `write_file`). registry에 `ordering_dependency` graph를 추가하고, 그 graph에 따라 parallel fan-out을 gate하세요. 이것이 dependency-aware scheduling에 필요한 최소 장치이며, future agent-engineering phase가 이를 formalize합니다.

5. OpenAI의 parallel-function-calling section과 Anthropic의 `disable_parallel_tool_use` docs를 읽으세요. Anthropic이 parallelism disable을 권장하는 실제 tool type 하나를 찾으세요. 힌트: 같은 resource에 대한 consequential mutation입니다.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Parallel tool calls | "한 turn의 fan-out" | 모델이 하나의 assistant message 안에 여러 tool call을 출력함 |
| `parallel_tool_calls` | "OpenAI의 flag" | multi-call emission을 enable 또는 disable |
| `disable_parallel_tool_use` | "Anthropic의 inverse" | opt-out flag. 기본값은 parallel enabled |
| Tool call id | "Correlation handle" | result message가 echo해야 하는 per-call identifier |
| Accumulator | "Stream buffer" | partial `arguments` chunk를 위한 per-id string buffer |
| Out-of-order completion | "Fastest first" | parallel call은 예측할 수 없는 순서로 끝나며 id가 glue 역할을 함 |
| Dependency graph | "Ordering constraints" | 어떤 tool의 output이 다른 tool의 input으로 들어가면 parallelize할 수 없음 |
| Parse-early trap | "JSON.parse exploded" | incomplete `arguments` string을 parse하려는 시도 |
| `streamFunctionCallArguments` | "Gemini 3 feature" | call마다 unique id가 있는 streamed argument chunk |
| Completion-order reply | "Don't wait for all" | id로 keying된 result를 도착하는 대로 답함 |

## 더 읽을거리

- [OpenAI - Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) - 기본 동작과 opt-out flag
- [Anthropic - Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) - `disable_parallel_tool_use`와 result batching
- [Google - Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) - Gemini 3의 id-correlated parallel call
- [OpenAI - Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) - OpenAI stream의 chunked argument reassembly
- [Anthropic - Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) - `input_json_delta`가 있는 `content_block_delta`
