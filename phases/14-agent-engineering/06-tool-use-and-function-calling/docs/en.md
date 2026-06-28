# Tool Use and Function Calling

> Toolformer(Schick et al., 2023)는 self-supervised tool annotation을 시작했습니다. Berkeley Function Calling Leaderboard V4(Patil et al., 2025)는 2026년 기준선을 세웁니다. 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination입니다. Single-turn은 해결되었습니다. Memory, dynamic decision-making, long-horizon tool chain은 아직 아닙니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 minutes

## Learning Objectives

- Toolformer의 self-supervised training signal을 설명합니다. execution이 next-token loss를 줄일 때만 tool annotation을 유지합니다.
- BFCL V4의 다섯 evaluation category와 각 category가 측정하는 것을 이름 붙입니다.
- schema validation, argument coercion, execution sandboxing을 갖춘 stdlib tool registry를 구현합니다.
- 2026년의 세 open problem인 long-horizon tool chaining, dynamic decision-making, memory를 진단합니다.

## The Problem

초기 tool use는 모델이 올바른 function call을 예측할 수 있는가를 물었습니다. 현대 tool use는 모델이 40 step에 걸쳐, memory와 partial observability를 갖고, tool failure에서 회복하며, 존재하지 않는 tool을 hallucinate하지 않고 tool을 chain할 수 있는가를 묻습니다.

Toolformer는 baseline을 세웠습니다. model은 self-supervision으로 언제 tool을 호출할지 배울 수 있습니다. BFCL V4는 2026년 evaluation target을 정의합니다. 그 사이의 gap이 production agent가 사는 공간입니다.

## The Concept

### Toolformer (Schick et al., NeurIPS 2023)

아이디어는 model이 자신의 pretraining corpus에 candidate API call을 annotate하게 하는 것입니다. 각 candidate를 실행합니다. tool result를 포함했을 때 next token에 대한 loss가 줄어드는 annotation만 유지합니다. filtered corpus로 fine-tune합니다.

다룬 tool은 calculator, QA system, search engines, translator, calendar입니다. self-supervision signal은 tool이 text prediction에 도움이 되는지에만 관한 것입니다. human label은 없습니다.

scale result: tool use는 scale에서 나타납니다. 작은 model은 tool annotation으로 손해를 보고, 큰 model은 이득을 봅니다. 그래서 2026년 frontier model은 강력한 tool use가 내장되어 있지만, 대부분 7B model은 reliable하려면 explicit tool-use fine-tuning이 필요합니다.

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL은 2026년 de facto evaluation입니다. V4 구성은 다음과 같습니다.

- **Agentic (40%)** - full agent trajectory: memory, multi-turn, dynamic decisions.
- **Multi-Turn (30%)** - tool chain이 있는 interactive conversation.
- **Live (10%)** - user-submitted real prompt(더 어려운 distribution).
- **Non-Live (10%)** - synthetic test case.
- **Hallucination (10%)** - tool을 호출하지 않아야 할 때를 detect.

V3는 state-based evaluation을 도입했습니다. tool sequence 후 API의 실제 state를 check합니다(예: "file이 생성되었는가?"). tool call AST를 match하는 것이 아닙니다. V4는 web search, memory, format sensitivity category를 추가했습니다.

2026년 핵심 발견: single-turn function calling은 거의 해결되었습니다. 실패는 memory(turn 사이 context 유지), dynamic decision-making(prior result에 따라 tool 선택), long-horizon chain(20+ step 이후 drift), hallucination detection(맞는 tool이 없을 때 call 거부)에 집중됩니다.

### Tool schema

모든 provider에는 schema가 있습니다. detail은 다르지만 shape는 같습니다.

```text
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic은 `input_schema`를 직접 사용합니다. OpenAI는 `function.parameters`를 사용합니다. 둘 다 JSON Schema를 받습니다. description은 load-bearing입니다. model은 올바른 tool을 고르기 위해 description을 읽습니다. 나쁜 tool description은 wrong-tool-picked failure의 #1 root cause입니다.

### Argument validation

tool call을 신뢰하지 마세요. 다음을 validate합니다.

1. **Type coercion.** model은 schema가 int라고 해도 string `"5"`를 반환할 수 있습니다. 모호하지 않으면 coerce하고, 그렇지 않으면 reject합니다.
2. **Enum validation.** schema가 `status in {"open", "closed"}`라고 하는데 model이 `"in_progress"`를 내면 descriptive error로 reject합니다.
3. **Required fields.** required field가 없으면 crash가 아니라 immediate error observation을 model에 되돌립니다.
4. **Format validation.** date, email, URL은 regex가 아니라 concrete parser로 validate합니다.

모든 validation failure는 model이 올바른 shape로 retry할 수 있도록 structured observation을 반환해야 합니다.

### Parallel tool calls

현대 provider는 한 assistant turn 안의 parallel tool call을 지원합니다. loop는 다음과 같습니다.

1. model이 distinct `tool_use_id`가 있는 tool call 3개를 방출합니다.
2. runtime이 이를 실행합니다(독립적이면 parallel).
3. 각 result는 `tool_use_id`로 correlate된 `tool_result` block으로 되돌아갑니다.

engineering rule: correlation ID를 load-bearing으로 취급하세요. 바뀌면 wrong tool에 wrong result가 route됩니다.

### Sandboxing

tool execution은 sandbox boundary입니다. 자세한 내용은 Lesson 09를 보세요. 짧게 말하면 모든 tool은 read/write surface, network access, timeout, memory cap을 명시해야 합니다. generic `run_shell(cmd)`는 red flag입니다. specific `git_status()`가 더 안전합니다.

```figure
tool-routing
```

## Build It

`code/main.py`는 production 형태의 tool registry를 구현합니다.

- JSON Schema subset validator(stdlib only).
- description, input schema, timeout, executor를 갖춘 tool registration.
- argument coercion과 enum validation.
- correlation ID를 사용한 parallel tool dispatch.
- structured string으로 된 error observation.

실행:

```bash
python3 code/main.py
```

trace는 mini agent가 한 turn에 세 tool을 호출하는 모습을 보여 줍니다. 그중 하나는 일부러 malformed call이며, model이 조치할 수 있는 descriptive error로 reject됩니다.

## Use It

Anthropic, OpenAI, Gemini, Bedrock 등 모든 provider는 고유한 tool schema를 가집니다. multi-provider가 필요하면 translation layer(OpenAI Agents SDK, Vercel AI SDK, LangChain tool adapter)를 사용하세요. BFCL은 reference benchmark입니다. tool use가 product의 중심이라면 배포 전에 agent에 대해 실행하세요.

## Ship It

`outputs/skill-tool-registry.md`는 주어진 task domain에 대해 tool catalog, schema, registry를 생성합니다. description-quality check도 포함합니다(each tool의 description이 model에게 언제 써야 하는지 말해 주는가?).

## Exercises

1. model이 다른 tool을 명시적으로 거부할 수 있게 하는 "no-op" tool을 추가하세요. BFCL-like hallucination test에서 측정하세요.
2. int-as-string과 float-as-string에 대한 argument coercion을 구현하세요. coercion은 어디서부터 real bug를 숨기기 시작하나요?
3. per-tool timeout과 circuit breaker(3번 연속 실패 후 60초 동안 tool 거부)를 추가하세요. model이 회복하는 방식은 어떻게 달라지나요?
4. BFCL V4 description을 읽으세요. category 하나(예: "multi-turn")를 고르고 example prompt 10개를 agent에 실행하세요. pass rate를 보고하세요.
5. stdlib validator를 Pydantic 또는 Zod로 port하세요. toy가 놓친 것 중 Pydantic/Zod가 잡은 것은 무엇인가요?

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Function calling | "Tool use" | validated schema가 있는 structured-output tool invocation |
| Toolformer | "Self-supervised tool annotation" | Schick 2023 - result가 next-token loss를 줄이는 tool call만 유지 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 benchmark: 40% agentic, 30% multi-turn, 10% live, 10% non-live, 10% hallucination |
| Tool schema | "Function signature for the model" | name, description, arguments의 JSON Schema |
| tool_use_id | "Correlation ID" | tool call과 result를 연결. parallel dispatch에 필수 |
| Hallucination detection | "Know when not to call" | 맞는 tool이 없을 때 call을 거부하는 V4 category |
| Argument coercion | "String-to-int repair" | predictable schema mismatch에 대한 좁은 수정. 모호하면 reject |
| Sandboxing | "Tool execution boundary" | per-tool read/write surface, network, timeout, memory cap |

## Further Reading

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761) - self-supervised tool annotation
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html) - 2026 eval benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude Agent SDK의 production tool schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - function tool type과 Guardrails
