# 구조화 출력과 Constrained Decoding

> LLM에 JSON을 요청합니다. 대부분 JSON을 받습니다. Production에서는 "대부분"이 문제입니다. Constrained decoding은 sampling 전에 logits를 수정해 "대부분"을 "항상"으로 바꿉니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 minutes

## 문제

classifier가 LLM에 prompt합니다: "Return one of {positive, negative, neutral}." 모델은 "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ..."를 반환합니다. parser가 crash합니다. classifier의 F1은 0.0입니다.

Free-form generation은 contract가 아닙니다. 제안일 뿐입니다. Production system에는 contract가 필요합니다.

2026년에는 세 layer가 있습니다.

1. **Prompting.** 정중하게 요청합니다. "Return only the JSON object." Frontier model에서는 ~80% 동작하고, smaller model에서는 더 낮습니다.
2. **Native structured output APIs.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. 지원되는 schema에서는 reliable합니다. Vendor-locked입니다.
3. **Constrained decoding.** 매 generation step에서 logits를 수정해 모델이 invalid token을 *emit할 수 없게* 만듭니다. 구조상 100% valid합니다. 어떤 local model에서도 동작합니다.

이 lesson은 세 방법 모두에 대한 intuition을 만들고, 언제 무엇을 선택해야 하는지 이름 붙입니다.

## 개념

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Constrained decoding의 작동 방식.** 각 generation step에서 LLM은 전체 vocabulary(~100k tokens)에 대한 logit vector를 생성합니다. *logit processor*가 model과 sampler 사이에 놓입니다. 현재 target grammar(JSON Schema, regex, context-free grammar)에서 어떤 token이 valid한지 계산하고, 모든 invalid token의 logits를 negative infinity로 설정합니다. 남은 logits에 대한 softmax는 valid continuation에만 probability mass를 둡니다.

2026년 구현체:

- **Outlines.** JSON Schema 또는 regex를 finite-state machine으로 compile합니다. 모든 token은 O(1) valid-next-token lookup을 갖습니다. FSM 기반이므로 recursive schema는 flattening이 필요합니다.
- **XGrammar / llguidance.** Context-free grammar engine입니다. Recursive JSON Schema를 처리합니다. decoding overhead가 거의 0입니다. OpenAI는 2025 structured output 구현에서 llguidance를 언급했습니다.
- **vLLM guided decoding.** Outlines, XGrammar, 또는 lm-format-enforcer backend를 통한 built-in `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`.
- **Instructor.** 모든 LLM 위의 Pydantic-based wrapper입니다. validation failure 시 retry합니다. Cross-provider이지만 logits는 수정하지 않습니다. retries + structured-output-aware prompts에 의존합니다.

### 직관에 반하는 결과

Constrained decoding은 종종 unconstrained generation보다 *빠릅니다*. 이유는 두 가지입니다. 첫째, next-token search space를 줄입니다. 둘째, 영리한 구현은 forced token(scaffolding like `{"name": "` — 모든 byte가 결정됨)에 대해서는 token generation을 완전히 건너뜁니다.

### 비용을 발생시키는 함정

Field order가 중요합니다. `answer`를 `reasoning`보다 앞에 두면 모델은 생각하기 전에 answer에 commit합니다. JSON은 valid합니다. Answer는 틀렸습니다. 어떤 validation도 이를 잡지 못합니다.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Schema field order는 formatting이 아니라 logic입니다.

## 직접 만들기

### 1단계: 처음부터 구현하는 regex-constrained generation

standalone FSM 구현은 `code/main.py`를 보세요. 30줄로 표현한 핵심 아이디어:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM은 지금까지 grammar의 어떤 부분을 만족했는지 추적합니다. `valid_tokens(state, tokenizer)`는 accepting path를 벗어나지 않고 FSM을 전진시킬 수 있는 vocabulary token을 계산합니다.

### 2단계: JSON Schema를 위한 Outlines

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Validation error가 없습니다. 절대 없습니다. FSM이 invalid output을 unreachable하게 만듭니다.

### 3단계: provider-agnostic Pydantic을 위한 Instructor

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

메커니즘이 다릅니다. Instructor는 logits를 건드리지 않습니다. schema를 prompt로 format하고, output을 parse하고, validation failure 시 retry합니다(default 3 times). 어떤 provider와도 동작합니다. retry는 latency와 cost를 추가합니다. Cross-provider portability가 selling point입니다.

### 4단계: native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Server-side constrained decoding입니다. 지원되는 schema에서는 Outlines와 reliability가 동등합니다. Local model management가 필요 없습니다. vendor에 lock-in됩니다.

## 함정

- **Recursive schemas.** Outlines는 recursion을 fixed depth로 flatten합니다. Tree-structured output(nested comments, AST)에는 XGrammar 또는 llguidance(CFG-based)가 필요합니다.
- **Huge enums.** 10,000-option enum은 compile이 느리거나 timeout됩니다. retriever로 전환하세요. 먼저 top-k candidate를 예측하고 그 안으로 constrain합니다.
- **Grammar too strict.** `date: "YYYY-MM-DD"` regex를 강제하면 missing date에 대해 모델이 `"unknown"`을 출력할 수 없습니다. 모델은 date를 지어내는 방식으로 보상합니다. `null` 또는 sentinel을 허용하세요.
- **Premature commitment.** 위 field-order 함정을 보세요. 항상 reasoning을 먼저 두세요.
- **Vendor JSON mode without schema.** 순수 JSON mode는 valid JSON만 보장할 뿐, *your use case*에 valid하다는 것은 보장하지 않습니다. 항상 full schema를 제공하세요.

## 사용하기

2026년 stack:

| 상황 | 선택 |
|-----------|------|
| OpenAI/Anthropic/Google model, simple schema | Native vendor structured output |
| Any provider, Pydantic workflow, retry를 감수 가능 | Instructor |
| Local model, 100% validity 필요, flat schema | Outlines (FSM) |
| Local model, recursive schema | XGrammar or llguidance |
| Self-hosted inference server | vLLM guided decoding |
| retry가 허용되는 batch processing | Instructor + cheapest model |

## 배포하기

`outputs/skill-structured-output-picker.md`로 저장하세요:

```markdown
---
name: structured-output-picker
description: structured output approach, schema design, validation plan을 선택합니다.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

use case(provider, latency budget, schema complexity, failure tolerance)가 주어지면 다음을 출력합니다.

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, 또는 XGrammar CFG. 한 문장 이유.
2. Schema design. Field order(reasoning first, answer last), "unknown"을 위한 nullable fields, enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate(target 100%), semantic validity(LLM-judge), field-coverage rate, latency p50/p99.

`answer` 또는 `decision`을 reasoning fields보다 앞에 두는 design을 거부하세요. schema 없는 bare JSON mode 사용을 거부하세요. recursive schema가 FSM-only library 뒤에 있으면 flag하세요.
```

## 연습 문제

1. **Easy.** constrained decoding 없이 small open-weights model(예: Llama-3.2-3B)에 `Review(sentiment, confidence, evidence_span)`를 prompt하세요. review 100개에서 valid JSON으로 parse되는 비율을 측정하세요.
2. **Medium.** 같은 corpus를 Outlines JSON mode로 실행하세요. compliance rate, latency, semantic accuracy를 비교하세요.
3. **Hard.** phone number(`\d{3}-\d{3}-\d{4}`)를 위한 regex-constrained decoder를 처음부터 구현하세요. sample 1000개에서 invalid output 0개를 검증하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Constrained decoding | valid output을 강제 | 모든 generation step에서 invalid-token logits를 mask합니다. |
| Logit processor | constrain하는 것 | Function: `(logits, state) -> masked_logits`. |
| FSM | Finite-state machine | compile된 grammar representation입니다. O(1) valid-next-token lookup. |
| CFG | Context-free grammar | recursion을 처리하는 grammar입니다. FSM보다 느리지만 더 expressive합니다. |
| Schema field order | 이게 중요한가? | 네. 첫 field가 commit합니다. 항상 answer보다 reasoning을 먼저 두세요. |
| Guided decoding | vLLM에서 부르는 이름 | 같은 개념을 inference server에 통합한 것입니다. |
| JSON mode | OpenAI의 초기 버전 | JSON syntax는 보장하지만 schema match는 보장하지 않습니다. |

## 더 읽을거리

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — Outlines paper.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — fast CFG-based constrained decoding.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — inference server integration.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — API reference + gotchas.
- [Instructor library](https://python.useinstructor.com/) — provider 전반의 Pydantic + retries.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — constrained decoding framework 6개의 benchmark.
