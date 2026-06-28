# 프롬프트 캐싱과 컨텍스트 캐싱

> System prompt가 4,000 token입니다. RAG context가 20,000 token입니다. 매 요청마다 둘 다 보냅니다. 그리고 매번 둘 다 비용을 냅니다. Prompt caching은 provider가 그 prefix를 자기 쪽에서 따뜻하게 유지하게 하고, 재사용 시 일반 요금의 10%만 청구하게 합니다. 제대로 쓰면 inference cost를 50-90%, first-token latency를 40-85% 줄일 수 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 minutes

## 문제

Coding agent가 대화의 모든 turn마다 같은 15,000-token system prompt를 Claude에 보냅니다. Input token이 $3/M이라면 20 turn의 input cost만 $0.90입니다. 사용자의 실제 message 비용은 아직 포함하지도 않았습니다. 매일 10,000개 대화로 곱하면, 절대 바뀌지 않는 text에만 하루 $9,000이 듭니다.

품질을 해치지 않고 prompt를 줄일 수는 없습니다. 보내지 않을 수도 없습니다. 모델은 매 turn마다 그것이 필요합니다. 유일한 수는 provider가 이미 본 prefix에 대해 full price를 내지 않는 것입니다.

그 수가 prompt caching입니다. Anthropic은 2024년 8월 이를 출시했고(2025년에는 1-hour extended-TTL 변형 포함), OpenAI는 그해 말 자동화를 도입했으며, Google은 Gemini 1.5와 함께 explicit context caching을 출시했습니다. 이제 세 provider 모두 frontier model에서 이를 first-class feature로 제공합니다.

## 개념

![Prompt caching: 한 번 쓰고, 싸게 읽기](../assets/prompt-caching.svg)

**Mechanic.** 요청의 prefix가 최근 요청의 prefix와 일치하면 provider는 token을 다시 encode하지 않고 이전 실행의 KV-cache를 제공합니다. 처음에는 작은 write premium을 내고, 이후에는 큰 read discount를 받습니다.

**2026년의 세 provider flavor.**

| 제공자 | API 방식 | hit 할인 | write 프리미엄 | 기본 TTL | 최소 캐시 가능량 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Content block에 명시적 `cache_control` marker | Input 90% 할인 | 25% surcharge | 5분(1시간까지 연장 가능) | 1,024 tokens(Sonnet/Opus), 2,048(Haiku) |
| OpenAI | Automatic prefix detection | Input 50% 할인 | 없음 | 최대 1시간(best-effort) | 1,024 tokens |
| Google (Gemini) | 명시적 `CachedContent` API | Storage-billed, read는 일반 요금의 약 25% | Token·hour당 storage fee | 사용자 설정(기본 1시간) | 4,096 tokens(Flash), 32,768(Pro) |

**Invariant.** 세 provider 모두 prefix만 cache합니다. 요청 사이에 token 하나라도 달라지면, 첫 번째로 다른 token 이후는 전부 miss입니다. *Stable* part는 위에, *variable* part는 아래에 두세요.

### 캐시 친화적 레이아웃

```text
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

순서를 어기면, 예를 들어 user message를 system prompt 위에 두거나 dynamic retrieval을 few-shot 사이에 끼우면 cache는 절대 hit하지 않습니다.

### Break-even 계산

Anthropic의 25% write premium은 cached block이 돈을 아끼려면 최소 두 번 읽혀야 한다는 뜻입니다. 1 write + 1 read는 요청당 평균 0.675x cost입니다(32% 절감). 1 write + 10 reads는 0.205x입니다(80% 절감). 경험칙: TTL 안에 최소 3번 재사용될 것으로 예상되는 것은 cache하세요.

## 직접 만들기

### 1단계: explicit marker를 쓰는 Anthropic prompt caching

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` marker는 Anthropic에게 block을 5분 동안 저장하라고 알립니다. 그 기간 안의 재사용은 hit합니다. 만료 뒤 재사용하면 다시 write합니다.

**Response usage field:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

CI에서 두 field를 모두 확인하세요. 여러 요청에 걸쳐 `cache_read_input_tokens`가 0에 머무른다면 cache key가 drift하고 있는 것입니다.

### 2단계: 1시간 확장 TTL

Long-running batch job에서는 5-minute default가 작업 사이에 만료됩니다. `ttl`을 설정하세요.

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1-hour TTL은 write premium이 2배 듭니다(baseline 대비 25%가 아니라 50%). 하지만 prefix를 5번 넘게 재사용하는 batch라면 금방 회수됩니다.

### 3단계: OpenAI automatic caching

OpenAI에서는 설정할 것이 없습니다. 최근 요청과 일치하는 1,024 token 초과 prefix는 자동으로 50% 할인됩니다.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

같은 cache-friendly layout 규칙이 적용됩니다. Anthropic cache는 죽이지 않지만 OpenAI cache를 죽이는 두 가지가 있습니다. `user` field를 바꾸는 것(cache key component로 사용됨)과 tool 순서를 바꾸는 것입니다.

### 4단계: Gemini explicit context caching

Gemini는 cache를 직접 만들고 이름 붙이는 first-class object로 취급합니다.

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini는 cache가 살아 있는 동안 token·hour 단위로 storage cost를 청구하고, read는 일반 input rate의 약 25%입니다. 며칠 동안 여러 session에서 같은 거대한 prompt를 재사용할 때 맞는 형태입니다.

### 5단계: production hit rate 측정

Write/read/miss count를 추적하고 1K request당 blended cost를 계산하는 simulated three-provider accountant는 `code/main.py`를 보세요. Target hit rate로 deploy를 gate하세요. 대부분의 production Anthropic setup은 warmup 뒤 read fraction이 80%를 넘어야 합니다.

## 2026년에도 여전히 출시되는 함정

- **상단의 dynamic timestamp.** System prompt 맨 위의 `"Current time: 2026-04-22 15:30:02"`. 모든 요청이 miss합니다. Timestamp를 cache breakpoint 아래로 옮기세요.
- **Tool reordering.** Tool을 stable order로 serialize하세요. Deploy 사이의 dict reshuffle 하나가 모든 hit를 깨뜨립니다.
- **Free-text near-duplicate.** "You are helpful."와 "You are a helpful assistant."의 차이처럼 1 byte만 달라도 full miss입니다.
- **너무 작은 block.** Anthropic은 1,024-token floor를 강제합니다(Haiku는 2,048). 더 작은 block은 조용히 cache되지 않습니다.
- **Blind cost dashboard.** "Input tokens"를 cached와 uncached로 나누세요. 그렇지 않으면 traffic drop이 cache win처럼 보입니다.

## 가져다 쓰기

2026년 caching stack:

| 상황 | 선택 |
|-----------|------|
| Stable 10k+ system prompt와 many turn을 가진 agent | 5-min TTL을 쓰는 Anthropic `cache_control` |
| 30분 이상 prefix를 재사용하는 batch job | `ttl: "1h"`를 쓰는 Anthropic |
| GPT-5 위의 serverless endpoint, custom infra 없음 | OpenAI automatic(prefix를 stable하고 길게 만들기만 하면 됨) |
| 거대한 code/doc corpus의 multi-day reuse | Gemini explicit `CachedContent` |
| Cross-provider fallback | 어떤 provider에서도 hit할 수 있도록 cacheable prefix layout을 동일하게 유지 |

User-message layer에는 semantic caching(Phase 11 · 11)을 함께 쓰세요. Prompt caching은 *token-identical* reuse를 처리하고, semantic caching은 *meaning-identical* reuse를 처리합니다.

## 출시하기

`outputs/skill-prompt-caching-planner.md`를 저장하세요:

```markdown
---
name: prompt-caching-planner
description: Cache-friendly prompt layout을 설계하고 알맞은 provider caching mode를 선택합니다.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Prompt(system + tools + few-shot + retrieval + history + user)와 usage profile(requests per hour, 필요한 TTL, provider)이 주어지면 다음을 출력하세요.

1. Layout. Section을 재정렬하고 단일 cache breakpoint를 표시하세요. 어떤 section이 stable이고 어떤 section이 volatile인지 설명하세요.
2. Provider mode. Anthropic cache_control, OpenAI automatic, Gemini CachedContent 중 하나를 선택하세요. TTL과 reuse pattern을 근거로 정당화하세요.
3. Break-even. TTL 안에서 write당 예상 read 수와 no-cache 대비 net cost를 수식으로 보여 주세요.
4. Verification plan. 두 번째 identical request에서 cache_read_input_tokens > 0임을 확인하는 CI assertion과 cached vs uncached token을 나눈 dashboard를 제시하세요.
5. Failure modes. 이 setup에서 cache miss가 날 가능성이 가장 높은 이유 세 가지(dynamic timestamp, tool reorder, near-duplicate text)와 각각을 막는 방법을 나열하세요.

Dynamic field를 breakpoint 위에 두는 cache plan은 출시를 거부하세요. 2x write premium을 회수할 reuse count 없이 1h TTL을 활성화하는 것도 거부하세요.
```

## 연습문제

1. **쉬움.** Claude에 대해 5,000-token system prompt가 있는 10-turn conversation을 준비하세요. `cache_control` 없이 한 번, 적용해서 한 번 실행하세요. 각각의 input-token bill을 보고하세요.
2. **보통.** Prompt template과 request log를 받아 provider별(Anthropic 5m, Anthropic 1h, OpenAI automatic, Gemini explicit) 예상 hit rate와 dollar saving을 계산하는 test harness를 작성하세요.
3. **어려움.** Layout optimizer를 만드세요. Prompt와 `stable=True/False`로 표시된 field 목록을 입력받아, 정보를 잃지 않으면서 cache-friendly한 최대 위치에 단일 cache breakpoint가 오도록 prompt를 다시 작성합니다. 실제 Anthropic endpoint에서 검증하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Prompt caching | "Long prompt를 싸게 만든다" | Matching prefix에 대해 provider-side KV-cache를 재사용하는 것입니다. 반복 input token에 50-90% discount가 적용됩니다. |
| `cache_control` | "Anthropic marker" | "여기까지는 cacheable"이라고 선언하는 content-block attribute입니다. 예: `{"type": "ephemeral"}`. |
| Cache write | "Premium을 내는 것" | Cache를 채우는 첫 요청입니다. Anthropic에서는 input rate의 약 1.25x로 청구되고, OpenAI에서는 무료입니다. |
| Cache read | "Discount" | Prefix가 일치하는 후속 요청입니다. Anthropic 10%, OpenAI 50%, Gemini 약 25%로 청구됩니다. |
| TTL | "얼마나 오래 살아 있는가" | Cache가 warm하게 유지되는 시간(초)입니다. Anthropic 기본 5m(1h까지 연장 가능), OpenAI는 best-effort 최대 1h, Gemini는 사용자 설정입니다. |
| Extended TTL | "1-hour Anthropic cache" | `{"type": "ephemeral", "ttl": "1h"}`입니다. Write premium이 2배지만 batch reuse에는 가치가 있습니다. |
| Prefix match | "내 cache가 miss한 이유" | 시작부터 breakpoint까지 모든 token이 byte-identical일 때만 cache hit합니다. |
| Context caching (Gemini) | "명시적인 것" | Google의 이름 붙은 storage-billed cache object입니다. 큰 corpus의 multi-day reuse에 가장 좋습니다. |

## 더 읽을거리

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, 1h TTL, break-even table입니다.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic prefix matching입니다.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API와 storage pricing입니다.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — latency number가 포함된 original launch post입니다.
- Phase 11 · 05 (Context Engineering) — cache가 맞을 수 있도록 prompt를 자르는 위치입니다.
- Phase 11 · 11 (Caching and Cost) — prompt caching과 user message semantic cache를 함께 쓰는 방법입니다.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — prompt caching이 사용자에게 노출하는 KV-cache memory model입니다. Cached prefix를 다시 읽는 것이 recompute보다 왜 약 10배 저렴한지 설명합니다.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill은 prompt caching이 shortcut하는 phase입니다. Cache hit에서 TTFT가 크게 떨어지고 TPOT는 영향을 받지 않는 이유를 설명합니다.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching은 speculative decoding, Flash Attention, MQA/GQA와 함께 inference cost curve를 꺾는 lever입니다. 나머지 세 가지도 이해하려면 읽어 보세요.
