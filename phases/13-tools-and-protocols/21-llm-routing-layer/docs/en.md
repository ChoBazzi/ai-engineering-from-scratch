# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Provider lock-in은 비용이 큽니다. tool-calling workload마다 적합한 model이 다릅니다. routing gateway는 하나의 API surface, retry, failover, cost tracking, guardrail을 제공합니다. 2026년에는 세 archetype이 지배적입니다. LiteLLM(open-source self-hosted), OpenRouter(managed SaaS), Portkey(production-grade, 2026년 3월 open-sourced). 이 lesson은 decision criteria를 이름 붙이고 stdlib routing gateway를 따라갑니다.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## 학습 목표

- self-hosted, managed, production-grade routing option을 구분합니다.
- provider failure에서 정해진 priority order로 retry하는 fallback chain을 구현합니다.
- provider 전반의 request별 cost와 token usage를 추적합니다.
- production constraint에 맞춰 LiteLLM, OpenRouter, Portkey 중 선택합니다.

## 문제

provider routing이 중요한 scenario:

1. **Cost.** Claude Sonnet은 Haiku보다 3배 비쌉니다. triage task에는 Haiku로 충분하고 synthesis task에는 Sonnet이 가치 있습니다. request별로 route하세요.

2. **Failover.** OpenAI가 한 시간 동안 장애입니다. 모든 request가 실패합니다. redeploy 없이 Anthropic으로 automatic fallback하고 싶습니다.

3. **Latency.** live chat UI에는 빠른 time-to-first-token이 필요합니다. batch summarizer에는 그렇지 않습니다. latency SLA로 route하세요.

4. **Compliance.** EU user는 EU region에 머물러야 합니다. region으로 route하세요.

5. **Experimentation.** 같은 workload에서 두 model을 A/B 테스트합니다. test bucket으로 route하세요.

이 모든 것을 integration마다 hand-code하는 것은 반복적입니다. routing gateway는 하나의 OpenAI-compatible API를 제공하고 나머지를 처리합니다.

## 개념

### OpenAI-compatible proxy shape

모두 OpenAI-shape를 말합니다. routing gateway는 `/v1/chat/completions`를 노출하고 OpenAI schema를 받은 뒤 내부적으로 Anthropic / Gemini / Cohere / Ollama / 기타 backend로 proxy합니다. client는 신경 쓰지 않습니다.

### Model aliases

`claude-3-5-sonnet-20251022` 대신 code는 `our_smart_model`이라고 말합니다. gateway가 alias를 실제 model에 매핑합니다. Anthropic이 Claude 4를 출시하면 server-side alias만 바꿉니다. code는 건드리지 않습니다.

### Fallback chains

```text
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateway는 이를 config에 정의합니다. fallback cascade가 cost를 폭발시키지 않도록 retry는 budget에 계산됩니다.

### Semantic caching

동일하거나 거의 동일한 prompt는 provider 대신 cache를 hit합니다. 반복 agent loop에서 30-60% 비용 절감이 가능합니다. key는 embedding-based입니다. 거의 동일한 prompt가 cache slot을 공유합니다.

### Guardrails

Gateway-level:

- **PII redaction.** prompt를 보내기 전 regex 또는 ML-based pass.
- **Policy violations.** 금지된 content가 있는 prompt 거절.
- **Output filters.** completion에서 leak scrub.

Portkey와 Kong은 opinionated guardrail을 제공합니다. LiteLLM은 이를 optional로 둡니다.

### Per-key rate limits

API key 하나는 team 하나입니다. per-key budget은 한 team이 shared quota를 다 쓰는 것을 막습니다. 대부분 gateway가 지원합니다.

### Self-hosted vs managed trade-offs

| 기준 | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| 코드 | open source, Python | managed SaaS | open source(2026년 3월) + managed |
| 설정 | proxy 배포 | 가입 | 둘 다 가능 |
| Providers | 100+ | 300+ | 100+ |
| Billing | 자체 key | OpenRouter credit | 자체 key |
| Observability | OpenTelemetry | dashboard | full OTel + PII redaction |
| 적합한 경우 | full control을 원하는 team | 빠른 prototyping | compliance가 필요한 production |

SRE team이 있고 data sovereignty를 원하면 LiteLLM이 이깁니다. single subscription과 no infra를 원하면 OpenRouter가 이깁니다. guardrail과 compliance가 out of the box로 필요하면 Portkey가 이깁니다.

### Cost tracking

모든 request는 `provider`, `model`, `input_tokens`, `output_tokens`를 갖습니다. gateway가 유지하는 pricing sheet에서 model별 token price를 가져와 곱합니다. user별 / team별 / project별로 aggregate합니다.

### MCP plus routing

gateway는 LLM call과 MCP sampling request를 모두 route할 수 있습니다. sampling request의 modelPreferences가 특정 model을 선호하면 gateway가 올바른 backend로 translate합니다. Phase 13 · 17(MCP gateway)과 이 lesson의 routing gateway가 때때로 하나의 service로 합쳐지는 지점입니다.

### Routing strategies

- **Static priority.** list의 첫 번째를 쓰고 error 시 fall back.
- **Load balancing.** round-robin 또는 weighted.
- **Cost-aware.** latency / quality를 만족하는 가장 싼 model 선택.
- **Latency-aware.** 최근 N분에서 가장 빠른 model 선택.
- **Task-aware.** prompt classifier가 coding은 한 model로, summarization은 다른 model로 route.

## 사용하기

`code/main.py`는 약 150줄의 routing gateway를 구현합니다. OpenAI-shaped request를 받고 provider별 stub으로 translate하며, priority fallback chain을 실행하고, request별 cost를 추적하고, input에 PII redaction pass를 적용합니다. 세 scenario로 실행하세요. normal request, primary-provider outage가 fallback을 trigger하는 경우, PII leak이 redaction에 걸리는 경우입니다.

볼 부분:

- `ROUTES` dict: alias -> concrete provider의 priority-ordered list.
- Fallback loop는 5xx에서 retry합니다.
- Cost tracker는 token usage에 model별 rate를 곱합니다.
- PII redactor는 forwarding 전에 SSN-shaped pattern을 scrub합니다.

## 산출물

이 lesson은 `outputs/skill-routing-config-designer.md`를 만듭니다. workload profile(latency, cost, compliance)이 주어지면 이 skill은 LiteLLM / OpenRouter / Portkey를 선택하고 routing config를 만듭니다.

## 연습 문제

1. `code/main.py`를 실행하세요. outage scenario를 trigger하고 fallback이 두 번째 provider에 도달하며 cost가 올바르게 attribution되는지 확인하세요.

2. semantic caching을 추가하세요. prompt의 SHA256을 lookup key로 삼고, cache hit는 즉시 반환하세요. 반복 call에서 cost saving을 측정하세요.

3. "code ..." prompt는 intelligence를 선호하는 alias로, "summarize ..." prompt는 speed를 선호하는 alias로 route하는 prompt classifier를 추가하세요.

4. team별 budget을 설계하세요. 각 team에는 monthly spend cap이 있고 cap에 도달하면 gateway가 request를 거절합니다. enforcement granularity(per-request 또는 windowed)를 고르세요.

5. LiteLLM, OpenRouter, Portkey docs를 나란히 읽으세요. 각 제품이 나머지 둘에는 없는 feature 하나씩을 이름 붙이세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Routing gateway | "LLM proxy" | 여러 provider 앞에 있는 one-API-surface layer |
| OpenAI-compatible | "Speaks the OpenAI schema" | `/v1/chat/completions` shape를 받아 backend로 translate |
| Model alias | "our_smart_model" | code 안의 이름을 gateway가 concrete model로 매핑 |
| Fallback chain | "Retry list" | failure 시 시도할 provider의 ordered list |
| Semantic caching | "Prompt-embedding cache" | key가 prompt embedding이며 near-duplicate가 cache hit를 공유 |
| Guardrails | "Input/output filters" | PII redact, policy violation reject |
| Per-key rate limit | "Team budget" | API key에 scoped된 quota |
| Cost tracking | "Per-request spend" | token usage x model별 price를 aggregate |
| LiteLLM | "The open proxy" | self-hostable OSS routing gateway |
| OpenRouter | "The managed SaaS" | credit-based billing이 있는 hosted gateway |
| Portkey | "The production option" | built-in guardrail이 있는 open-source + managed gateway |

## 더 읽을거리

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — guardrail이 있는 production routing
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — decision guide
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
