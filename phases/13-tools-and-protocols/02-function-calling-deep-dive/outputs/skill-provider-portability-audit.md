---
name: provider-portability-audit
description: 한 provider의 function-calling integration을 audit해 다른 두 provider로 port할 때 무엇이 깨지는지 확인합니다.
version: 1.0.0
phase: 13
lesson: 02
tags: [function-calling, openai, anthropic, gemini, portability]
---

한 provider(OpenAI, Anthropic 또는 Gemini)의 function-calling integration이 주어지면, 같은 logic을 다른 두 provider에 출하할 때 나타나는 모든 field rename, behavior difference, hard-limit collision을 나열하는 portability audit을 작성하세요.

작성할 것:

1. Declaration diff. integration의 각 tool에 대해 다른 두 provider 각각에 필요한 envelope / field rename / schema translation을 보여 주세요. target provider가 지원하지 않는 JSON Schema construct를 flag하세요(Gemini: OpenAPI 3.0 subset, OpenAI strict: `$ref` 금지, ambiguous `oneOf` 금지).
2. Response diff. 각 provider의 response shape에서 tool call이 어디에 있는지(`tool_calls[]` vs `content[]` block vs `parts[]` entry), 그리고 누가 `arguments`를 parse할 책임이 있는지(OpenAI는 string, Anthropic과 Gemini는 object)를 문서화하세요.
3. `tool_choice` diff. integration의 현재 choice setting(auto / forbid / force / required)을 target provider shape로 map하고, 빠진 mode를 flag하세요.
4. Limit collisions. tool-count(128 / 64 / 64), schema depth(5 / 10 / effectively unbounded), per-argument length cap을 보고하세요. target provider의 limit을 초과하는 integration에는 block severity를 올리세요.
5. Strict-mode mapping. strict-mode semantics가 target에서 보존되는지 말하세요. OpenAI `strict: true`는 Anthropic에 정확한 equivalent가 없습니다. Gemini `responseSchema`는 근사하지만 request level입니다.

강한 거부 조건:
- non-OpenAI target에서 `arguments`가 string이라고 가정하는 모든 integration. 조용히 잘못된 결과를 냅니다.
- router 없이 Anthropic 또는 Gemini로 port할 때 tool count가 64를 초과하는 모든 integration.
- target이 OpenAI strict mode일 때 schema에 `$ref`를 사용하는 모든 integration.

거부 규칙:
- target equivalent가 없는 provider-specific feature에 의존하는 integration(예: OpenAI Responses API stateful turns, Anthropic computer-use blocks)을 port하라고 요청받으면 refuse하고 어떤 feature에 target equivalent가 없는지 설명하세요.
- winner를 고르라고 요청받으면 refuse하세요. 선택은 host의 strict-mode needs, cost profile, parallel-call requirements에 달려 있습니다.

출력: per-tool diff table, limits table, target provider별 최종 "port verdict"(ship / needs-router / blocked-by-feature)가 있는 one-page audit. 가장 leverage가 큰 migration change를 말하는 한 문장으로 끝내세요.
