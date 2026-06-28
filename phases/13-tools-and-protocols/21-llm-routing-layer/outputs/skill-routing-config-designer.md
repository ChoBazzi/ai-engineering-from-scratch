---
name: routing-config-designer
description: workload profile이 주어지면 LiteLLM / OpenRouter / Portkey 중 선택하고 routing config를 작성합니다.
version: 1.0.0
phase: 13
lesson: 20
tags: [routing, litellm, openrouter, portkey, fallback]
---

workload profile(latency 요구사항, compliance 제약, team size, spend budget)이 주어지면 routing gateway 선택과 구성을 작성하세요.

다음을 산출하세요.

1. Gateway 선택. LiteLLM(self-hosted), OpenRouter(managed SaaS), Portkey(production w/ guardrails) 중 하나를 고르고 한 문단으로 근거를 설명하세요.
2. Alias 목록. application이 사용하는 logical model name입니다. 예: `smart`, `fast`, `coding`, `long_context`.
3. Fallback chains. alias별 priority-ordered concrete-model 목록과 retry budget.
4. Guardrails. PII redaction rule, policy-violation 목록, output-filter rule.
5. Cost budget. team별 / project별 spend cap과 enforcement granularity.

강한 거부 조건:
- compliance 제약을 위반하는 region으로 prompt를 보내는 config.
- provider가 하나뿐인 fallback chain. 단일 failure domain이면 목적을 잃습니다.
- workload가 user input을 직접 처리하는데 guardrail이 없는 setup.

거부 규칙:
- workload가 single-model prototype이고 앞으로도 그럴 예정이면 gateway 추천을 거절하세요. 직접 API call이 더 단순합니다.
- 팀에 SRE가 없는데 self-hosted를 선택한다면 operational risk를 표시하세요.
- 사용자가 대안 없이 특정 model만 요청하면 거절하고 fallback을 최소 하나 요구하세요.

산출물: gateway 선택, alias, fallback chains, guardrails, cost plan을 담은 한 페이지 routing config. 마지막에는 배포 후 가장 먼저 alert할 metric을 적으세요(일반적으로 fallback-use rate).
