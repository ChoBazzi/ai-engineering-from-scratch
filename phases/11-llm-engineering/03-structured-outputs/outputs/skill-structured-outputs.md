---
name: skill-structured-outputs
description: provider, reliability, complexity에 따라 올바른 structured output 전략을 선택하는 의사결정 프레임워크
version: 1.0.0
phase: 11
lesson: 03
tags: [structured-output, json, schema, constrained-decoding, pydantic, function-calling]
---

# 구조화 출력 전략

structured data가 필요한 LLM 애플리케이션을 만들 때 이 의사결정 프레임워크를 적용하세요.

## 각 접근법을 사용할 때

**Prompt-based ("Return JSON"):** prototyping 전용입니다. 가끔 parse failure가 발생해도 괜찮은 internal tool에서는 허용됩니다. retry가 있는 try/except를 추가하세요. production pipeline에서는 절대 사용하지 마세요.

**JSON mode (API flag):** 유효한 JSON 보장은 필요하지만 schema가 단순하거나 유연할 때 사용합니다. 애플리케이션 측에서 shape를 검증할 때 잘 작동합니다. 사용 가능: OpenAI, Anthropic(tool use 경유), Google.

**Schema mode (constrained decoding):** 모든 출력이 특정 schema와 일치해야 하는 production system에 사용합니다. parse failure 0, schema violation 0을 목표로 합니다. production extraction 또는 classification task에서는 기본값으로 사용하세요. 사용 가능: OpenAI structured outputs, Outlines, Guidance.

**Function calling / tool use:** 모델이 parameter만 채우는 것이 아니라 어떤 function을 호출할지 선택해야 할 때 사용합니다. 여러 schema가 있고 모델이 적절한 것을 선택합니다. 기존 tool/function infrastructure와 통합할 때도 사용하세요.

**Instructor library:** provider에 상관없이 automatic retry가 있는 Pydantic validation을 원할 때 사용합니다. Python project에서 DX가 가장 좋습니다. OpenAI, Anthropic, Google, open-source model을 감쌉니다.

## 제공자별 가이드

**OpenAI:** `json_schema` type의 `response_format`을 사용하세요. constrained decoding이 내장되어 있습니다. Pydantic model이 직접 작동합니다. 가장 신뢰할 수 있는 structured output 구현입니다.

**Anthropic:** structured output에는 tool use를 사용하세요. 원하는 schema를 가진 단일 tool을 정의합니다. 모델은 schema와 일치하는 tool call argument를 반환합니다. 신뢰할 수 있지만 tool use API pattern이 필요합니다.

**Open-source models (vLLM, Ollama):** constrained decoding에는 Outlines 또는 Guidance를 사용하세요. 이 library들은 JSON Schema를 generation 중 invalid token을 mask하는 finite state machine으로 compile합니다. local inference 실행이 필요합니다.

## 스키마 설계 가이드라인

1. 가능하면 schema를 flat하게 유지하세요. 2 level을 넘는 nested object는 extraction error를 늘립니다.
2. categorical field에는 enum을 사용하세요. 모델이 올바른 string을 만들어내리라 기대하지 마세요.
3. 모호한 field는 optional로 두기보다 명시적인 null 지원이 있는 required로 만드세요. 모델이 결정을 내리게 합니다.
4. schema property에 description을 추가하세요. 모델은 이를 instruction으로 읽습니다.
5. 필요하지 않다면 union type(oneOf/anyOf)을 피하세요. decoding complexity를 높입니다.
6. number에는 minimum/maximum을 설정하세요. hallucinated extreme value를 잡아냅니다.
7. array에는 minItems/maxItems를 사용해 비어 있거나 무제한인 출력을 방지하세요.

## 흔한 실패 패턴과 수정법

- **모델이 JSON을 markdown fence로 감쌈**: prompt-based에서 JSON mode 또는 schema mode로 전환하세요.
- **Schema-valid하지만 사실적으로 틀림**: extraction 후 LLM-as-judge validation step을 추가하세요.
- **일관되지 않은 enum 값**: constrained decoding으로 전환하거나 post-processing normalization을 추가하세요.
- **optional field 누락**: required로 만들거나 application code에 default value를 추가하세요.
- **매우 느린 extraction**: constrained decoding은 latency를 5-15% 추가합니다. latency-sensitive하다면 schema complexity를 줄이세요.
- **다양한 item이 있는 큰 array**: 입력을 chunk하고 chunk별로 추출한 뒤 결과를 merge하세요.

## 신뢰성 사다리

| 접근법 | 파싱 성공률 | 스키마 일치 | 설정 비용 |
|----------|-------------|-------------|-------------|
| Prompt-based | ~90% | ~80% | 1분 |
| JSON mode | 100% | ~90% | 5분 |
| Schema mode | 100% | ~99% | 15분 |
| Constrained decoding | 100% | 100% | 30분 |
| Instructor + retry | 100% | ~99.5% | 10분 |
