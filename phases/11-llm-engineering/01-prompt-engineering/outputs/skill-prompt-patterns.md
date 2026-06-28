---
name: skill-prompt-patterns
description: 작업 유형, 신뢰성 요구사항, 대상 모델에 따라 올바른 prompt pattern을 선택하는 의사결정 프레임워크
version: 1.0.0
phase: 11
lesson: 01
tags: [prompt-engineering, patterns, llm, temperature, cross-model, few-shot, chain-of-thought]
---

# 프롬프트 패턴 선택 가이드

LLM 기반 기능을 만들 때는 프롬프트를 쓰기 전에 prompt pattern을 먼저 선택하세요. 패턴이 구조를 결정하고, 내용은 그 구조를 채웁니다.

## 패턴 의사결정 매트릭스

| 작업 유형 | 주요 패턴 | 보조 패턴 | Temperature | Few-Shot 필요? |
|-----------|----------------|-------------------|-------------|-----------------|
| 데이터 추출 | Template Fill | Few-Shot | 0.0 | 예(예제 2-3개) |
| 분류 | Few-Shot | Guardrail | 0.0 | 예(예제 3-5개) |
| 요약 | Persona + Template | Audience Adapt | 0.3 | 아니요 |
| 코드 생성 | Persona | Chain-of-Thought | 0.0 | 선택 |
| 창작 글쓰기 | Persona | Critique | 0.7-1.0 | 아니요 |
| 다단계 추론 | Chain-of-Thought | Decomposition | 0.3 | 선택 |
| 질의응답 | Persona + Guardrail | Boundary | 0.3 | 아니요 |
| 프롬프트 생성 | Meta-Prompt | Critique | 0.7 | 예(예제 1-2개) |
| 콘텐츠 모더레이션 | Guardrail + Boundary | Few-Shot | 0.0 | 예(예제 5개 이상) |
| 번역/변환 | Audience Adapt | Few-Shot | 0.3 | 예(예제 2-3개) |

## 각 패턴을 사용할 때

**Persona Pattern**: 모든 프롬프트의 baseline으로 사용하세요. 핵심 질문은 역할을 얼마나 구체적으로 만들지입니다. 일반 작업에는 넓은 역할이면 충분합니다. domain-specific 작업에는 domain, seniority level, context를 역할에 명시해야 합니다.

**Few-Shot Pattern**: 내용보다 출력 형식이 더 중요할 때 사용하세요. 모델이 특정 JSON shape, CSV format, classification label을 만들어야 한다면 지시보다 예제가 더 효과적입니다. 경험칙: 단순한 형식은 예제 2-3개, 복잡하거나 모호한 형식은 5개 이상입니다.

**Chain-of-Thought Pattern**: 수학, 논리, multi-step analysis, 모델이 "풀이 과정"을 보여야 하는 작업에 사용하세요. reasoning task에서 정확도를 10-40% 높입니다(Wei et al., 2022). 단순 사실 조회나 extraction에는 사용하지 마세요. 토큰을 낭비합니다.

**Template Fill Pattern**: 모든 출력이 같은 shape를 가져야 하는 structured extraction에 사용하세요. temperature=0.0, 누락 필드에 대한 명시적인 "N/A" 처리와 함께 가장 잘 작동합니다.

**Critique Pattern**: 속도보다 품질이 중요할 때 사용하세요. 모델이 생성하고, 비평하고, 개선합니다. 토큰 비용은 대략 두 배가 되지만 정확도와 완성도가 크게 좋아집니다. high-stakes 출력(보고서, 추천, public-facing content)에 가장 적합합니다.

**Guardrail Pattern**: 모든 user-facing system에 사용하세요. 항상 scope boundary, 범위 밖 요청에 대한 refusal behavior, 명시적인 "I don't know" 처리를 포함하세요. 애플리케이션 측 input validation과 함께 사용하세요.

**Meta-Prompt Pattern**: 새 작업용 프롬프트를 생성할 때 사용하세요. 처음부터 프롬프트를 쓰는 대신 작업을 설명하고 모델이 프롬프트를 쓰게 하세요. 그런 다음 테스트하고 반복하세요. 초기 prompt development 시간을 줄여 줍니다.

**Decomposition Pattern**: divide-and-conquer가 도움이 되는 복잡한 문제에 사용하세요. 모델이 문제를 부분으로 나누고, 각각을 풀고, 결합합니다. 3-7개의 sub-problem이 있는 작업에 가장 효과적입니다.

**Audience Adaptation Pattern**: 같은 콘텐츠가 서로 다른 독자를 대상으로 해야 할 때 사용하세요. 대상 독자를 명시하세요. 모델이 context에서 추측하리라 기대하지 마세요.

**Boundary Pattern**: 특정 유형의 질문에는 절대 답하면 안 되는 production system에 사용하세요. 정확한 refusal message가 있는 hard scope를 정의하므로 guardrail보다 강합니다. compliance-sensitive domain에 필수입니다.

## Cross-Model 호환성

GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, Llama 3 전반에서 얼마나 일관되게 작동하는지 기준으로 패턴을 순위화했습니다.

| 패턴 | 모델 간 일관성 | 참고 |
|---------|------------------------|-------|
| Few-Shot | 매우 높음 | 예제는 모든 모델에 잘 전이됩니다 |
| Template Fill | 매우 높음 | 명시적인 구조는 차이가 생길 여지를 줄입니다 |
| Chain-of-Thought | 높음 | 모든 주요 모델이 "think step by step"을 지원합니다 |
| Persona | 높음 | 어디서나 작동하지만 모델마다 역할 구체성 수준에 대한 반응이 다릅니다 |
| Guardrail | 보통 | Claude가 guardrail을 가장 엄격히 따르며, GPT-4o는 긴 대화에서 가끔 drift합니다 |
| Critique | 보통 | self-critique 품질은 모델별로 크게 다릅니다 |
| Meta-Prompt | 보통 | GPT-4o와 Claude는 서로 다른 prompt style을 만듭니다 |
| Boundary | 낮음-보통 | refusal behavior가 다르므로 모델별로 테스트하세요 |

## 흔한 실수

1. **Chain-of-Thought를 모든 곳에 사용**: CoT는 토큰과 latency를 늘립니다. reasoning step이 필요할 때만 사용하세요.
2. **너무 많은 제약**: 제약이 5-7개를 넘으면 모델이 일부를 누락하기 시작합니다. 가장 중요한 3개를 우선하세요.
3. **모순되는 persona + constraint**: "You are a creative writer" + "Never use metaphors"는 모델을 혼란스럽게 합니다.
4. **temperature 지정 없음**: deterministic output이 필요한데 temperature를 default(보통 1.0)로 두는 문제입니다.
5. **모델 간 prompt copy-paste**: 항상 테스트하세요. GPT-4o에 맞춘 프롬프트가 Claude에서는 성능이 떨어질 수 있고 반대도 마찬가지입니다.
6. **system message 무시**: 지속 규칙을 system message에 두지 않고 모든 것을 user message에 넣는 문제입니다.
7. **부정 제약 과신**: "Do NOT do X, Y, Z, A, B, C"는 "ONLY do W"보다 덜 효과적입니다. 긍정적 framing은 모델에 명확한 target을 줍니다.

## 신뢰성 목표

| 사용 사례 | 패턴 조합 | 예상 정확도 | 토큰 비용 |
|----------|-------------------|-------------------|------------|
| 프로덕션 추출 | Template + Few-Shot | 95%+ | 낮음(500-1K) |
| 사용자 대상 Q&A | Persona + Guardrail + Boundary | 90%+ | 중간(1-2K) |
| 코드 생성 | Persona + Chain-of-Thought | 85%+ | 중간(1-3K) |
| 콘텐츠 생성 | Persona + Critique | 품질 90%+ | 높음(2-4K, double pass) |
| 분류 | Few-Shot + Guardrail | 95%+ | 낮음(300-800) |
| 복잡한 분석 | Decomposition + Chain-of-Thought | 85%+ | 높음(3-5K) |
