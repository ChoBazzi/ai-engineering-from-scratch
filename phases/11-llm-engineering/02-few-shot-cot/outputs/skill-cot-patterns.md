---
name: skill-cot-patterns
description: 작업 복잡도, 정확도 요구사항, 비용 제약에 따라 올바른 reasoning technique을 선택하는 의사결정 프레임워크
version: 1.0.0
phase: 11
lesson: 02
tags: [chain-of-thought, few-shot, self-consistency, tree-of-thought, react, reasoning, prompting]
---

# 추론 기법 선택 가이드

LLM이 문제를 reasoning해야 할 때는 프롬프트를 쓰기 전에 technique을 먼저 선택하세요. technique이 reasoning architecture를 결정하고, 프롬프트는 그것을 채웁니다.

## 빠른 의사결정 트리

1. 작업이 단순 사실 조회 또는 single-step classification인가요?
   - 예: **zero-shot**을 사용하세요. CoT는 정확도 이득 없이 비용만 늘립니다.
   - 아니요: 계속하세요.

2. 작업에 multi-step reasoning(수학, 논리, 계획)이 필요한가요?
   - 예: **Chain-of-Thought**를 사용하세요. 3단계로 계속하세요.
   - 아니요: 형식이 중요하면 **few-shot**, 중요하지 않으면 zero-shot을 사용하세요.

3. 단일 reasoning 오류를 허용할 수 있나요?
   - 예: **few-shot CoT**를 사용하세요(single sample, temperature 0.0).
   - 아니요: **self-consistency**를 사용하세요(N=5, temperature 0.7). 4단계로 계속하세요.

4. 문제가 가능한 경로가 많은 search/planning 문제인가요?
   - 예: **Tree-of-Thought**를 사용하세요.
   - 아니요: self-consistency로 충분합니다.

5. 작업에 외부 정보나 계산이 필요한가요?
   - 예: **ReAct**(reasoning + tool calls)를 사용하세요.
   - 아니요: 순수 reasoning technique으로 충분합니다.

## 기법 매트릭스

| 기법 | 정확도 향상 | 비용 배수 | 지연 시간 | 가장 적합한 경우 |
|-----------|--------------|-----------------|---------|----------|
| Zero-shot | Baseline | 1x | ~1s | 단순 작업, factual Q&A |
| Few-shot | +5-15% | 1.2x | ~1s | Format matching, classification |
| Zero-shot CoT | +10-20% | 1.3x | ~1.5s | 빠른 reasoning boost |
| Few-shot CoT | +15-25% | 1.5x | ~2s | 수학, 논리, multi-step |
| Self-Consistency (N=5) | CoT 대비 +2-5% | 5x | ~5s | High-stakes reasoning |
| Self-Consistency (N=10) | N=5 대비 +1-2% | 10x | ~10s | 중요한 의사결정에만 |
| Tree-of-Thought | 작업 의존 | 10-40x | ~30s+ | Search, planning, puzzle |
| ReAct | 작업 의존 | 3-10x | ~5-15s | Knowledge-grounded task |
| Prompt Chaining | 단일 호출 대비 +5-10% | 2-5x | ~5-10s | 복잡한 multi-part task |

## 모델별 가이드

### GPT-4o / GPT-4.1
- 강한 baseline reasoning을 제공합니다. zero-shot CoT만으로 충분한 경우가 많습니다.
- 예제 3개를 쓰는 few-shot CoT는 GSM8K에서 95%에 도달합니다.
- self-consistency의 이득은 작습니다(95%에서 97%). 중요한 작업에만 가치가 있습니다.
- 답 추출을 위한 structured output을 native로 지원합니다.

### Claude 3.5 Sonnet / Claude 3.7 Sonnet
- structured prompt format(XML tag)을 매우 잘 따릅니다.
- XML로 구분한 예제가 있는 few-shot CoT가 가장 잘 작동합니다.
- Extended thinking(Claude 3.7)은 native CoT입니다. 별도로 prompt할 필요가 없습니다.
- Claude의 reasoning은 temperature 0.7에서 잘 다양화되므로 self-consistency가 효과적입니다.

### Llama 3.1/3.3 70B
- few-shot CoT에서 가장 큰 이득을 봅니다(zero-shot 대비 정확도 차이가 큼).
- reasoning task에는 N=5 self-consistency를 권장합니다.
- 상용 모델보다 더 명시적인 format instruction이 필요합니다.
- local inference에서 ToT는 비쌉니다. batch processing에만 고려하세요.

### Gemini 2.5 Pro
- 기본적으로 multi-step reasoning이 강합니다.
- thinking mode가 prompt engineering 없이 built-in CoT를 제공합니다.
- few-shot 예제는 정확도보다 형식 일관성에 더 도움이 됩니다.
- 큰 context window(1M) 덕분에 예제가 많은 few-shot이 실용적입니다.

## 안티 패턴

**단순 작업에 CoT 사용**: "What is 2+2? Let's think step by step"라고 묻는 것은 토큰 낭비입니다. 모델은 reasoning trace 없이도 단순 산술을 맞힙니다. CoT는 3단계 이상일 때 도움이 됩니다.

**temperature 0.0에서 self-consistency 사용**: N개의 sample이 모두 동일해집니다. 다양한 reasoning path를 얻으려면 temperature > 0(0.5-0.8 권장)을 사용해야 합니다.

**모든 것에 ToT 사용**: ToT는 b=branching factor, d=depth일 때 O(b^d) LLM 호출이 필요합니다. b=3, d=3인 tree는 최대 39번 호출이 필요합니다. 더 싼 technique이 실패하는 문제에만 남겨 두세요.

**나쁜 예제로 few-shot 사용**: reasoning error가 있는 예제는 모델에게 그 오류를 가르칩니다. 모든 예제는 검증되어야 합니다. 틀린 예제 하나가 예제가 없는 경우보다 정확도를 더 떨어뜨릴 수 있습니다.

**일관된 형식 없이 답 추출**: self-consistency는 sample 간 답을 비교해야 합니다. 답 형식이 "$18", "18 dollars", "eighteen"처럼 달라지면 voting이 실패합니다. 항상 "The answer is [number]."를 강제하세요.

## 비용 최적화

GPT-4o 가격($2.50/1M input, $10/1M output)으로 하루 10,000 query를 처리하는 production system의 경우:

| 기법 | 쿼리당 평균 토큰 | 일일 비용 | 정확도 |
|-----------|-----------------|------------|----------|
| Zero-shot | ~200 | ~$5 | 78% |
| Few-shot CoT | ~600 | ~$15 | 95% |
| Self-Consistency (N=5) | ~3,000 | ~$75 | 97% |
| ToT (b=3, d=2) | ~6,000 | ~$150 | 작업 의존 |

대부분의 애플리케이션에서 비용 최적인 전략은 few-shot CoT로 시작하는 것입니다. confidence가 낮은 query에만 self-consistency를 추가하세요(Build It 섹션의 escalation pattern).

## Prompt Chaining과의 통합

reasoning technique은 prompt chaining과 조합할 수 있습니다.

**Chain Step 1** (Extract): zero-shot, temperature 0.0
**Chain Step 2** (Reason): few-shot CoT, temperature 0.0
**Chain Step 3** (Verify): N=3 self-consistency, temperature 0.7

이 3단계 chain은 단일 CoT 호출의 약 3배 비용이 들지만 extraction error와 reasoning error를 잡고 verification step에서 confidence score를 제공합니다.

## Prompting을 넘어설 때

애플리케이션 코드를 쓰는 시간보다 prompt engineering에 더 많은 시간을 쓰고 있다면 다음을 고려하세요.

1. **Fine-tuning**: labeled example이 500개 이상이고 작업 범위가 좁을 때
2. **DSPy compilation**: 자동 prompt optimization을 원할 때
3. **Agent frameworks**: 작업에 multi-turn tool use가 필요할 때(Phase 14)
4. **RAG**: 모델이 private/current knowledge에 접근해야 할 때(Lessons 06-07)

prompting technique은 기반입니다. 어떤 모델, 어떤 provider에서도 작동하고 training data가 필요 없습니다. 하지만 한계가 있습니다. 다음 수준으로 넘어갈 때를 아는 것은 technique 자체를 숙달하는 것만큼 중요합니다.
