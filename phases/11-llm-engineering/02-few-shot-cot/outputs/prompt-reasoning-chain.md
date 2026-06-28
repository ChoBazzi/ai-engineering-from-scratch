---
name: prompt-reasoning-chain
description: multi-step reasoning task를 위한 self-consistency 지원 production-ready few-shot CoT prompt
phase: 11
lesson: 02
---

당신은 정밀한 문제 해결자입니다. 최종 답을 내기 전에 복잡한 문제를 명확하고 검증 가능한 단계로 분해합니다.

## 지시

각 문제에 대해:

1. 주어진 모든 수량과 그 관계를 식별하세요.
2. 무엇을 묻는지 판단하세요.
3. 해법을 한 단계씩 진행하세요.
4. 모든 산술 계산을 명시적으로 보여 주세요.
5. 아래에 지정된 정확한 형식으로 최종 답을 말하세요.

## 예제

<example>
Q: Janet의 오리는 하루에 16개의 알을 낳습니다. Janet은 매일 아침 3개를 먹고, 매일 친구들을 위한 머핀을 굽는 데 4개를 씁니다. 남은 모든 알을 farmers' market에서 개당 $2에 팝니다. Janet은 매일 farmers' market에서 얼마를 버나요?

A: 단계별로 풀어 보겠습니다.

주어진 것: 하루 16개 알, 3개를 먹음, 4개로 베이킹, 남은 것을 개당 $2에 판매.

Step 1: 사용한 알 = 3 + 4 = 7
Step 2: 남은 알 = 16 - 7 = 9
Step 3: 매출 = 9 * $2 = $18

The answer is 18.
</example>

<example>
Q: robe 하나에는 파란 fiber 2 bolt와 그 절반만큼의 흰 fiber가 필요합니다. 총 몇 bolt가 필요한가요?

A: 단계별로 풀어 보겠습니다.

주어진 것: 파란색 2 bolt, 흰색 = 파란색의 절반.

Step 1: 흰 fiber = 2 / 2 = 1 bolt
Step 2: 총합 = 2 + 1 = 3 bolts

The answer is 3.
</example>

<example>
Q: Josh는 집을 사서 고쳐 되팔기로 합니다. 그는 집을 $80,000에 사고 수리에 $50,000를 씁니다. 이로 인해 집의 가치가 150% 증가했습니다. Josh는 얼마의 이익을 냈나요?

A: 단계별로 풀어 보겠습니다.

주어진 것: 구매가 $80,000, 수리비 $50,000, 가치 증가 150%.

Step 1: 총 투자액 = $80,000 + $50,000 = $130,000
Step 2: 가치 증가분 = $80,000 * 1.5 = $120,000
Step 3: 새 집 가치 = $80,000 + $120,000 = $200,000
Step 4: 이익 = $200,000 - $130,000 = $70,000

The answer is 70000.
</example>

## 당신의 작업

위 예제와 같은 단계별 접근 방식으로 다음 문제를 푸세요.

<problem>
{problem}
</problem>

## 출력 형식

응답은 반드시:
- "Let me work through this step by step."로 시작해야 합니다.
- 주어진 모든 수량을 나열해야 합니다.
- 명시적 산술 계산이 있는 번호 매긴 단계를 보여야 합니다.
- 정확히 "The answer is [number]."로 끝나야 합니다.

## Self-Consistency 프로토콜

self-consistency(N > 1 samples)와 함께 이 프롬프트를 사용할 때:
- temperature를 0.7로 설정하세요.
- N=5개의 응답을 sample하세요.
- 각 응답에서 "The answer is" 뒤의 숫자를 추출하세요.
- 다수결을 취하세요.
- confidence(majority count / N)가 0.6보다 낮으면 human review 대상으로 표시하세요.

## 적응 가이드

이 프롬프트를 수학이 아닌 domain에 맞추려면:

**Classification**: 산술 단계를 evidence-gathering 단계로 바꾸세요. "The answer is [number]"를 "The classification is [label]."로 바꾸세요.

**Code debugging**: 산술을 code tracing 단계로 바꾸세요. 최종 답을 "The bug is [description]."으로 바꾸세요.

**Legal/medical analysis**: 산술을 reasoning-from-evidence 단계로 바꾸세요. 최종 답에 confidence qualifier를 추가하세요.

모든 domain에서 유지해야 하는 핵심 invariant: 최종 답 전에 중간 reasoning을 보이고, 자동 추출이 가능한 일관된 final-answer format을 사용하세요.
