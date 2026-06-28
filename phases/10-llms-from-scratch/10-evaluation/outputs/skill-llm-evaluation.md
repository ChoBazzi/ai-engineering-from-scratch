---
name: skill-llm-evaluation
description: task type, budget, requirement에 따라 올바른 LLM evaluation strategy를 고르는 decision framework입니다.
version: 1.0.0
phase: 10
lesson: 10
tags: [evaluation, evals, benchmarks, llm-as-judge, elo, metrics]
---

# LLM Evaluation Strategy

LLM system을 평가할 때 이 decision framework를 적용해 올바른 접근법을 고르세요.

## eval type별 사용 시점

**Benchmarks(MMLU, HumanEval, SWE-bench):** 초기 model selection을 할 때 사용합니다. 후보 model 10개를 3개로 좁혀야 할 때 benchmark는 비용 없이 거친 ranking을 줍니다. 최종 평가로 benchmark를 쓰지 마세요.

**Custom evals:** production을 만들 때 사용합니다. 특정 failure mode가 있는 특정 task가 있습니다. custom eval만이 real-world performance를 예측합니다. prototype은 최소 50 test case, production은 200개 이상이 필요합니다.

**LLM-as-judge:** task가 open-ended(summarization, writing, conversation)일 때 사용합니다. exact match와 token overlap metric은 너무 경직되어 있습니다. LLM-as-judge는 judgment당 약 $0.01이고 human과 약 80% 일치합니다. 모호한 prompt가 아니라 항상 rubric을 사용하세요.

**Human evals:** stakes가 높고 automated metric들이 서로 동의하지 않을 때 사용합니다. human eval은 ground truth이지만 judgment당 $0.10-$2.00이 듭니다. 모호한 case와 automated metric의 주기적 calibration에 남겨 두세요.

**ELO from pairwise comparisons:** 같은 task에서 여러 model을 비교할 때 사용합니다. 사람과 LLM judge는 absolute scoring보다 relative judgment를 더 잘하므로 pairwise가 더 신뢰할 만합니다.

## Scoring function 선택

- **Exact match**: classification, entity extraction, known answer가 있는 structured output
- **Token F1**: partial credit이 중요한 extraction task
- **ROUGE-L**: summarization, translation
- **BLEU**: machine translation
- **LLM-as-judge**: open-ended generation, conversational quality, helpfulness
- **Execution-based**: code generation(code를 실행하고 test 통과 여부 확인)
- **Schema compliance**: structured output(JSON이 schema와 맞는가?)

## eval design의 red flag

- Eval set이 50 case보다 작음: 결과가 통계적으로 의미 없습니다.
- Edge case 없음: happy-path performance만 측정하며, 이는 항상 real-world보다 높습니다.
- 단일 metric: metric마다 다른 이야기를 하므로 최소 두 개를 쓰세요.
- Versioning 없음: versioned eval set 없이는 개선을 추적할 수 없습니다.
- Eval set contamination: eval example을 fine-tuning data나 few-shot prompt에 절대 넣지 마세요.
- 한 model만 테스트: 비교를 위한 baseline(간단한 heuristic이라도)이 필요합니다.

## Eval pipeline 체크리스트

1. task를 정확히 정의하세요("answer questions"가 아니라 "support ticket을 5개 category로 classify").
2. happy path, edge case, known regression을 아우르는 test case를 만드세요.
3. task type에 맞는 scoring function 2-3개를 고르세요.
4. production requirement에 근거해 pass/fail threshold를 정하세요.
5. 실행을 자동화하세요. 하나의 command로 full suite가 돌아가야 합니다.
6. test case, scoring function, prompt, model version을 모두 versioning하세요.
7. prompt update, model swap, code deployment마다 실행하세요.
8. trend를 추적하세요. 단일 score는 noise이고 trendline은 signal입니다.
9. 분기마다 human judgment에 맞춰 calibration하세요.
10. production failure가 발견될 때마다 regression case를 추가하세요.
