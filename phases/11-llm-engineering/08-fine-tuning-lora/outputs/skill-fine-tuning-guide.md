---
name: skill-fine-tuning-guide
description: LoRA와 QLoRA로 LLM을 언제 어떻게 fine-tune할지 결정하는 의사결정 트리
version: 1.0.0
phase: 11
lesson: 8
tags: [fine-tuning, lora, qlora, peft, llm-engineering]
---

# Fine-Tuning 의사결정 가이드

fine-tuning 전에 다음을 순서대로 시도하세요.

```text
1. Prompt engineering (minutes, $0)
2. Few-shot examples in prompt (minutes, $0)
3. RAG for knowledge retrieval (days, $10-100/month)
4. Fine-tuning with LoRA/QLoRA (days, $5-50 per experiment)
5. Full fine-tuning (weeks, $100-10,000 per run)
```

이전 단계가 측정 가능하게 부족할 때만 다음 단계로 이동하세요.

## fine-tune해야 할 때

- 프롬프팅으로 달성할 수 없는 일관된 출력 스타일이나 형식이 필요할 때
- 더 큰 모델을 distill할 때(8B 모델에서 GPT-4 품질)
- 지연 시간이 중요하고 few-shot 예시가 너무 많은 토큰을 추가할 때
- 모델이 복잡한 reasoning 패턴을 안정적으로 따르도록 해야 할 때
- 원하는 input-output 동작에 대한 고품질 예시가 1,000개 이상 있을 때

## fine-tune하지 말아야 할 때

- 올바른 프롬프트만으로 모델이 이미 원하는 일을 할 때
- 모델이 사실을 알아야 할 때(RAG를 사용하세요)
- 학습 예시가 500개 미만일 때(overfit 가능성이 큼)
- 작업이 자주 바뀔 때(재학습 비용이 큼)
- 특정 출력에 어떤 데이터가 영향을 주었는지 감사해야 할 때(fine-tuning은 black box입니다)

## 방법 선택

| GPU VRAM | 7B 모델 | 13B 모델 | 70B 모델 |
|----------|----------|-----------|-----------|
| 16GB (T4) | QLoRA | 불가능 | 불가능 |
| 24GB (3090/4090) | QLoRA 또는 LoRA | QLoRA | 불가능 |
| 40GB (A100) | LoRA 또는 Full | QLoRA 또는 LoRA | QLoRA |
| 80GB (A100/H100) | Full | LoRA 또는 Full | QLoRA 또는 LoRA |

## LoRA 설정 체크리스트

1. r=16, alpha=32로 시작합니다(대부분 작업의 안전한 기본값)
2. 먼저 q_proj와 v_proj를 target으로 삼습니다(minimum viable LoRA)
3. QLoRA에는 learning rate 2e-4, LoRA fp16에는 5e-5를 사용합니다
4. lora_dropout=0.05를 설정합니다
5. 1-3 epoch 학습합니다. 더 길면 overfitting 위험이 커집니다
6. held-out set에서 100 step마다 평가합니다
7. checkpoint를 저장하고 eval loss 기준으로 최고를 고릅니다

## 흔한 실수

- 너무 많은 epoch 동안 학습함. 작은 데이터셋에서는 epoch 2-3 이후 overfitting됩니다
- full fine-tuning과 같은 learning rate를 사용함. LoRA에는 더 높은 LR이 필요합니다
- pad token 설정을 잊음. Llama 모델에서 NaN loss를 유발합니다
- base model을 freeze하지 않음. LoRA의 목적을 무너뜨립니다
- training data에서만 평가함. 항상 10-20%를 eval용으로 분리하세요
- prompt engineering baseline을 건너뜀. 프롬프팅이 이미 푸는 문제를 fine-tuning하게 됩니다

## 품질 검증

학습 후 200개 이상의 held-out 예시에서 비교하세요.
1. 최적 프롬프트를 사용한 base model(baseline)
2. LoRA adapter를 얹은 base model(당신의 fine-tuned model)
3. 같은 프롬프트를 사용한 GPT-4 또는 Claude(ceiling)

LoRA model이 prompted baseline을 이기지 못한다면 더 많은 compute가 아니라 학습 데이터나 설정을 개선해야 합니다.

## Adapter 관리

- multi-task serving에는 adapter를 분리해 유지합니다(요청마다 adapter 교체)
- single-task deployment에는 adapter를 base weights에 merge합니다
- adapter는 Hugging Face Hub에 저장합니다(10-100MB, version/share가 쉬움)
- 배포 전에 merged model 출력이 unmerged 출력과 일치하는지 테스트합니다
- 여러 adapter를 하나로 결합하려면 TIES-Merging 또는 DARE를 사용합니다

## 학습 디버깅

loss가 감소하지 않는다면:
1. learning rate를 확인합니다. LoRA에 너무 낮을 수 있으니 2e-4를 시도하세요
2. LoRA layer가 실제로 gradient를 받는지 확인합니다
3. base model weights가 frozen인지 확인합니다
4. 데이터 형식을 확인합니다. tokenizer가 모델의 예상 형식과 맞아야 합니다

loss는 감소하지만 eval 품질이 나쁘다면:
1. 학습 데이터 품질 문제입니다(garbage in, garbage out)
2. Overfitting입니다(epoch를 줄이고, dropout을 늘리고, 데이터를 더 추가하세요)
3. target modules가 잘못되었습니다(복잡한 작업에는 MLP layer 추가)
4. rank가 너무 낮습니다(r=32 또는 r=64 시도)
