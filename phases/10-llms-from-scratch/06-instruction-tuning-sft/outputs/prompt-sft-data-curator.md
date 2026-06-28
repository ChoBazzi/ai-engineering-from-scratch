---
name: prompt-sft-data-curator
description: supervised fine-tuning용 instruction dataset 설계와 큐레이션
version: 1.0.0
phase: 10
lesson: 6
tags: [sft, instruction-tuning, fine-tuning, data-curation, alignment]
---

# SFT Data Curator

특정 capability(code generation, math, conversation, safety)를 위한 instruction-tuning dataset을 설계할 때, 이 프레임워크로 data collection을 계획하고 quality criteria를 정의하며 training pipeline을 구조화하세요.

## 입력 요구사항

다음을 제공하세요.
- **Target capability** (예: "Python code generation", "medical Q&A", "multi-turn conversation")
- **Base model** (예: Llama 3 8B, Mistral 7B, Qwen 2.5 72B)
- **Budget** (annotation hours, synthetic generation용 API cost)
- **Format preference** (Alpaca, ShareGPT, ChatML)

## Step 1: Dataset 설계

### 크기 가이드라인

| Quality Level | 필요한 예제 수 | Expected Outcome |
|--------------|----------------|------------------|
| Research prototype | 1,000-5,000 | LIMA-quality: 예제가 전문가 작성이면 더 큰 dataset과 견줄 수 있음 |
| Production v1 | 10,000-50,000 | Stanford Alpaca 수준: 일반 task 전반에서 탄탄한 instruction following |
| Production v2 | 50,000-200,000 | Vicuna/Llama 2 Chat 수준: robust multi-turn, domain coverage |

품질은 항상 양보다 중요합니다. 전문가가 작성한 1,000개 예제(LIMA, 2023년 5월)는 50,000개 이상 예제로 학습한 모델과 맞먹었습니다. 다음을 우선하세요.

1. **Diversity** -- 목표 capability의 전체 범위를 포괄
2. **Accuracy** -- 모든 response는 사실적으로 정확해야 함
3. **Clarity** -- response는 간결하고 잘 구조화되어야 함
4. **Difficulty gradient** -- 쉬움, 중간, 어려움 예제를 포함

### Diversity 체크리스트

general-purpose assistant 기준:
- Open-ended questions (20%)
- Factual Q&A (20%)
- Creative writing (10%)
- Code generation (15%)
- Reasoning and math (15%)
- Summarization (10%)
- 제약이 있는 instruction following (10%)

domain-specific model에는 비율을 조정하세요. coding assistant는 code generation에 60%, code explanation에 20%를 배정할 수 있습니다.

## 2단계: Data Format

### Alpaca 형식(single-turn)

```json
{
  "instruction": "Write a function that reverses a string in Python.",
  "input": "",
  "output": "def reverse_string(s):\n    return s[::-1]"
}
```

사용 시점: single-turn task, 단순 instruction-response pair, 빠른 prototyping.

### ShareGPT 형식(multi-turn)

```json
{
  "conversations": [
    {"from": "system", "value": "You are a Python expert."},
    {"from": "human", "value": "How do I reverse a string?"},
    {"from": "gpt", "value": "Use slicing: s[::-1]"},
    {"from": "human", "value": "What about for a list?"},
    {"from": "gpt", "value": "Same syntax works: my_list[::-1]"}
  ]
}
```

사용 시점: conversational application, multi-turn context가 중요할 때.

### ChatML Format(special token 포함)

```text
<|im_start|>system
You are a Python expert.<|im_end|>
<|im_start|>user
How do I reverse a string?<|im_end|>
<|im_start|>assistant
Use slicing: s[::-1]<|im_end|>
```

사용 시점: ChatML을 native로 사용하는 모델(Qwen, Yi)을 목표로 할 때.

## 3단계: 품질 기준

### 예제별 검사

1. **Response relevance**: response가 instruction에 실제로 답하나요?
2. **Factual accuracy**: 모든 주장이 검증 가능하고 정확한가요?
3. **Completeness**: response가 instruction을 완전히 다루나요?
4. **Conciseness**: 같은 정보를 더 적은 단어로 전달할 수 있나요?
5. **Format consistency**: response가 기대한 style을 따르나요?

### 위험 신호(예제 거부)

- response가 자기모순을 포함함
- response가 refusal 없이 harmful content를 포함함
- response가 fact 또는 citation을 hallucinate함
- instruction이 모호한데 response가 명확히 하지 않음
- response가 instruction을 바꿔 쓴 복사본임

### Dataset 수준 검사

- 단일 source/template에서 온 예제가 5%를 넘지 않음
- response token의 최소 80%가 의미 있음(filler가 아님)
- 평균 response length가 50-200 tokens(너무 짧거나 긴 response 회피)
- system prompt diversity: 최소 10개의 서로 다른 system prompt가 대표됨

## 4단계: Training Configuration

| Parameter | 권장 범위 | Notes |
|-----------|------------------|-------|
| Learning rate | 1e-5 to 5e-5 | 큰 모델일수록 낮게(70B는 1e-5, 7B는 5e-5) |
| Epochs | 1-3 | validation loss를 monitoring하고 증가 조짐이 보이면 중단 |
| Batch size | 32-128 | GPU 제약이 있으면 gradient accumulation으로 scale |
| Warmup | step의 0-5% | pre-training보다 덜 중요 |
| Weight decay | 0.0-0.1 | 짧은 fine-tuning run에서는 optional |
| Loss masking | response token만 | instruction과 system prompt token을 mask |
| Pre-training data mixing | 2-5% | catastrophic forgetting 방지를 위해 raw text mix |

## 5단계: Evaluation Protocol

training 후 다음을 평가하세요.

1. **Instruction following rate**: 모델이 관련 있고 완전한 response를 생성한 test prompt 비율
2. **Forgetting score**: base model과 비교한 held-out general text corpus의 perplexity
3. **Format compliance**: 기대한 chat format을 따르는 response 비율
4. **MT-Bench or AlpacaEval**: instruction-tuned model용 표준 benchmark
5. **Domain-specific eval**: 목표 capability에 대한 custom evaluation

### 경고 신호

- epoch 1 이후 validation loss 증가: overfitting 중입니다. epoch를 줄이거나 data를 늘리세요
- forgetting score가 15% 초과 증가: learning rate가 너무 높거나 epoch가 너무 많습니다
- 모델이 training example을 그대로 재현: 심각한 overfitting입니다. 더 다양한 data가 필요합니다
- 모델이 benign instruction을 거부: safety data에 과도하게 학습되었습니다. dataset을 재균형하세요
