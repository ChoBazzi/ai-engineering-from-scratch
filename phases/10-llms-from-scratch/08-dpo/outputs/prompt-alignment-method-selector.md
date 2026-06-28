---
name: prompt-alignment-method-selector
description: use case에 맞는 alignment method(SFT, RLHF, DPO, KTO, ORPO, SimPO)를 고릅니다.
version: 1.0.0
phase: 10
lesson: 8
tags: [alignment, dpo, rlhf, kto, orpo, simpo, preference-optimization, fine-tuning]
---

# Alignment Method 선택기

language model의 alignment method를 고를 때 이 framework로 data, compute, quality requirement를 평가하고 constraint에 가장 잘 맞는 method를 선택하세요.

## 입력 요구사항

다음을 제공하세요.

- **Base model**(예: Llama 3 8B, Mistral 7B, Qwen 2.5 72B)
- **Starting point**(base model인지, 이미 SFT되었는지)
- **Available data**(instruction pair, preference pair, unpaired rating, 또는 없음)
- **Compute budget**(GPU hour, GPU 수)
- **Quality target**(prototype에 충분, open-source와 경쟁, state-of-the-art)
- **Timeline**(days, weeks, months)

## 결정 매트릭스

### 빠른 선택

| 상황 | 추천 방법 | 이유 |
|---------------|-------------------|-----|
| preference data가 없고 instruction pair만 있음 | SFT only | preference signal 없이는 align할 수 없습니다 |
| < 5,000 preference pair, 제한된 compute | DPO | 더 단순한 pipeline이며 small data에서도 잘 동작합니다 |
| unpaired feedback(thumbs up/down만 있음) | KTO | pairwise comparison 없이 동작하는 유일한 방법입니다 |
| 단일 training run으로 alignment 원함 | ORPO | SFT + alignment를 결합하고 reference model이 없습니다 |
| memory-constrained(reference model이 들어가지 않음) | SimPO | reference model이 필요 없습니다 |
| large-scale, multi-objective alignment | RLHF (PPO) | 별도 reward model이 복잡한 preference를 포착합니다 |
| online data를 쓰는 iterative alignment | RLHF (PPO) | generate, rate, retrain을 loop로 돌릴 수 있습니다 |
| post-RLHF refinement | DPO | RLHF model을 targeted preference로 fine-tune합니다 |

### 상세 비교

| 방법 | Data 요구사항 | 메모리 내 model | Training loop | 안정성 | 적합한 scale |
|--------|-----------------|-----------------|----------------|-----------|------------|
| SFT | Instruction pairs (10K+) | 1 | 1 | High | Any |
| RLHF | Preference pairs (20K+) | 3-4 | 3 | Low | Large (70B+) |
| DPO | Preference pairs (5K+) | 2 | 2 (SFT + DPO) | High | Small-Medium (7B-70B) |
| KTO | Unpaired ratings (5K+) | 2 | 2 (SFT + KTO) | High | Any |
| ORPO | Preference pairs (10K+) | 1 | 1 | High | Small-Medium |
| SimPO | Preference pairs (5K+) | 1 | 2 (SFT + SimPO) | High | Small-Medium |

## 방법별 설정

### SFT

- **When to stop**: 1-3 epoch 뒤 또는 validation loss가 더 이상 감소하지 않을 때
- **Key hyperparameter**: learning rate(1e-5 to 5e-5, 큰 model일수록 낮게)
- **Critical detail**: loss에서 instruction token을 mask하세요.
- **Gotcha**: 3 epoch를 넘으면 memorization이 생깁니다. pre-training data를 2-5% 섞으세요.

### RLHF (PPO)

- **When to use**: 20K+ comparison pair가 있고 multi-objective alignment가 필요하거나 iterative online learning을 원할 때
- **Key hyperparameters**: KL coefficient(0.01-0.05), PPO clip ratio(0.1-0.3), learning rate(5e-6 to 3e-5)
- **Critical detail**: reward model은 policy model size 이상이어야 합니다.
- **Gotcha**: PPO는 불안정합니다. KL divergence와 reward curve를 계속 monitoring하세요.

### DPO

- **When to use**: preference pair가 있고 RLHF보다 단순한 pipeline을 원할 때
- **Key hyperparameter**: Beta(0.1-0.5; 낮을수록 reference에서 더 벗어날 수 있음)
- **Critical detail**: reference model은 SFT checkpoint의 frozen copy여야 합니다.
- **Gotcha**: beta에 매우 민감합니다. [0.05, 0.1, 0.2, 0.5] sweep을 실행하세요.

### KTO

- **When to use**: pairwise comparison 없이 "good" 또는 "bad" label만 있을 때
- **Key hyperparameter**: Beta(DPO와 동일), loss aversion multiplier(bad response에 1.5x)
- **Critical detail**: good/bad example이 대략 균형(40-60% split)을 이뤄야 합니다.
- **Gotcha**: pair가 없으므로 gradient signal이 약합니다. DPO보다 더 많은 data가 필요할 수 있습니다.

### ORPO

- **When to use**: SFT를 생략하고 base에서 aligned model로 바로 가고 싶을 때
- **Key hyperparameter**: Lambda(preference term과 SFT term의 weight)
- **Critical detail**: 하나의 dataset 안에 instruction label과 preference pair가 모두 필요합니다.
- **Gotcha**: combined objective는 balance가 어렵습니다. SFT loss가 지배하면 alignment가 약합니다.

### SimPO

- **When to use**: reference model을 들고 있을 수 없는 memory-constrained setup
- **Key hyperparameter**: Beta, gamma(length normalization exponent)
- **Critical detail**: length normalization은 model이 짧은 response를 선호하지 않게 합니다.
- **Gotcha**: reference model anchor가 없으면 model이 더 멀리 drift할 수 있습니다. 주의 깊게 monitoring하세요.

## Pipeline 템플릿

### 템플릿 1: 빠른 prototype(1-2일)

```text
Base Model -> SFT (1 epoch, 10K examples) -> DPO (3 epochs, 5K pairs)
```

Compute: A100에서 7B model 기준 약 4 GPU-hours
Quality: 탄탄한 instruction following, 기본 preference alignment

### 템플릿 2: Production quality(1-2주)

```text
Base Model -> SFT (2 epochs, 50K examples) -> DPO (5 epochs, 20K pairs) -> Eval -> Iterate
```

Compute: 7B는 약 40 GPU-hours, 70B는 약 200 GPU-hours
Quality: open-source RLHF model과 경쟁 가능

### 템플릿 3: State-of-the-art(1-3개월)

```text
Base Model -> SFT (2 epochs, 100K+ examples) -> RLHF (PPO, 50K+ pairs) -> DPO (targeted refinement) -> Eval -> Iterate
```

Compute: 70B 기준 약 500+ GPU-hours
Quality: frontier model alignment에 접근

### 템플릿 4: Minimal data(1-2일)

```text
Base Model -> SFT (1 epoch, 5K examples) -> KTO (unpaired thumbs up/down from users)
```

Compute: 7B 기준 약 2 GPU-hours
Quality: 최소 data collection overhead로 SFT-only보다 좋음

## 평가 프로토콜

alignment 후 다음 차원에서 평가하세요.

1. **Preference win rate**: 200개 이상의 test prompt에서 aligned model과 SFT model을 human judge로 비교합니다. Target: > 60% win rate.
2. **Benchmark retention**: MMLU, HumanEval, 또는 domain-specific benchmark. SFT baseline에서 > 5% 떨어지면 안 됩니다.
3. **MT-Bench or AlpacaEval**: 표준 alignment quality benchmark입니다. published baseline과 비교하세요.
4. **Safety evaluation**: adversarial prompt, jailbreak, harmful request category에 대해 test합니다.
5. **Response diversity**: 100개 prompt에 대한 response entropy를 측정합니다. 낮은 entropy는 mode collapse입니다.

## 흔한 실패 모드

| 증상 | 원인 | Method별 수정 |
|---------|-------|-------------------|
| Verbose, padded responses | Reward model / implicit reward가 length를 선호 | DPO: beta 증가. RLHF: length penalty 추가. SimPO: gamma 조정. |
| Model agrees with everything | preference data bias로 인한 sycophancy | correct response가 user와 disagree하는 preference pair 추가 |
| Refuses benign requests | safety data에 대한 over-alignment | safety example 비율을 줄이고 benign-refusal pair를 더 추가 |
| Outputs are nearly identical to SFT | beta가 너무 높음(DPO/KTO) 또는 KL coefficient가 너무 높음(PPO) | beta / KL coefficient를 낮추세요. model이 배우지 못하고 있습니다 |
| Training loss oscillates | learning rate가 너무 높거나 data 부족 | lr을 2-3배 낮추고 preference data를 늘리세요 |
