---
name: prompt-reward-model-designer
description: RLHF alignment를 위한 reward model training pipeline을 설계합니다.
version: 1.0.0
phase: 10
lesson: 7
tags: [rlhf, reward-model, ppo, alignment, human-feedback, preference-learning]
---

# Reward Model 설계기

language model을 목표 행동(helpfulness, coding ability, safety, honesty)에 맞춰 align하는 RLHF pipeline을 만들 때, 이 framework를 사용해 data collection protocol을 설계하고 reward model을 학습하며 PPO를 설정하세요.

## 입력 요구사항

다음을 제공하세요.

- **Target behavior**(예: "helpful and harmless assistant", "expert Python coder", "medical Q&A with safety")
- **Base model**(예: SFT 후 Llama 3 8B, Mistral 7B Chat)
- **Reward model size**(보통 policy model과 같거나 더 큼)
- **Annotation budget**(사용 가능한 human hour 또는 comparison pair)
- **Compute budget**(reward model training + PPO에 쓸 GPU hour)

## Step 1: Preference Data 수집

### Annotation protocol

1. **Prompt selection**: SFT training distribution과 out-of-distribution prompt(10-20% novel)에서 sample합니다.
2. **Response generation**: SFT model로 prompt마다 2-4개 response를 생성하고 temperature를 다르게 둡니다(0.3, 0.7, 1.0).
3. **Comparison format**: annotator에게 정확히 2개 response를 보여 주고 "어느 response가 더 나은가?"를 묻습니다.
4. **Criteria rubric**: use case에서 "better"가 무엇을 뜻하는지 정의합니다.

### Rubric template

| 기준 | 가중치 | 설명 |
|-----------|--------|-------------|
| Helpfulness | 40% | 질문에 완전하고 정확하게 답하는가? |
| Harmlessness | 25% | harmful, biased, misleading content를 피하는가? |
| Honesty | 20% | hallucinate하지 않고 uncertainty를 인정하는가? |
| Conciseness | 15% | 질문에 맞는 적절한 길이인가? |

use case에 맞게 weight를 조정하세요. coding assistant는 correctness를 60%, conciseness를 20%로 둘 수 있습니다.

### Data size guideline

| Scale | Comparison Pairs | Annotator Hours | Expected RM Accuracy |
|-------|-----------------|-----------------|---------------------|
| Minimum viable | 5,000-10,000 | 400-800 | 60-65% |
| Production v1 | 20,000-50,000 | 1,600-4,000 | 65-72% |
| Production v2 | 100,000-500,000 | 8,000-40,000 | 72-78% |

InstructGPT는 40명 contractor가 만든 33,000개 comparison을 사용했습니다. Anthropic의 초기 논문은 20명 annotator가 만든 22,000개를 사용했습니다. inter-annotator agreement는 보통 70-75%입니다. reward model은 사람 간 합의 수준을 넘을 수 없습니다.

### Quality control

- **Agreement filtering**: annotator의 70% 미만만 동의한 pair는 버립니다.
- **Annotator calibration**: 실제 annotation 전에 known-good pair로 calibration round를 실행합니다.
- **Bias detection**: annotator가 긴 response, formal language, 특정 pattern을 계속 선호하는지 monitoring합니다.
- **Adversarial examples**: 꼼꼼히 읽지 않는 annotator를 잡기 위한 example을 5-10% 포함합니다.

## 2단계: Reward Model Architecture

### Architecture decision

| 결정 항목 | 추천 | 근거 |
|----------|---------------|-----------|
| Base architecture | policy와 같은 transformer | SFT checkpoint의 weight initialization이 강한 시작 feature를 줍니다 |
| Output head | last hidden state에서 single linear projection | 가장 완전한 position representation에서 scalar reward를 뽑습니다 |
| Model size | >= policy model size | 작은 RM은 PPO를 불안정하게 만드는 unreliable signal을 냅니다 |
| Initialization | 새 output head를 단 SFT checkpoint | pre-trained feature가 이미 language quality를 포착합니다 |

### Training configuration

| Parameter | 범위 | 참고 |
|-----------|-------|-------|
| Learning rate | 1e-5 to 5e-5 | task가 더 단순하므로 SFT보다 낮게 둡니다 |
| Epochs | 1-3 | 제한된 comparison data에서는 overfitting이 큰 risk입니다 |
| Batch size | 64-256 | 각 "example"이 pair이므로 effective data는 2배입니다 |
| Loss function | Bradley-Terry: -log(sigmoid(r_preferred - r_rejected)) | pairwise comparison의 표준입니다 |
| Validation split | 10-20% | held-out pair의 accuracy를 monitoring합니다 |

### Evaluation metric

1. **Pairwise accuracy**: held-out preference pair 중 RM이 몇 fraction을 올바르게 rank하는가? Target: > 65%
2. **Margin distribution**: (r_preferred - r_rejected)의 distribution을 그립니다. 0보다 위에 중심이 있고 negative가 적어야 합니다.
3. **Calibration**: sigmoid(r_preferred - r_rejected)가 실제 human preference probability에 가까운가?
4. **OOD generalization**: training과 다른 distribution의 prompt로 test합니다. accuracy drop은 < 10%여야 합니다.

## Step 3: PPO 설정

### Hyperparameters

| Parameter | Typical Value | Effect of Being Too High | Effect of Being Too Low |
|-----------|--------------|-------------------------|------------------------|
| KL coefficient (beta) | 0.01-0.05 | 모델이 거의 배우지 못하고 SFT에 너무 가까움 | reward hacking, degenerate output |
| Learning rate | 5e-6 to 3e-5 | training instability, divergence | slow convergence, wasted compute |
| Clip ratio (epsilon) | 0.1-0.3 | 크고 불안정할 수 있는 update | 매우 보수적인 update, 느린 학습 |
| PPO epochs per batch | 1-4 | current batch에 overfit | batch를 충분히 활용하지 못함 |
| Generation batch size | 128-512 | memory issue | noisy gradient estimate |
| Max response length | 256-1024 | 느린 generation, memory issue | useful response를 truncate |

### Monitoring dashboard

PPO training 중 다음 metric을 추적하세요.

1. **Mean reward**: 학습 동안 증가해야 합니다. plateau는 괜찮지만 감소는 instability입니다.
2. **KL divergence**: 10-20 nats 아래에 있어야 합니다. spike는 reward hacking 신호입니다.
3. **Response length**: 안정적으로 유지되어야 합니다. monotonic increase는 verbosity reward hacking입니다.
4. **Entropy**: token distribution entropy는 천천히 감소해야 합니다. 빠른 감소는 mode collapse입니다.
5. **Reward model agreement**: PPO response를 reward model로 score하세요. agreement가 좋아져야 합니다.

### PPO 중 red flag

| 증상 | 가능성 높은 원인 | 수정 |
|---------|-------------|-----|
| Reward는 오르지만 output이 나빠짐 | Reward hacking | KL coefficient를 높이고 adversarial example로 RM을 재학습 |
| KL divergence 폭발 | learning rate가 너무 높거나 KL coefficient가 너무 낮음 | lr을 낮추고 beta를 높임 |
| Response length가 단조 증가 | RM이 verbosity에 reward를 줌 | reward에 length penalty 추가, length-controlled pair로 RM 재학습 |
| 모든 response가 같아짐 | Mode collapse | generation temperature를 높이고 PPO epoch를 줄임 |
| Reward가 심하게 진동 | PPO instability | learning rate를 낮추고 clip ratio를 높임 |

## Step 4: End-to-End 검증

RLHF-trained model을 배포하기 전에:

1. **A/B test vs SFT**: SFT와 RLHF model을 200개 이상의 test prompt에서 실행합니다. evaluator 3명 이상이 response를 비교합니다. RLHF model은 > 60% win rate여야 합니다.
2. **Safety evaluation**: 알려진 adversarial prompt(jailbreak, harmful request)에서 test합니다. RLHF model은 적절히 refuse해야 합니다.
3. **Regression check**: 표준 benchmark(MMLU, HumanEval, MT-Bench)를 실행해 RLHF model이 core capability를 잃지 않았는지 확인합니다.
4. **Forgetting check**: general text corpus에서 perplexity를 측정합니다. SFT model 대비 증가는 < 10%여야 합니다.
5. **Length analysis**: SFT와 RLHF model의 average response length를 비교합니다. RLHF가 50% 넘게 길다면 reward model에 verbosity bias가 있을 가능성이 큽니다.
