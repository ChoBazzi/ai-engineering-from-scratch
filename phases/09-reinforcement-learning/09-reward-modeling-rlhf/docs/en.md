# Reward Modeling과 RLHF

> 사람은 "좋은 assistant 응답"을 위한 reward function을 직접 쓸 수는 없지만, 두 응답을 비교해 더 나은 쪽을 고를 수는 있습니다. 그런 비교 데이터에 reward model을 맞춘 뒤, 그 reward를 기준으로 language model에 RL을 적용합니다. Christiano 2017. InstructGPT 2022. GPT-3를 ChatGPT로 바꾼 레시피입니다. 2026년에는 대부분 DPO로 대체되는 중이지만, mental model은 그대로 남아 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## 문제

당신은 next-token-prediction 목표로 language model을 학습했습니다. 이 모델은 문법적인 영어를 씁니다. 하지만 거짓말도 하고, 장황하게 떠들고, 거절해야 할 때 거절하지 못하기도 합니다. pretraining을 더 한다고 해결되지 않습니다. 웹 텍스트가 문제이지, 치료제가 아니기 때문입니다.

"instruction X에 대해 response A가 response B보다 낫다"라고 말해 주는 *scalar reward*가 필요합니다. 그 reward function을 손으로 쓰는 것은 불가능합니다. "helpfulness"는 token 위의 closed-form expression이 아닙니다. 하지만 사람은 두 출력을 비교해 preference를 표시할 수 있습니다. 그리고 이것은 대규모로 모으기 비교적 저렴합니다.

RLHF(Christiano et al. 2017; Ouyang et al. 2022)는 preference를 reward model로 변환한 뒤, 그 reward를 기준으로 PPO로 LM을 최적화합니다. 세 단계로 말하면 SFT → RM → PPO입니다. 2023-2025년에 ChatGPT, Claude, Gemini, 그리고 거의 모든 aligned-LLM을 출시하게 만든 레시피입니다.

2026년에는 PPO 단계가 대부분 DPO(Phase 10 · 08)로 대체되었습니다. DPO가 더 저렴하고 alignment tuning 품질도 거의 비슷하기 때문입니다. 하지만 *reward model* 부분은 여전히 모든 Best-of-N sampler, 모든 RL-from-verifiable-rewards pipeline, 그리고 process reward model을 쓰는 모든 reasoning model의 기반입니다. RLHF를 이해하면 alignment stack 전체를 이해하게 됩니다.

## 개념

![3단계 RLHF: SFT, pairwise preference로 RM 학습, KL penalty를 둔 PPO](../assets/rlhf.svg)

**1단계: Supervised Fine-Tuning (SFT).** pretrained base model에서 시작합니다. 목표 행동(instruction-following responses, helpful replies 등)에 대한 사람이 작성한 demonstration으로 fine-tune합니다. 결과는 `π_SFT`라는 모델입니다. 이 모델은 *좋은 행동 쪽으로 편향*되어 있지만, 여전히 action space가 무한히 열려 있습니다.

**2단계: Reward Model training.**

- prompt `x`에 대한 response 쌍 `(y_+, y_-)`을 모으고, 사람이 "y_+가 y_-보다 선호된다"라고 label합니다.
- reward model `R_φ(x, y)`가 `y_+`에 더 높은 점수를 주도록 학습합니다.
- Loss는 **Bradley-Terry pairwise logistic**입니다.

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ는 sigmoid입니다. reward 차이는 preference의 log-odds를 암시합니다. BT는 1952년부터 표준(Bradley-Terry)이었고, 현대 RLHF에서 지배적으로 쓰이는 선택입니다.

- `R_φ`는 보통 SFT model에서 초기화하고, 위에 scalar head를 얹습니다. 같은 transformer backbone을 쓰고, 단일 linear layer가 reward를 출력합니다.

**3단계: KL penalty를 포함해 RM을 기준으로 PPO.**

- 학습 가능한 policy `π_θ`를 `π_SFT`에서 초기화합니다. frozen *reference* `π_ref = π_SFT`를 유지합니다.
- response `y` 끝에서의 reward:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty는 `π_θ`가 `π_SFT`에서 임의로 멀어지는 것을 막습니다. 이것은 hard trust region이 아니라 *regularizer*입니다. `β`는 보통 `0.01`-`0.05`입니다.
- 이 reward로 PPO(Lesson 08)를 실행합니다. advantage는 token-level trajectory에서 계산하지만, RM은 전체 response에만 점수를 줍니다.

**왜 KL이 필요한가?** KL이 없으면 PPO는 reward-hacking 전략을 기꺼이 찾아냅니다. RM은 in-distribution completion으로만 학습되었기 때문입니다. out-of-distribution response는 사람이 쓴 어떤 응답보다 높은 점수를 받을 수도 있습니다. KL은 `π_θ`를 RM이 학습된 manifold 근처에 묶어 둡니다. RLHF에서 가장 중요한 단일 knob입니다.

**2026년의 상태:**

- **DPO** (Rafailov 2023): closed-form algebra로 Stage 2+3을 preference data 위의 단일 supervised loss로 접습니다. RM도 PPO도 없습니다. alignment benchmark에서 훨씬 적은 compute로 비슷한 품질을 냅니다. Phase 10 · 08에서 다룹니다.
- **GRPO** (DeepSeek 2024-2025): critic 대신 group-relative baseline을 쓰는 PPO 계열이며, human-trained RM 대신 *verifier*(code runs / math answer matches)에서 reward를 얻습니다. reasoning model에서 지배적입니다. Phase 9 · 12에서 다룹니다.
- **Process reward models (PRMs):** partial solution(각 reasoning step)에 점수를 매깁니다. reasoning을 위한 RLHF와 GRPO 변형 모두에서 쓰입니다.
- **Constitutional AI / RLAIF:** 사람 대신 aligned LLM이 preference를 생성하게 합니다. preference budget을 확장합니다.

## 직접 만들기

이 레슨은 문자열로 표현된 아주 작은 synthetic "prompts"와 "responses"를 사용합니다. RM은 bag-of-tokens 표현 위의 linear scorer입니다. 실제 LLM은 없습니다. 중요한 것은 규모가 아니라 pipeline의 *형태*입니다. `code/main.py`를 보세요.

### 1단계: synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

실제 RLHF에서는 이것이 human labeler로 대체됩니다. 형태, 즉 `(prompt, preferred_response, rejected_response)`는 동일합니다.

### 2단계: Bradley-Terry reward model

Linear score는 `R(x, y) = w · bag(y)`입니다. BT pairwise log-loss를 최소화하도록 학습합니다.

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

몇백 번 update하면 `w`는 good-word token에는 양의 weight를, bad token에는 음의 weight를 할당합니다.

### 3단계: RM 위의 PPO-like policy

우리 toy policy는 vocabulary에서 token 하나를 생성합니다. RM으로 token을 채점하고, `log π_θ(token | prompt)`를 계산하고, reference까지의 KL penalty를 더한 뒤, clipped PPO surrogate를 적용합니다.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 4단계: KL 모니터링

매 update마다 평균 `KL(π_θ || π_ref)`를 추적합니다. 이것이 `~5-10`을 넘어 서서히 올라가면 policy가 `π_SFT`에서 멀리 drift한 것입니다. `β`가 너무 낮거나 reward hacking이 시작되고 있다는 뜻입니다. 실제 RLHF에서 가장 중요한 진단 지표입니다.

### 5단계: TRL을 사용한 production recipe

toy pipeline을 이해했다면, 실제 library user가 쓰는 동일한 loop는 다음과 같습니다. Hugging Face의 [TRL](https://huggingface.co/docs/trl)이 reference implementation입니다. Stage 2에는 `RewardTrainer`, Stage 3에는 KL-to-reference가 내장된 `PPOTrainer`를 씁니다.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

library가 대신 해 주는 일은 세 가지입니다. `adap_kl_ctrl=True`는 adaptive-β schedule을 구현합니다. 관측 KL이 `target_kl`을 넘으면 β를 두 배로 하고, 절반보다 낮으면 β를 절반으로 줄입니다. reference model은 관례상 frozen입니다. `policy`와 parameter를 실수로 공유하면 안 됩니다. value head는 policy와 같은 backbone 위에 놓입니다(`AutoModelForCausalLMWithValueHead`가 scalar MLP head를 붙입니다). 그래서 TRL은 `policy/kl`과 `value/loss`를 따로 보고합니다.

## 함정

- **Over-optimization / reward hacking.** RM은 불완전합니다. `π_θ`는 점수는 높지만 품질은 나쁜 adversarial completion을 찾아냅니다. 증상: reward는 계속 오르는데 human eval score는 정체하거나 떨어집니다. 해결: early stop, `β` 증가, RM training data 확대.
- **Length hacking.** helpful response로 학습한 RM은 종종 암묵적으로 길이를 보상합니다. policy는 response를 padding하는 법을 배웁니다. 대응: length-normalized reward, 또는 length-aware RM을 쓰는 RLAIF.
- **너무 작은 RM.** RM은 policy와 적어도 같은 크기여야 합니다. 작은 RM은 policy output을 충실하게 채점할 수 없습니다.
- **KL tuning.** β가 너무 낮으면 drift와 reward hacking이 생깁니다. β가 너무 높으면 policy가 거의 변하지 않습니다. 표준 요령은 step당 고정 KL을 목표로 하는 *adaptive* β입니다.
- **Preference-data noise.** human label의 약 30%는 noisy하거나 ambiguous합니다. agreement-filtered data로 RM을 학습하거나 BT에 temperature를 넣어 calibrate합니다.
- **Off-policy problems.** PPO data는 첫 epoch 뒤에는 약간 off-policy가 됩니다. Lesson 08처럼 clip fraction을 모니터링하세요.

## 활용하기

2026년의 RLHF는 층을 나누어 쓰입니다.

| 계층 | 목표 | 방법 |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | RLHF-PPO보다 DPO(Phase 10 · 08)가 선호됩니다. |
| 추론 정확도(math, code) | Capability | verifier reward를 쓰는 GRPO(Phase 9 · 12). |
| 긴 horizon의 multi-step task | Agentic | step 위의 process reward model을 쓰는 PPO / GRPO. |
| Safety / refusal behavior | Safety | 별도 safety RM을 쓰는 RLHF-PPO, 또는 Constitutional AI. |
| inference 시점의 Best-of-N | 빠른 alignment | decode time에 RM을 사용합니다. policy training은 필요 없습니다. |
| Reward distillation | inference compute | frozen LM 위에 작은 "reward head"를 학습합니다. |

RLHF는 2022-2024년에 *대표적인* 방법이었습니다. 2026년 production alignment pipeline은 DPO-first이고, PPO는 RM-intensive 단계나 safety-critical 단계에만 남아 있습니다.

## 출시하기

`outputs/skill-rlhf-architect.md`로 저장하세요.

```markdown
---
name: rlhf-architect
description: RM, KL, 데이터 전략을 포함해 언어 모델을 위한 RLHF / DPO / GRPO alignment 파이프라인을 설계합니다.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

기반 LM, 목표 행동(alignment / reasoning / refusal / agent), 그리고 preference 또는 verifier 예산이 주어지면 다음을 출력하세요.

1. 단계. SFT? RM? DPO? GRPO? 근거와 함께 제시합니다.
2. preference 또는 verifier 출처. 사람, AI 피드백, 규칙 기반, unit-test-pass, 또는 reward distillation.
3. KL 전략. 고정 β, adaptive β, 또는 DPO(암묵적 KL).
4. 진단 지표. 평균 KL, reward 안정성, over-optimization 방어 장치(holdout human eval).
5. 안전 게이트. red-team 세트, refusal rate, helpfulness RM과 분리된 safety RM.

KL 모니터 없이 RLHF-PPO를 출시하지 마세요. 목표 policy보다 작은 RM을 사용하지 마세요. 길이만 보상하는 reward를 사용하지 마세요. blind human-eval 세트를 보류하지 않는 파이프라인은 over-optimization 보호가 부족하다고 표시하세요.
```

## 연습 문제

1. **Easy.** `code/main.py`에서 Bradley-Terry reward model을 500개의 synthetic preference pair로 학습하세요. hold-out 100 pair에서 pairwise accuracy를 측정하세요. 90%를 넘어야 합니다.
2. **Medium.** `β ∈ {0.0, 0.1, 1.0}`로 toy PPO-RLHF loop를 실행하세요. 각각에 대해 update 동안 RM score와 KL-to-reference를 그리세요. 어떤 run이 reward-hack하나요?
3. **Hard.** 같은 preference data에 DPO(closed-form preference-likelihood loss)를 구현하고, 사용한 compute와 달성한 최종 RM score를 RLHF-PPO pipeline과 비교하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| RLHF | "alignment RL" | 세 단계 SFT + RM + PPO pipeline(Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "점수화 network" | Bradley-Terry로 pairwise preference에 맞춘 learned scalar function. |
| Bradley-Terry | "pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; 표준 RM objective. |
| KL penalty | "reference 근처에 머물기" | reward 안의 `β · KL(π_θ \|\| π_ref)`; reward hacking을 막는 regularizer. |
| Reward hacking | "Goodhart의 법칙" | policy가 RM 결함을 이용합니다. 증상: reward 상승, human eval 정체. |
| RLAIF | "AI가 label한 preference" | label이 사람 대신 다른 LM에서 나오는 RLHF. |
| PRM | "Process Reward Model" | partial reasoning step에 점수를 매깁니다. reasoning pipeline에서 사용됩니다. |
| Constitutional AI | "Anthropic 방법" | 명시적 규칙으로 유도된 AI-generated preference. |

## 더 읽을거리

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — RLHF를 시작한 논문입니다.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 뒤의 레시피입니다.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — summarization을 위한 초기 RLHF입니다.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO입니다. 2026년 post-RLHF 기본값입니다.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF와 self-critique loop.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 논문입니다.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — production `RewardTrainer`와 `PPOTrainer`. adaptive-KL과 value-head 세부 사항은 trainer source를 읽어 보세요.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — diagram을 곁들인 세 단계 pipeline의 대표적인 walkthrough입니다.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — library입니다. `examples/`에는 Llama, Mistral, Qwen을 위한 end-to-end RLHF script가 있습니다.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward-hacking을 생각하는 데 필수인 reward-hypothesis 관점입니다.
