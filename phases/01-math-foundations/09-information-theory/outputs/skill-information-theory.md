---
name: skill-information-theory
description: ML loss function, model evaluation, feature selection에 information theory 개념을 적용한다
version: 1.0.0
phase: 1
lesson: 9
tags: [information-theory, entropy, loss-functions]
---

# ML을 위한 정보 이론

machine learning system에서 entropy, cross-entropy, KL divergence, mutual information을 언제 사용할지 정리합니다.

## 결정 체크리스트

1. 단일 distribution의 uncertainty를 측정하나요? **entropy**를 사용하세요.
2. model이 true label을 얼마나 잘 근사하는지 측정하나요? **cross-entropy**를 사용하세요(이것이 classification loss입니다).
3. 두 distribution 사이의 distance를 측정하나요? **KL divergence**를 사용하세요.
4. 두 variable이 관련 있는지 확인하나요? **mutual information**을 사용하세요.
5. language model quality를 report하나요? **perplexity**를 사용하세요(cross-entropy의 exponential).
6. 한 model을 다른 model로 distill하나요? teacher에서 student로의 **KL divergence**를 최소화하세요.

## 각 measure를 언제 사용할지

| Measure | Formula | Use case | ML application |
|---|---|---|---|
| Entropy H(P) | -sum(p log p) | How uncertain is this distribution? | Data complexity, maximum entropy models |
| Cross-entropy H(P,Q) | -sum(p log q) | How good is model Q at predicting true P? | Classification loss, language model loss |
| KL divergence D(P\|\|Q) | sum(p log(p/q)) | How different are P and Q? | VAE loss (ELBO), knowledge distillation, RLHF |
| Mutual information I(X;Y) | H(X) - H(X\|Y) | How much does Y tell us about X? | Feature selection, representation learning |
| Perplexity | exp(H(P,Q)) or 2^H | How confused is the model? | Language model evaluation |
| Conditional entropy H(X\|Y) | -sum(p(x,y) log p(x\|y)) | Remaining uncertainty in X after knowing Y | Feature informativeness |

## 핵심 관계

```text
Cross-entropy  = Entropy + KL divergence
H(P, Q)        = H(P)   + D_KL(P || Q)

Since H(P) is constant during training:
  Minimizing cross-entropy = Minimizing KL divergence

Mutual information = Entropy - Conditional entropy
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)

Perplexity = exp(cross-entropy in nats)
           = 2^(cross-entropy in bits)
```

## 빠른 참고: 공식과 단위

| Formula | Bits (log base 2) | Nats (log base e) |
|---|---|---|
| Information: -log(p) | -log2(p) | -ln(p) |
| Entropy: -sum(p log p) | bits | nats |
| 1 nat = | 1.4427 bits | 1 nat |
| PyTorch default | -- | nats |
| Information theory papers | bits | -- |

## 값 해석하기

| Entropy value | What it means |
|---|---|
| 0 | Deterministic. One outcome has probability 1. |
| log(n) | Maximum uncertainty. Uniform distribution over n outcomes. |
| Low | Distribution is peaked. Model is confident. |
| High | Distribution is flat. Model is uncertain. |

| Perplexity value | Language model quality |
|---|---|
| 1 | Perfect prediction (never happens in practice) |
| 10 | Choosing among ~10 equally likely tokens on average |
| 50 | GPT-2 level on standard benchmarks |
| < 10 | State-of-the-art for well-represented domains |

## 흔한 실수

- KL divergence를 계산하고 symmetric한 것처럼 다루기. D_KL(P||Q) != D_KL(Q||P). symmetric measure가 필요하면 Jensen-Shannon divergence를 사용하세요: JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5*(P+Q).
- one-hot label이 있는 cross-entropy가 -log(p_true_class)로 단순화된다는 점을 잊기. true distribution이 one-hot이면 모든 class에 대해 합을 계산할 필요가 없습니다.
- code에서는 log base 2를 쓰고 nats로 report하기(또는 반대). PyTorch는 기본적으로 natural log를 사용합니다. nats를 bits로 변환하려면 log2(e) = 1.4427을 곱하세요.
- empty 또는 zero-probability event의 entropy 계산하기. 관례: 0 * log(0) = 0, because lim(p->0) p*log(p) = 0.
- vocabulary가 다른 model의 perplexity를 비교하기. vocab size 50k이고 perplexity 30인 model은 vocab size 10k이고 perplexity 30인 model과 직접 비교할 수 없습니다.

## production ML에서 각 개념이 나타나는 곳

| Concept | Where you see it |
|---|---|
| Cross-entropy loss | Every classification model (nn.CrossEntropyLoss) |
| KL divergence | VAE ELBO, PPO clipping, knowledge distillation |
| Entropy regularization | Exploration bonus in RL (higher entropy = more exploration) |
| Mutual information | Feature selection, InfoNCE loss (contrastive learning) |
| Perplexity | Language model benchmarks (lower = better) |
| Label smoothing | Replaces one-hot with soft targets, reduces cross-entropy overconfidence |
| Temperature scaling | Divides logits by T before softmax, controls entropy of output |
