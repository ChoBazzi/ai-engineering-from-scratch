---
name: skill-probability-reasoning
description: 주어진 ML 문제에 맞는 확률분포를 선택한다
version: 1.0.0
phase: 1
lesson: 6
tags: [probability, distributions, modeling]
---

# 확률분포 선택

데이터를 모델링하거나, 손실 함수를 설계하거나, prior를 설정할 때 알맞은 분포를 고르는 방법입니다.

## 결정 체크리스트

1. 결과가 이산적인가요(categories, counts), 아니면 연속적인가요(measurements, scores)?
2. 결과가 bounded인가요(예: [0, 1]), 아니면 unbounded인가요?
3. 가능한 결과는 몇 개인가요? 두 개? k개? 무한개?
4. 데이터가 symmetric인가요, skewed인가요?
5. 사건들이 independent인가요, correlated인가요?
6. rate, count, proportion, measurement 중 무엇을 모델링하나요?

## 분포 결정 트리

```text
Is the variable discrete?
  Yes --> Only 2 outcomes? --> Bernoulli (p)
     |    k outcomes, one trial? --> Categorical (p1...pk)
     |    k outcomes, n trials? --> Multinomial (n, p1...pk)
     |    Count of successes in n trials? --> Binomial (n, p)
     |    Count of events per interval? --> Poisson (lambda)
     |    Count of trials until first success? --> Geometric (p)
     |    Count of trials until r successes? --> Negative Binomial (r, p)
  No --> Symmetric, bell-shaped? --> Normal (mu, sigma)
     |   Positive values, right-skewed? --> Log-normal or Exponential
     |   Bounded in [0, 1]? --> Beta (alpha, beta)
     |   Positive values, flexible shape? --> Gamma (alpha, beta)
     |   Time between events? --> Exponential (lambda)
     |   Heavy tails needed? --> Student's t (nu) or Cauchy
     |   Multivariate, bell-shaped? --> Multivariate Normal
     |   On a simplex (sums to 1)? --> Dirichlet (alpha)
```

## 실제 ML 시나리오를 분포에 매핑하기

| Scenario | Distribution | Parameters |
|---|---|---|
| Binary classification output | Bernoulli | p = sigmoid(logit) |
| Multi-class classification output | Categorical | p = softmax(logits) |
| Token prediction in language models | Categorical over vocab | p from softmax |
| Pixel intensity (normalized) | Beta or Uniform [0, 1] | Depends on image stats |
| Word count in a document | Poisson | lambda = avg word count |
| Time between user requests | Exponential | lambda = request rate |
| Measurement error | Normal | mu = 0, sigma from data |
| Weight initialization | Normal or Uniform | Kaiming/Xavier rules |
| VAE latent space prior | Standard Normal | mu = 0, sigma = 1 |
| Bayesian prior on proportions | Beta | alpha, beta from belief |
| Bayesian prior on category weights | Dirichlet | alpha vector |
| Noise in regression targets | Normal | mu = 0, sigma estimated |
| Outlier-robust regression | Student's t | low degrees of freedom |
| Duration/lifetime modeling | Weibull or Gamma | shape and scale |
| Topic distribution per document (LDA) | Dirichlet | alpha < 1 for sparse |

## 분포 선택이 잘못될 때

- 데이터에 단단한 하한이 있는데도 Normal을 사용하는 경우(예: prices, distances). normal은 음수 값에도 0이 아닌 확률을 부여합니다. 대신 log-normal이나 gamma를 사용하세요.
- 분산이 평균과 다른데 Poisson을 사용하는 경우. Poisson은 mean = variance를 가정합니다. variance > mean이면 negative binomial을 사용하세요.
- multi-class 문제에 Bernoulli를 사용하는 경우. Bernoulli는 엄격히 binary입니다. k > 2에는 categorical을 사용하세요.
- 관측값이 correlated인데 independence를 가정하는 경우. Time series, spatial data, grouped data는 independence를 위반합니다. autoregressive 또는 hierarchical model을 사용하세요.

## 흔한 실수

- PDF 값을 확률과 혼동하기. PDF는 1을 초과할 수 있습니다. 확률은 PDF를 구간에 대해 적분해서 나옵니다.
- softmax 출력이 independent Bernoulli probability가 아니라 categorical probability라는 점을 잊기. softmax 출력은 구성상 합이 1입니다.
- domain knowledge가 있는데 uniform prior를 사용하기. Informative prior는 잘 선택하면 결과를 편향시키지 않고 variance를 줄입니다.
- log-probability를 probability처럼 다루기. Log-prob는 항상 음수(또는 0)입니다. 합이 1이 되지 않습니다.

## 빠른 참고: 분포 속성

| Distribution | Support | Mean | Variance | Key property |
|---|---|---|---|---|
| Bernoulli(p) | {0, 1} | p | p(1-p) | Simplest discrete |
| Binomial(n, p) | {0..n} | np | np(1-p) | Sum of n Bernoulli |
| Poisson(lam) | {0, 1, 2, ...} | lam | lam | Mean = variance |
| Normal(mu, s^2) | (-inf, inf) | mu | s^2 | Max entropy for given mean/var |
| Exponential(lam) | [0, inf) | 1/lam | 1/lam^2 | Memoryless |
| Beta(a, b) | [0, 1] | a/(a+b) | ab/((a+b)^2(a+b+1)) | Conjugate to Binomial |
| Gamma(a, b) | (0, inf) | a/b | a/b^2 | Conjugate to Poisson |
| Dirichlet(alpha) | Simplex | alpha_i/sum | (see formula) | Conjugate to Categorical |
