---
name: skill-sampling-strategy
description: generation, estimation, inference에 맞는 sampling method 선택
version: 1.0.0
phase: 1
lesson: 16
tags: [sampling, mcmc, generation]
---

# Sampling Strategy 선택

text generation, Bayesian inference, Monte Carlo estimation, training에 맞는 sampling method를 고르는 방법입니다.

## 결정 체크리스트

1. output(text, images)을 생성하나요, 아니면 quantity(integral, expectation)를 추정하나요?
2. target distribution에서 직접 sample할 수 있나요, 아니면 density만 평가할 수 있나요?
3. target distribution은 discrete인가요, continuous인가요?
4. sample space의 dimension은 얼마인가요? 낮음(< 5), 중간(5-100), 높음(> 100)?
5. exact samples가 필요한가요, approximate samples가 필요한가요?
6. sampling operation을 통과하는 gradients가 필요한가요?

## 각 method를 사용할 때

| Method | When to use | Complexity | Exact? |
|---|---|---|---|
| Direct sampling | CDF가 있거나 library function을 쓸 수 있음 | sample당 O(1) | Yes |
| Inverse CDF | closed-form CDF inverse가 알려져 있음(exponential, Cauchy) | sample당 O(1) | Yes |
| Box-Muller | library 없이 normal samples가 필요함 | sample당 O(1) | Yes |
| Rejection sampling | target PDF를 평가할 수 있고 dimension이 낮음(1-3) | sample당 O(1/acceptance) | Yes |
| Importance sampling | individual samples가 아니라 expectations가 필요함 | n samples에 O(n) | Approximate |
| Stratified sampling | Monte Carlo estimation에서 더 낮은 variance가 필요함 | n samples에 O(n) | Approximate |
| Metropolis-Hastings | high-dimensional이고 unnormalized density를 평가할 수 있음 | step당 O(1) + burn-in | Asymptotically |
| Gibbs sampling | 각 conditional distribution에서 sample할 수 있음 | full sweep당 O(d) | Asymptotically |
| HMC/NUTS | high-dimensional continuous, smooth density | step당 O(L * d) | Asymptotically |
| Temperature sampling | LLM text generation에서 creativity 제어 | vocab size V에 O(V) | N/A |
| Top-k sampling | LLM generation에서 unlikely tokens 제거 | O(V log k) | N/A |
| Top-p (nucleus) | LLM generation에서 adaptive candidate set 사용 | O(V log V) | N/A |
| Reparameterization | Gaussian sampling(VAEs)을 통과하는 gradients가 필요함 | O(d) | Yes |
| Gumbel-Softmax | categorical sampling을 통과하는 gradients가 필요함 | k classes에 O(k) | Approximate |

## LLM generation 설정

| Use case | Temperature | Top-p | Top-k | Notes |
|---|---|---|---|---|
| Factual Q&A | 0.0 (greedy) | -- | -- | Deterministic, randomness 없음 |
| Code generation | 0.2-0.5 | 0.9 | -- | 낮은 creativity, 높은 coherence |
| General chat | 0.7 | 0.9 | -- | 균형형 |
| Creative writing | 0.9-1.2 | 0.95 | -- | 더 높은 diversity |
| Brainstorming | 1.0-1.5 | 0.95 | -- | 최대 diversity, coherence를 잃을 수 있음 |

Temperature와 top-p는 함께 쓸 수 있습니다. 먼저 temperature를 적용해 logits를 scale한 뒤 top-p filtering을 적용하세요.

## MCMC method 선택

| Property | Metropolis-Hastings | Gibbs | HMC/NUTS |
|---|---|---|---|
| Dimension | 아무 dimension | 아무 dimension(best < 100) | High (100+) |
| Requires conditionals | No | Yes | No |
| Requires gradient | No | No | Yes |
| Acceptance rate | ~23%로 tune | 항상 100% | ~65%로 tune |
| Correlation | 높음(random walk) | 중간 | 낮음 |
| Burn-in | 김 | 중간 | 짧음 |
| Best for | Exploration, simple models | Conjugate models, Bayesian networks | Continuous posteriors, deep probabilistic models |

## 흔한 실수

- high dimensions에서 rejection sampling을 사용함. Acceptance rate는 dimension에 따라 지수적으로 떨어집니다. 5 dimensions를 넘으면 MCMC로 전환하세요.
- MCMC proposal variance를 너무 높거나 낮게 설정함. 너무 높으면 대부분의 proposals가 거부되어 chain이 멈춥니다. 너무 낮으면 모든 proposals가 accept되지만 chain이 천천히 움직입니다. random walk MH에서는 ~23% acceptance를 목표로 하세요.
- burn-in을 잊음. MCMC의 첫 N samples는 starting point에 의해 bias됩니다. 최소 1000 steps(복잡한 distributions에서는 더 많이)를 버리세요.
- target과 매우 다른 proposal로 importance sampling을 사용함. 몇 samples가 엄청난 weights를 받아 estimate가 불안정해집니다. effective sample size를 모니터링하세요: ESS = (sum w_i)^2 / sum(w_i^2).
- deterministic output이 필요한 tasks(예: classification, structured extraction)에 temperature > 0을 사용함. 대신 greedy(T=0)나 beam search를 사용하세요.
- top-p와 temperature를 결합하지 않음. Temperature만으로는 long tail의 garbage tokens를 제거하지 못합니다. Top-p는 제거합니다.
- standard sampling operation을 통해 backpropagate함. continuous(Gaussian)에는 reparameterization trick을, discrete(categorical)에는 Gumbel-Softmax를 사용하세요.

## 빠른 참조: variance reduction techniques

| Technique | How it works | Variance reduction |
|---|---|---|
| Stratified sampling | space를 strata로 나누고 각각 sample | 항상 <= standard MC |
| Antithetic variates | U와 1-U를 모두 사용 | monotone functions에서 작동 |
| Control variates | mean을 아는 variable을 빼기 | correlation에 비례 |
| Importance sampling | 더 나은 proposal의 samples를 reweight | proposal quality에 의존 |
| Latin hypercube | 각 dimension을 독립적으로 stratify | high-d에서 stratified보다 나음 |
