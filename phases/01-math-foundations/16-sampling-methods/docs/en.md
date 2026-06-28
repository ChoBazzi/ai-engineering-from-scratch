# 샘플링 방법

> 샘플링은 AI가 가능성의 공간을 탐색하는 방식입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 minutes

## 학습 목표

- uniform random numbers만 사용해 inverse CDF, rejection, importance sampling을 처음부터 구현한다
- language model token generation을 위한 temperature, top-k, top-p(nucleus) sampling을 만든다
- reparameterization trick과 그것이 VAE에서 sampling을 통과하는 backpropagation을 가능하게 하는 이유를 설명한다
- Metropolis-Hastings MCMC를 실행해 unnormalized target distribution에서 sample을 뽑는다

## 문제

language model이 prompt 처리를 마치고 50,000개의 logits vector를 생성합니다. vocabulary의 token마다 하나씩입니다. 이제 하나를 골라야 합니다. 어떻게 고를까요?

항상 가장 높은 probability의 token을 고르면 모든 response가 동일해집니다. Deterministic하고 지루합니다. uniformly at random으로 고르면 output은 gibberish가 됩니다. 정답은 이 극단들 사이 어딘가에 있으며, 그 어딘가는 sampling이 제어합니다.

sampling은 text generation에만 국한되지 않습니다. Reinforcement learning은 trajectory를 sampling해 policy gradient를 추정합니다. VAE는 learned distributions에서 sample을 뽑고 randomness를 통과해 backpropagation하면서 latent representation을 학습합니다. Diffusion model은 noise를 sampling하고 반복적으로 denoising해 image를 생성합니다. Monte Carlo methods는 closed-form solution이 없는 integral을 추정합니다. MCMC algorithms는 enumerate할 수 없는 high-dimensional posterior distributions를 탐색합니다.

모든 generative AI system은 sampling system입니다. sampling strategy가 output의 quality, diversity, controllability를 결정합니다. 이 lesson은 uniform random numbers에서 시작해 modern LLMs와 generative models를 구동하는 technique까지, 주요 sampling method를 모두 처음부터 만듭니다.

## 개념

### Sampling이 중요한 이유

sampling은 AI와 machine learning 전반에서 네 가지 fundamental role로 나타납니다.

**Generation.** Language models, diffusion models, GANs는 모두 sampling으로 output을 생성합니다. sampling algorithm은 creativity, coherence, diversity를 직접 제어합니다. Temperature, top-k, nucleus sampling은 engineer가 매일 조절하는 knob입니다.

**Training.** Stochastic gradient descent는 mini-batch를 sampling합니다. Dropout은 deactivate할 neuron을 sampling합니다. Data augmentation은 random transformation을 sampling합니다. Importance sampling은 reinforcement learning(PPO, TRPO)에서 gradient variance를 줄이기 위해 sample에 reweighting을 적용합니다.

**Estimation.** ML의 많은 quantity에는 closed-form solution이 없습니다. data distribution에 대한 expected loss, energy-based model의 partition function, Bayesian inference의 evidence가 그렇습니다. Monte Carlo estimation은 sample을 평균내어 이 모든 것을 근사합니다.

**Exploration.** MCMC algorithms는 Bayesian inference에서 posterior distributions를 탐색합니다. Evolutionary strategies는 parameter perturbation을 sampling합니다. Thompson sampling은 bandit에서 exploration과 exploitation의 균형을 맞춥니다.

핵심 challenge는 simple distribution(uniform, normal)에서만 직접 sample할 수 있다는 점입니다. 그 밖의 모든 경우에는 simple sample을 target distribution의 sample로 변환하는 method가 필요합니다.

### 균등 random sampling

모든 sampling method는 여기에서 시작합니다. uniform random number generator는 [0, 1) 안의 value를 생성하며, 길이가 같은 모든 sub-interval은 같은 probability를 가집니다.

```text
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

n개 item으로 이루어진 discrete set에서 uniformly sample하려면 U를 생성하고 floor(n * U)를 반환합니다. continuous range [a, b]에서 sample하려면 a + (b - a) * U를 계산합니다.

핵심 insight는 single uniform random number가 어떤 distribution에서든 sample 하나를 만들기에 정확히 충분한 randomness를 담고 있다는 점입니다. trick은 올바른 transformation을 찾는 것입니다.

### Inverse CDF method(inverse transform sampling) 설명

cumulative distribution function(CDF)은 value를 probability로 mapping합니다.

```text
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

inverse CDF는 probability를 다시 value로 mapping합니다. U ~ Uniform(0, 1)이면 X = F_inverse(U)는 target distribution을 따릅니다.

```text
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential distribution 예시:**

```text
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

F_inverse를 closed form으로 쓸 수 있을 때 이 방법은 완벽하게 작동합니다. normal distribution에는 closed-form inverse CDF가 없으므로 다른 method(Box-Muller 또는 numerical approximation)를 사용합니다.

**Discrete version:** discrete distribution에서는 cumulative sum으로 CDF를 만들고, U를 생성한 뒤 cumulative sum이 U를 초과하는 첫 index를 찾습니다. 이것이 Lesson 06의 `sample_categorical`이 작동하는 방식입니다.

### Rejection sampling 설명

CDF를 invert할 수 없지만 target PDF를 constant까지 evaluate할 수 있다면 rejection sampling이 작동합니다.

```text
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

bound M이 tight할수록 acceptance rate가 높아집니다. low dimension(1-3)에서는 rejection sampling이 잘 작동합니다. high dimension에서는 proposal volume 대부분이 reject되므로 acceptance rate가 exponentially 떨어집니다. 이것이 rejection sampling의 curse of dimensionality입니다.

**Example: sampling from a truncated normal.** truncated range 위의 uniform proposal을 사용합니다. envelope M은 그 range에서 normal PDF의 maximum입니다.

**Example: sampling from a semicircle.** bounding rectangle 안에서 uniformly propose합니다. point가 semicircle 안에 있으면 accept합니다. Monte Carlo가 pi를 계산하는 방식이 이것입니다. acceptance rate는 area ratio pi/4와 같습니다.

### Importance sampling 설명

때로는 target distribution p(x)의 sample 자체가 필요하지 않습니다. p(x) 아래에서 expectation을 estimate해야 하는데, 다른 distribution q(x)의 sample을 가지고 있을 수 있습니다.

```text
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

이것은 reinforcement learning에서 매우 중요합니다. PPO(Proximal Policy Optimization)에서는 old policy pi_old 아래에서 trajectory를 수집하지만 new policy pi_new를 optimize하고 싶습니다. importance weight는 pi_new(a|s) / pi_old(a|s)입니다. PPO는 new policy가 old policy에서 너무 멀리 벗어나지 않도록 이 weight를 clip합니다.

importance sampling estimator의 variance는 q가 p와 얼마나 비슷한지에 달려 있습니다. q가 p와 매우 다르면 몇 개의 sample이 enormous weight를 받아 estimate를 지배합니다. Self-normalized importance sampling은 이 문제를 줄이기 위해 weight의 합으로 나눕니다.

```text
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo estimation 설명

Monte Carlo estimation은 random sample의 average로 integral을 approximate합니다. law of large numbers가 convergence를 보장합니다.

```text
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

error rate는 dimension과 무관합니다. 그래서 grid-based integration이 불가능한 high dimension에서는 Monte Carlo methods가 지배적입니다.

**pi 추정:**

```text
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**기댓값 추정:**

```text
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo(MCMC): Metropolis-Hastings 설명

MCMC는 stationary distribution이 target distribution p(x)인 Markov chain을 구성합니다. 충분히 많은 step 뒤에는 chain의 sample이 p(x)의 sample과 approximate하게 같습니다.

```text
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

symmetric proposal(q(x'|x) = q(x|x'))에서는 ratio가 p(x')/p(x)로 단순해집니다. 이것이 original Metropolis algorithm입니다.

**작동 이유.** acceptance rule은 detailed balance를 보장합니다. x에 있다가 x'로 이동할 probability가 x'에 있다가 x로 이동할 probability와 같아집니다. detailed balance는 p(x)가 chain의 stationary distribution임을 의미합니다.

**실전 고려사항:**
- Burn-in: chain이 equilibrium에 도달하기 전의 early sample을 버립니다
- Thinning: autocorrelation을 줄이기 위해 k-th sample마다 하나씩 유지합니다
- Proposal scale: 너무 작으면 chain이 느리게 움직입니다(high acceptance, slow exploration). 너무 크면 대부분의 proposal이 reject됩니다(low acceptance, stuck in place)
- high dimension에서 Gaussian proposal의 optimal acceptance rate는 약 0.234입니다

### Gibbs sampling 설명

Gibbs sampling은 multivariate distribution을 위한 MCMC의 special case입니다. 모든 dimension에서 한 번에 move를 propose하는 대신 conditional distribution에서 variable을 하나씩 update합니다.

```text
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling을 쓰려면 각 conditional distribution p(x_i | x_{-i})에서 sample할 수 있어야 합니다. 많은 model에서는 이것이 straightforward합니다.
- Bayesian networks: conditional은 graph structure에서 나온다
- Gaussian mixtures: conditional은 Gaussian이다
- Ising models: 각 spin의 conditional은 neighbor에만 의존한다

exact conditional에서 sampling하면 detailed balance를 자동으로 만족하므로 acceptance rate는 항상 1입니다. 모든 proposal이 accept됩니다.

**한계.** variable 사이의 correlation이 높으면 Gibbs sampling은 천천히 mix합니다. variable을 하나씩 update하는 방식으로는 distribution을 가로지르는 큰 diagonal move를 만들 수 없기 때문입니다.

### Temperature sampling (LLM에서 사용)

language model은 vocabulary의 각 token에 대해 logits z_1, ..., z_V를 output합니다. Softmax는 이것을 probability로 변환합니다. Temperature는 softmax 전에 logits를 rescale합니다.

```text
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**작동 이유.** logits를 T < 1로 나누면 logits 사이의 difference가 amplify됩니다. z_1 = 2이고 z_2 = 1일 때 T = 0.5로 나누면 z_1/T = 4, z_2/T = 2가 되어 gap이 커집니다. softmax 뒤에는 highest-logit token이 훨씬 더 큰 share를 받습니다.

**실전에서는:**
- T = 0.0: greedy decoding, factual Q&A에 가장 좋음
- T = 0.3-0.7: 약간 creative, code generation에 좋음
- T = 0.7-1.0: balanced, general conversation에 좋음
- T = 1.0-1.5: creative writing, brainstorming
- T > 1.5: 점점 random해지며 거의 유용하지 않음

Temperature는 가능한 token의 종류를 바꾸지 않습니다. 각 token에 할당되는 probability mass를 바꿉니다.

### Top-k sampling 설명

Top-k sampling은 candidate set을 highest probability를 가진 k개 token으로 제한한 뒤 renormalize하고 그 restricted set에서 sample합니다.

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k는 vocabulary distribution의 long tail에 존재하는 매우 unlikely token(typo, nonsense)을 model이 선택하지 못하게 합니다. 문제는 k가 context와 무관하게 fixed라는 점입니다. model이 confident해서 한 token이 95% probability를 가질 때도 k = 40은 39개의 alternative를 허용합니다. model이 uncertain해서 probability가 1000개 token에 퍼져 있을 때는 k = 40이 plausible option을 잘라냅니다.

### Top-p(nucleus) sampling 설명

Top-p sampling은 candidate set size를 dynamic하게 조정합니다. fixed number의 token을 유지하는 대신 cumulative probability가 p를 초과하는 가장 작은 token set을 유지합니다.

```text
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

model이 confident할 때 nucleus sampling은 token을 적게 유지합니다(아마 2-3개). model이 uncertain할 때는 많이 유지합니다(아마 200개). 이런 adaptive behavior 때문에 nucleus sampling은 일반적으로 top-k보다 더 나은 text를 생성합니다.

**흔한 조합:**
- Temperature 0.7 + top-p 0.9: good general-purpose setting
- Temperature 0.0 (greedy): deterministic task에 가장 좋음
- Temperature 1.0 + top-k 50: Fan et al. (2018) original paper setting

Top-k와 top-p는 결합할 수 있습니다. 먼저 top-k를 적용한 뒤 remaining set에 top-p를 적용합니다.

### Reparameterization trick (VAE에서 사용)

Variational autoencoders(VAEs)는 input을 latent space의 distribution으로 encode하고, 그 distribution에서 sample한 뒤 sample을 다시 decode하면서 학습합니다. 문제는 sampling operation을 통과해 backpropagate할 수 없다는 점입니다.

```text
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

reparameterization trick은 randomness를 parameter에서 분리합니다.

```text
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

N(mu, sigma^2)는 mu + sigma * N(0, 1)과 같은 distribution을 가지므로 이것이 작동합니다. 핵심 insight는 randomness를 parameter-free source(epsilon)로 옮긴 다음 sample을 parameter의 differentiable transformation으로 표현하는 것입니다.

**VAE training loop에서는:**
1. Encoder가 각 input에 대해 mu와 log(sigma^2)를 output한다
2. epsilon ~ N(0, 1)을 sample한다
3. z = mu + sigma * epsilon을 계산한다
4. z를 decode해 input을 reconstruct한다
5. step 4, 3, 2, 1을 통해 backpropagate한다(step 3이 differentiable이므로 가능)

reparameterization trick이 없으면 VAE는 standard backpropagation으로 학습할 수 없습니다. 이 하나의 insight가 VAE를 practical하게 만들었습니다.

### Gumbel-Softmax(differentiable categorical sampling) 설명

reparameterization trick은 continuous distributions(Gaussian)에 작동합니다. discrete categorical distributions에는 다른 approach가 필요합니다. Gumbel-Softmax는 categorical sampling에 대한 differentiable approximation을 제공합니다.

**Gumbel-Max trick(non-differentiable) 설명:**

```text
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax(differentiable approximation) 설명:**

```text
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax는 discrete sample의 continuous relaxation을 생성합니다. output은 hard one-hot이 아니라 probability vector(soft one-hot)입니다. gradient는 softmax를 통과해 흐릅니다. training의 forward pass 중에는 "straight-through" estimator를 사용할 수 있습니다. forward pass에는 hard argmax를 사용하고 backward pass에는 soft Gumbel-Softmax gradients를 사용하는 방식입니다.

**응용:**
- VAEs의 discrete latent variables
- Neural architecture search(discrete operations 선택)
- Hard attention mechanisms
- discrete actions가 있는 reinforcement learning

### Stratified sampling 설명

standard Monte Carlo sampling은 우연히 sample space에 gap을 남길 수 있습니다. Stratified sampling은 space를 strata로 나누고 각 strata에서 sample해 even coverage를 강제합니다.

```text
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling은 standard Monte Carlo에 비해 variance가 항상 낮거나 같습니다.

```text
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**응용:**
- Numerical integration(quasi-Monte Carlo)
- Training data splits(each fold의 class balance 보장)
- stratification을 사용하는 importance sampling(두 technique 결합)
- NeRF(Neural Radiance Fields)는 camera ray를 따라 stratified sampling을 사용한다

### Diffusion Models와의 연결

Diffusion model은 sampling process를 통해 image를 생성합니다. forward process는 T step에 걸쳐 image에 Gaussian noise를 더해 pure noise가 될 때까지 진행합니다. reverse process는 denoising을 학습해 original image를 step by step으로 복구합니다.

```text
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

이 lesson의 method들과의 연결:
- 각 denoising step은 reparameterization trick을 사용한다(noise를 sample하고 deterministic transform 적용)
- noise schedule {alpha_t}는 temperature annealing의 한 형태를 제어한다
- training은 Monte Carlo estimation을 사용해 ELBO(evidence lower bound)를 approximate한다
- diffusion model의 ancestral sampling은 Markov chain이다(각 step은 current state에만 의존)

전체 image generation process는 iterative sampling입니다. noise에서 시작해 각 step에서 learned denoising model에 condition된, 조금 덜 noisy한 version을 sample합니다.

```figure
monte-carlo-pi
```

## 직접 만들기

### 단계 1: Uniform and inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

10,000개의 exponential sample을 생성하고 mean이 1/lambda인지 검증하세요.

### 단계 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

rejection sampling을 사용해 truncated normal distribution에서 sample을 뽑으세요. sample을 histogram으로 만들어 shape를 검증하세요.

### 단계 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

uniform proposal을 사용해 normal distribution 아래의 E[X^2]를 estimate하세요. 알려진 answer(mu^2 + sigma^2)와 비교하세요.

### 단계 4: Monte Carlo estimation of pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 단계 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

bimodal distribution(two Gaussians의 mixture)에서 sample하세요. chain의 trajectory를 visualize하세요.

### 단계 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 단계 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

token logits set에 대해 temperature가 output distribution을 어떻게 바꾸는지 보여 주세요.

### 단계 8: Top-k and top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 단계 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

gradient가 reparameterized sample을 통과해 흐르지만 direct sampling은 통과하지 못한다는 것을 보여 주세요.

### 단계 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

temperature를 낮추면 output이 one-hot vector에 가까워지는 방식을 보여 주세요.

모든 visualization을 포함한 full implementation은 `code/sampling.py`에 있습니다.

## 사용하기

NumPy와 SciPy를 사용하는 production version:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

scale 있는 MCMC에는 dedicated libraries를 사용합니다.
- PyMC: NUTS(adaptive HMC)를 갖춘 full Bayesian modeling
- emcee: ensemble MCMC sampler
- NumPyro/JAX: GPU-accelerated MCMC

여러분은 이것들을 처음부터 만들었습니다. 이제 library call이 무엇을 하는지 압니다.

## 연습 문제

1. Cauchy distribution을 위한 inverse CDF sampling을 구현하세요. CDF는 F(x) = 0.5 + arctan(x)/pi입니다. 10,000개의 sample을 생성하고 histogram을 true PDF와 비교해 plot하세요. heavy tail(center에서 멀리 떨어진 extreme value)을 관찰하세요.

2. Uniform(0, 1) proposal을 사용해 Beta(2, 5) distribution에서 sample을 생성하도록 rejection sampling을 사용하세요. accepted sample을 true Beta PDF와 비교해 plot하세요. theoretical acceptance rate는 얼마인가요?

3. Monte Carlo를 사용해 0부터 pi까지 sin(x)의 integral을 1,000, 10,000, 100,000 sample로 estimate하세요. 각 level에서 error를 비교하세요. error가 O(1/sqrt(N))로 scale하는지 검증하세요.

4. Metropolis-Hastings를 구현해 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2)에 proportional한 2D distribution p(x, y)에서 sample하세요. sample과 chain trajectory를 plot하세요. 다른 proposal standard deviation으로 실험하세요.

5. complete text generation demo를 만드세요. 10개 word vocabulary와 logits가 주어졌을 때 (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9를 사용해 20 token sequence를 생성하세요. 5 run에 걸쳐 output diversity를 비교하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Sampling | "random value를 뽑기" | probability distribution에 따라 value를 생성하는 것. 모든 generative AI 뒤의 mechanism |
| Uniform distribution | "모두 equally likely" | [a, b]의 모든 value가 같은 probability density 1/(b-a)를 가진다. 모든 sampling method의 starting point |
| Inverse CDF | "Probability transform" | F_inverse(U)는 uniform sample을 known CDF를 가진 어떤 distribution의 sample로 바꾼다. exact하고 efficient하다 |
| Rejection sampling | "Propose and accept/reject" | simple proposal에서 생성하고 target/proposal ratio에 proportional한 probability로 accept한다. exact하지만 sample을 낭비한다 |
| Importance sampling | "sample에 reweighting" | q(x)의 sample을 사용해 각 sample에 p(x)/q(x)를 곱함으로써 p(x) 아래의 expectation을 estimate한다. RL의 PPO 핵심 |
| Monte Carlo | "random sample 평균" | integral을 sample average로 approximate한다. dimension과 무관하게 error O(1/sqrt(N)) |
| MCMC | "converge하는 random walk" | stationary distribution이 target인 Markov chain을 구성한다. Metropolis-Hastings가 foundational algorithm이다 |
| Metropolis-Hastings | "uphill은 accept, downhill도 가끔 accept" | move를 propose하고 density ratio에 따라 accept한다. detailed balance가 target distribution으로의 convergence를 보장한다 |
| Gibbs sampling | "한 번에 variable 하나" | 다른 variable을 fixed로 둔 conditional distribution에서 각 variable을 update한다. 100% acceptance rate |
| Temperature | "Confidence knob" | softmax 전에 logits를 T로 나눈다. T<1은 sharpen(more confident), T>1은 flatten(more diverse) |
| Top-k sampling | "k개의 best만 유지" | k개의 highest-probability token을 제외한 나머지를 zero out하고 renormalize한 뒤 sample한다. fixed candidate set size |
| Nucleus sampling (top-p) | "probable한 것들만 유지" | cumulative probability가 p를 초과하는 가장 작은 token set을 유지한다. adaptive candidate set size |
| Reparameterization trick | "randomness를 밖으로 옮기기" | epsilon ~ N(0,1)일 때 z = mu + sigma * epsilon으로 쓴다. sampling을 differentiable하게 만든다. VAE training에 필수 |
| Gumbel-Softmax | "Soft categorical sampling" | Gumbel noise + temperature가 있는 softmax를 사용한 categorical sampling의 differentiable approximation |
| Stratified sampling | "coverage 강제" | sample space를 strata로 나누고 각 strata에서 sample한다. naive Monte Carlo보다 항상 낮은 variance |
| Burn-in | "Warm-up period" | chain이 stationary distribution에 도달하기 전의 initial MCMC sample을 버리는 것 |
| Detailed balance | "Reversibility condition" | p(x) * T(x->y) = p(y) * T(y->x). p가 Markov chain의 stationary distribution이기 위한 sufficient condition |
| Diffusion sampling | "Iterative denoising" | noise에서 시작해 learned denoising step을 적용해 data를 생성한다. 각 step은 conditional sampling operation |

## 더 읽을거리

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC foundation에 대한 detailed tutorial
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - original Gumbel-Softmax paper
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus(top-p) sampling paper
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - reparameterization trick을 소개한 VAE paper
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - sampling과 image generation을 연결하는 DDPM
