# 확률과 분포

> 확률은 AI가 불확실성을 표현할 때 쓰는 언어입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## 학습 목표

- Bernoulli, categorical, Poisson, uniform, normal 분포의 PMF와 PDF를 처음부터 구현한다
- 기댓값과 분산을 계산하고, 중심극한정리를 사용해 왜 Gaussian이 자주 등장하는지 설명한다
- 수치 안정성 트릭인 max logit 빼기를 사용해 softmax와 log-softmax 함수를 만든다
- logits에서 cross-entropy loss를 계산하고 이를 negative log-likelihood와 연결한다

## 문제

분류기가 `[0.03, 0.91, 0.06]`을 출력합니다. 언어 모델은 50,000개 후보에서 다음 단어를 고릅니다. diffusion model은 학습된 분포에서 샘플링해 이미지를 생성합니다. 이 모든 것이 확률이 실제로 작동하는 모습입니다.

모델이 내놓는 모든 예측은 확률분포입니다. 모든 손실 함수는 예측 분포가 실제 분포에서 얼마나 먼지 측정합니다. 모든 학습 단계는 한 분포가 다른 분포와 더 비슷해지도록 파라미터를 조정합니다. 확률을 모르면 ML 논문 하나를 읽을 수도, 모델 하나를 디버깅할 수도, 학습 손실이 왜 NaN이 되는지 이해할 수도 없습니다.

## 개념

### 사건, 표본공간, 확률

표본공간 S는 가능한 모든 결과의 집합입니다. 사건은 표본공간의 부분집합입니다. 확률은 사건을 0과 1 사이의 숫자로 매핑합니다.

```text
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

세 공리가 모든 확률을 정의합니다:
1. 어떤 사건 A에 대해서도 P(A) >= 0
2. P(S) = 1 (무언가는 반드시 일어납니다)
3. A와 B가 동시에 일어날 수 없을 때 P(A or B) = P(A) + P(B)

나머지 모든 것, 즉 Bayes' theorem, 기댓값, 분포는 이 세 규칙에서 나옵니다.

### 조건부확률과 독립성

P(A|B)는 B가 일어났다는 조건에서 A가 일어날 확률입니다.

```text
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

두 사건은 하나를 안다고 해서 다른 하나에 대해 아무것도 알 수 없을 때 독립입니다:

```text
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

동전 던지기는 독립입니다. 복원하지 않고 카드를 뽑는 것은 독립이 아닙니다.

### 확률질량함수와 확률밀도함수

이산 확률변수에는 확률질량함수(PMF)가 있습니다. 각 결과에는 직접 읽을 수 있는 특정 확률이 있습니다.

```text
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

연속 확률변수에는 확률밀도함수(PDF)가 있습니다. 한 점에서의 밀도는 확률이 아닙니다. 확률은 구간에 대해 밀도를 적분해서 얻습니다.

```text
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

이 구분은 ML에서 중요합니다. 분류 출력은 PMF입니다(이산 선택). VAE latent space는 PDF를 사용합니다(연속).

### 자주 쓰는 분포

**Bernoulli:** 한 번의 시행, 두 결과. 이진 분류를 모델링합니다.

```text
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** 한 번의 시행, k개 결과. 다중 클래스 분류를 모델링합니다(softmax output).

```text
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** 모든 결과의 가능성이 같습니다. 무작위 초기화에 사용됩니다.

```text
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** 종 모양 곡선입니다. 평균(mu)과 분산(sigma^2)으로 매개변수화됩니다.

```text
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** 고정된 구간 안에서 드문 사건이 발생한 횟수입니다. 사건 발생률을 모델링합니다.

```text
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 기댓값과 분산

기댓값은 결과의 가중평균입니다.

```text
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

분산은 평균 주변의 퍼짐을 측정합니다.

```text
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

ML에서 기댓값은 손실 함수로 나타납니다(데이터 분포에 대한 평균 손실). 분산은 모델 안정성에 대해 알려줍니다. gradient의 분산이 높으면 학습이 noisy하다는 뜻입니다.

### 결합분포와 주변분포

결합분포 P(X, Y)는 두 확률변수를 함께 설명합니다.

결합 PMF 예시(X = weather, Y = umbrella):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

주변분포는 다른 변수를 합으로 제거합니다:

```text
P(X = x) = sum over all y of P(X = x, Y = y)
```

위 표의 행 합계와 열 합계가 주변분포입니다.

### 정규분포가 어디에나 나타나는 이유

중심극한정리: 원래 분포가 무엇이든, 많은 독립 확률변수의 합(또는 평균)은 정규분포로 수렴합니다.

```text
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

그래서:
- 측정 오차는 대략 정규분포입니다(많은 작은 독립 원천)
- 신경망의 weight initialization은 normal distribution을 사용합니다
- SGD의 gradient noise는 대략 정규분포입니다(많은 sample gradient의 합)
- 정규분포는 주어진 평균과 분산에서 maximum entropy distribution입니다

### 로그 확률

원시 확률은 수치 문제를 일으킵니다. 작은 확률을 많이 곱하면 빠르게 0으로 underflow됩니다.

```text
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

로그 확률은 이를 해결합니다. 곱셈은 덧셈이 됩니다.

```text
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

규칙:
- log(a * b) = log(a) + log(b)
- 로그 확률은 항상 <= 0입니다(0 < P <= 1이므로)
- 더 음수일수록 가능성이 낮습니다
- Cross-entropy loss는 정답 클래스의 negative log probability입니다

### 확률분포로서의 Softmax

신경망은 원시 점수(logits)를 출력합니다. Softmax는 이를 유효한 확률분포로 바꿉니다.

```text
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax trick: overflow를 막기 위해 지수화하기 전에 max logit을 뺍니다.

```text
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax는 수치 안정성을 위해 softmax와 log를 결합합니다. PyTorch는 cross-entropy loss에서 이를 내부적으로 사용합니다.

### 샘플링

샘플링은 분포에서 무작위 값을 뽑는다는 뜻입니다. ML에서는:
- Dropout이 어떤 neuron을 0으로 만들지 무작위로 샘플링합니다
- Data augmentation이 무작위 변환을 샘플링합니다
- 언어 모델이 예측 분포에서 다음 token을 샘플링합니다
- Diffusion model이 noise를 샘플링하고 점진적으로 denoise합니다

임의의 분포에서 샘플링하려면 inverse transform sampling, rejection sampling, 또는 VAE에서 쓰는 reparameterization trick 같은 기법이 필요합니다.

```figure
gaussian-pdf
```

## 직접 만들기

### Step 1: 확률 기초

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Step 2: PMF와 PDF를 처음부터 만들기

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Step 3: 기댓값과 분산

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Step 4: 분포에서 샘플링하기

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Step 5: Softmax와 로그 확률

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Step 6: 중심극한정리 시연

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Step 7: 시각화

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

모든 시각화를 포함한 전체 구현은 `code/probability.py`에 있습니다.

## 사용하기

NumPy와 SciPy를 쓰면 위의 모든 것이 한 줄짜리 호출이 됩니다:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

당신은 이것들을 처음부터 만들었습니다. 이제 라이브러리 호출이 무엇을 하는지 압니다.

## 연습문제

1. exponential distribution에 대한 inverse transform sampling을 구현하세요. 10,000개 값을 샘플링하고 histogram을 실제 PDF와 비교해 검증하세요.

2. 편향된 주사위 두 개에 대한 결합분포 표를 만드세요. 주변분포를 계산하고 두 주사위가 독립인지 확인하세요.

3. 정답 클래스가 index 3일 때 logits `[2.0, 0.5, -1.0, 3.0, 0.1]`을 출력하는 5-class classifier의 cross-entropy loss를 계산하세요. 그런 다음 PyTorch의 `nn.CrossEntropyLoss`로 답을 검증하세요.

4. log probability 목록을 받아 가장 가능성 높은 sequence, 전체 log probability, 이에 해당하는 raw probability를 반환하는 함수를 작성하세요. 각 단어의 확률이 0.01인 50단어 문장으로 테스트하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Sample space | "모든 가능성" | 실험에서 가능한 모든 결과의 집합 S |
| PMF | "확률 함수" | 각 이산 결과의 정확한 확률을 주며 합이 1이 되는 함수 |
| PDF | "확률 곡선" | 연속 변수의 밀도 함수입니다. 구간에 대해 적분해야 확률을 얻습니다 |
| Conditional probability | "무언가가 주어졌을 때의 확률" | P(A\|B) = P(A and B) / P(B). Bayesian thinking과 Bayes' theorem의 기반 |
| Independence | "서로 영향을 주지 않는다" | P(A and B) = P(A) * P(B). 한 사건을 알아도 다른 사건에 대해 알 수 없습니다 |
| Expected value | "평균" | 모든 결과의 확률 가중합입니다. 손실 함수는 기댓값입니다 |
| Variance | "얼마나 퍼져 있는지" | 평균으로부터의 제곱 편차의 기댓값입니다. 높은 분산 = noisy하고 불안정한 추정 |
| Normal distribution | "종 모양 곡선" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). CLT 때문에 어디에나 나타납니다 |
| Central Limit Theorem | "평균은 정규분포가 된다" | 원천 분포와 관계없이 많은 독립 sample의 평균은 정규분포로 수렴합니다 |
| Joint distribution | "두 변수를 함께" | P(X, Y)는 X와 Y 결과의 모든 조합에 대한 확률을 설명합니다 |
| Marginal distribution | "다른 변수를 합으로 제거" | P(X) = sum_y P(X, Y). 결합분포에서 한 변수의 분포를 회수합니다 |
| Log probability | "확률의 로그" | log P(x). 곱을 합으로 바꾸어 긴 sequence에서 수치 underflow를 막습니다 |
| Softmax | "점수를 확률로 바꾸기" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). 실수값 logits를 유효한 확률분포로 매핑합니다 |
| Cross-entropy | "손실 함수" | -sum(p_true * log(p_predicted)). 두 분포가 얼마나 다른지 측정합니다. 낮을수록 좋습니다 |
| Logits | "원시 모델 출력" | softmax 전의 정규화되지 않은 점수입니다. logistic function에서 이름이 왔습니다 |
| Sampling | "무작위 값 뽑기" | 확률분포에 따라 값을 생성하는 것입니다. 모델이 output을 생성하는 방식입니다 |

## 더 읽을거리

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 평균이 왜 정규분포가 되는지 보여주는 시각적 증명
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 여기의 모든 내용과 그 이상을 다루는 간결한 참고자료
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 수치 안정성이 왜 중요하고 어떻게 달성하는지
