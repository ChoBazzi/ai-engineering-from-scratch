# 수치 안정성

> Floating point는 새는 추상화입니다. 학습 중 당신을 물 것이고, 당신은 그것이 오는 것을 보지 못할 것입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## 학습 목표

- max-subtraction trick을 사용해 numerically stable softmax와 log-sum-exp 구현하기
- floating-point computation에서 overflow, underflow, catastrophic cancellation 식별하기
- centered finite differences로 analytical gradients를 numerical gradients와 검증하기
- bfloat16이 학습에서 float16보다 선호되는 이유와 loss scaling이 gradient underflow를 막는 방법 설명하기

## 문제

모델이 세 시간 동안 학습하다가 loss가 NaN이 됩니다. print statement를 추가합니다. step 9,000에서는 logits가 멀쩡합니다. step 9,001에서는 `inf`입니다. step 9,002가 되면 모든 gradient가 `nan`이고 학습은 끝났습니다.

또는 모델이 끝까지 학습되지만 accuracy가 논문보다 2% 낮습니다. 모든 것을 확인합니다. Architecture도 맞고, hyperparameters도 맞고, data도 맞습니다. 문제는 논문은 float32를 썼고 당신은 적절한 scaling 없이 float16을 썼다는 것입니다. 32비트의 누적 rounding error가 조용히 accuracy를 갉아먹었습니다.

또는 cross-entropy loss를 처음부터 구현합니다. 작은 logits에서는 작동합니다. logits가 100을 넘으면 `inf`를 반환합니다. `exp(100)`이 float32가 표현할 수 있는 것보다 크기 때문에 softmax가 overflow했습니다. 모든 ML framework는 이 문제를 두 줄짜리 trick으로 처리합니다. 당신은 그 trick이 있다는 것을 몰랐습니다.

Numerical stability는 이론적인 관심사가 아닙니다. 성공하는 training run과 조용히 실패하는 training run의 차이입니다. 결국 디버깅하게 될 심각한 ML bug는 모두 floating point로 내려갑니다.

## 개념

### IEEE 754: 컴퓨터가 실수를 저장하는 방법

컴퓨터는 IEEE 754 표준을 따르는 floating point 값으로 실수를 저장합니다. float는 sign bit, exponent, mantissa(significand)의 세 부분으로 구성됩니다.

```text
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

mantissa는 precision(몇 개의 significant digits)을 결정합니다. exponent는 range(숫자가 얼마나 크거나 작을 수 있는지)를 결정합니다.

```text
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32는 약 7개의 decimal digits precision을 제공합니다. 1.0000001과 1.0000002는 구분할 수 있지만, 1.00000001과 1.00000002는 구분할 수 없습니다. 7자리 이후는 rounding noise입니다.

float16은 약 3자리 precision을 제공합니다. 표현할 수 있는 가장 큰 수는 65,504입니다. logits, gradients, activations가 이 값을 자주 넘는 ML에서는 불안할 정도로 작습니다.

bfloat16은 float16의 range 문제에 대한 Google의 답입니다. float32와 같은 8-bit exponent(같은 range, 최대 3.4e38)를 가지지만 mantissa는 7비트뿐입니다(float16보다 precision이 낮음). neural network 학습에서는 precision보다 range가 더 중요하므로 보통 bfloat16이 이깁니다.

### 왜 0.1 + 0.2 != 0.3인가

0.1은 binary floating point에서 정확히 표현할 수 없습니다. base 2에서는 반복 소수입니다.

```text
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32는 이를 23 mantissa bits로 자릅니다. 저장된 값은 대략 0.100000001490116입니다. 0.2도 대략 0.200000002980232로 저장됩니다. 둘의 합은 0.3이 아니라 0.300000004470348입니다.

```text
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

ML에서 이것이 중요한 이유:

1. `if loss < threshold` 같은 loss comparison이 잘못된 답을 줄 수 있습니다
2. 많은 작은 값을 누적하면(수천 step의 gradient updates) true sum에서 drift합니다
3. float를 `==`로 비교하면 checksum과 reproducibility test가 실패합니다

수정: float를 `==`로 비교하지 마세요. `abs(a - b) < epsilon` 또는 `math.isclose()`를 사용하세요.

### 치명적 소거

거의 같은 두 floating point number를 빼면 significant digits가 상쇄되고 rounding noise가 leading digits로 승격됩니다.

```text
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

단 한 번의 subtraction으로 19% relative error가 생깁니다. ML에서는 다음 상황에서 발생합니다.

- 큰 mean을 가진 data의 variance 계산: E[x]가 클 때 `E[x^2] - E[x]^2`
- 거의 같은 log-probabilities 빼기
- epsilon이 너무 작은 finite-difference gradients 계산

수정: 크고 거의 같은 수를 빼지 않도록 공식을 재배열하세요. variance에는 Welford algorithm을 사용하거나 data를 먼저 center하세요. log-probabilities는 처음부터 끝까지 log-space에서 작업하세요.

### Overflow와 Underflow

Overflow는 결과가 너무 커서 표현할 수 없을 때 발생합니다. Underflow는 너무 작을 때(표현 가능한 가장 작은 양수보다 0에 가까울 때) 발생합니다.

```text
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

ML에서 `exp()` function은 overflow의 주요 원인입니다.

```text
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` function은 반대 방향에서 문제를 만납니다.

```text
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

ML에서 `exp()`는 softmax, sigmoid, probability computation에 나타납니다. `log()`는 cross-entropy, log-likelihoods, KL divergence에 나타납니다. `log(exp(x))` 조합은 올바른 trick 없이는 지뢰밭입니다.

### Log-Sum-Exp trick 설명

`log(sum(exp(x_i)))`를 직접 계산하는 것은 수치적으로 위험합니다. 어떤 `x_i`가 크면 `exp(x_i)`가 overflow합니다. 모든 `x_i`가 매우 음수이면 모든 `exp(x_i)`가 0으로 underflow하고 `log(0)`은 `-inf`입니다.

trick은 exponentiating 전에 최대값을 빼는 것입니다.

```text
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

작동 이유: `max(x)`를 빼면 가장 큰 exponent는 `exp(0) = 1`입니다. overflow가 불가능합니다. 합에는 적어도 하나의 항 1이 있으므로 합은 최소 1이고, `log(1) = 0`입니다. `-inf`로 underflow하지 않습니다.

증명:

```text
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

`c = max(x)`로 두면 overflow가 제거됩니다.

이 trick은 ML 전반에 나타납니다.
- Softmax normalization
- Cross-entropy loss computation
- Sequence model의 log-probability summation
- Mixture of Gaussians
- Variational inference

### Softmax에 max-subtraction trick이 필요한 이유

Softmax는 logits를 probabilities로 변환합니다.

```text
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

trick 없이 logits [100, 101, 102]는 overflow를 일으킵니다.

```text
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

trick을 사용해 max(x) = 102를 뺍니다.

```text
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

probabilities는 동일합니다. computation은 안전합니다. 이것은 optimization이 아니라 correctness의 요구사항입니다.

### NaN과 Inf: 감지와 예방

`nan`(Not a Number)과 `inf`(infinity)는 계산을 통해 전염성 있게 퍼집니다. gradient update 하나에 `nan`이 있으면 weight가 `nan`이 되고, 이후 모든 output이 `nan`이 됩니다. 한 step 안에 학습이 죽습니다.

`inf`가 나타나는 방식:
- 큰 양수에 대한 `exp()`
- 0으로 나누기: `1.0 / 0.0`
- accumulation에서 `float32` overflow

`nan`이 나타나는 방식:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 음수의 `sqrt()`
- 음수의 `log()`
- 기존 `nan`이 포함된 모든 산술

감지:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

예방 전략:

1. `exp()` 입력 clamp: `exp(clamp(x, -80, 80))`
2. denominator에 epsilon 추가: `x / (y + 1e-8)`
3. `log()` 안에 epsilon 추가: `log(x + 1e-8)`
4. stable implementations 사용(log-sum-exp, stable softmax)
5. weight explosion을 막기 위한 gradient clipping
6. debugging 중 모든 forward pass 후 `nan`/`inf` 확인

### 수치적 gradient checking

Analytical gradients(backpropagation에서 나온 것)는 버그를 가질 수 있습니다. Numerical gradient checking은 finite differences로 gradients를 계산해 이를 검증합니다.

중심 차분 공식:

```text
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

이는 O(h^2) 정확도이며, O(h)뿐인 forward difference `(f(x+h) - f(x)) / h`보다 훨씬 좋습니다.

h 선택: 너무 크면 approximation이 틀립니다. 너무 작으면 catastrophic cancellation이 답을 망칩니다. `h = 1e-5`에서 `1e-7`이 일반적입니다.

check: analytical gradient와 numerical gradient 사이의 relative difference를 계산합니다.

```text
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

경험칙:
- relative_error < 1e-7: 완벽, gradient가 맞음
- relative_error < 1e-5: 허용 가능, 아마 맞음
- relative_error > 1e-3: 무언가 틀림
- relative_error > 1: gradient가 완전히 틀림

새 layer나 loss function을 구현할 때는 항상 gradient를 확인하세요. PyTorch는 이를 위해 `torch.autograd.gradcheck()`를 제공합니다.

### Mixed precision training 설명

현대 GPU에는 float16 matrix multiplication을 float32보다 2-8x 빠르게 계산하는 specialized hardware(Tensor Cores)가 있습니다. Mixed precision training은 이를 활용합니다.

```text
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

pure float16 training의 문제: gradients는 종종 매우 작습니다(1e-8 또는 더 작음). Float16은 ~6e-8 아래를 0으로 underflow합니다. 모든 gradient update가 0이 되어 모델이 더 이상 학습하지 않습니다.

수정은 loss scaling입니다.

```text
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling은 scale factor를 자동으로 조정합니다. 큰 값(65536)으로 시작합니다. gradients가 `inf`로 overflow하면 절반으로 줄입니다. N steps 동안 overflow가 없으면 두 배로 늘립니다.

### bfloat16 vs float16: 왜 bfloat16이 학습에서 이기는가

```text
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16은 precision이 더 높지만(10 mantissa bits vs 7) range가 제한적입니다(max ~65,504). bfloat16은 precision이 더 낮지만 float32와 같은 range를 가집니다(max ~3.4e38).

neural network 학습에서는:

- Activations와 logits가 training spike 중 65,504를 자주 넘습니다. float16은 overflow하지만 bfloat16은 처리합니다.
- float16에는 loss scaling이 필요하지만 bfloat16은 range가 gradient magnitude spectrum을 덮기 때문에 보통 필요 없습니다.
- bfloat16은 float32의 단순 truncation입니다. mantissa의 하위 16 bits를 버립니다. 변환이 단순하고 exponent에서는 lossless입니다.

float16은 값이 bounded되고 precision이 더 중요한 inference에 선호됩니다. bfloat16은 range가 더 중요한 training에 선호됩니다. 그래서 TPU와 현대 NVIDIA GPU(A100, H100)는 native bfloat16 support를 가집니다.

### Gradient clipping 설명

Exploding gradients는 많은 layer를 지나며 gradients가 지수적으로 커질 때 발생합니다(RNNs, deep networks, transformers에서 흔함). 하나의 큰 gradient가 한 step에서 모든 weights를 망칠 수 있습니다.

두 가지 clipping:

**Clip by value:** 각 gradient element를 독립적으로 clamp합니다.

```text
grad = clamp(grad, -max_val, max_val)
```

단순하지만 gradient vector의 방향을 바꿀 수 있습니다.

**Clip by norm:** 전체 gradient vector를 scale해 norm이 threshold를 넘지 않게 합니다.

```text
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

gradient의 방향을 보존합니다. 이것이 `torch.nn.utils.clip_grad_norm_()`가 하는 일입니다. 표준 선택입니다.

일반적인 값: transformers는 `max_norm=1.0`, RL은 `max_norm=0.5`, 더 단순한 network는 `max_norm=5.0`.

Gradient clipping은 hack이 아닙니다. safety mechanism입니다. 없으면 outlier batch 하나가 몇 주의 training을 망칠 만큼 큰 gradient를 만들 수 있습니다.

### 수치 안정화 장치로서의 normalization layers

Batch normalization, layer normalization, RMS normalization은 보통 training convergence를 돕는 regularizers로 소개됩니다. 이들은 numerical stabilizers이기도 합니다.

normalization이 없으면 activations가 layers를 지나며 지수적으로 커지거나 작아질 수 있습니다.

```text
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization은 각 layer에서 activations를 recenter하고 rescale합니다.

```text
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`(보통 1e-5)은 모든 activation이 같을 때 division by zero를 막습니다. learned parameters `gamma`와 `beta`는 network가 필요한 scale을 복원할 수 있게 합니다.

이는 network 전체에서 값을 수치적으로 안전한 range에 유지해 forward pass의 overflow와 backward pass의 gradient explosion을 모두 막습니다.

### 흔한 ML numerical bugs

**Bug: Loss가 몇 epoch 후 NaN이 됩니다.**
Cause: logits가 너무 커져 softmax가 overflow했습니다. 또는 learning rate가 너무 높아 weights가 diverge했습니다.
Fix: stable softmax(max subtraction)를 사용하고, learning rate를 줄이며, gradient clipping을 추가하세요.

**Bug: Loss가 log(num_classes)에 고정됩니다.**
Cause: model outputs가 거의 uniform probabilities입니다. 보통 gradients가 vanishing하거나 모델이 전혀 학습하지 않는다는 뜻입니다.
Fix: data labels가 올바른지 확인하고, loss function을 검증하며, dead ReLUs를 확인하세요.

**Bug: Validation accuracy가 기대보다 1-3% 낮습니다.**
Cause: proper loss scaling 없는 mixed precision. Gradient underflow가 작은 update를 조용히 0으로 만듭니다.
Fix: dynamic loss scaling을 활성화하거나 bfloat16으로 전환하세요.

**Bug: 일부 layer의 gradient norms가 0.0입니다.**
Cause: dead ReLU neurons(모든 input이 negative), 또는 float16 underflow.
Fix: LeakyReLU 또는 GELU를 사용하고, gradient scaling을 사용하며, weight initialization을 확인하세요.

**Bug: 모델이 한 GPU에서는 작동하지만 다른 GPU에서는 다른 결과를 냅니다.**
Cause: non-deterministic floating point accumulation order. GPU parallel reductions는 hardware마다 다른 순서로 합산하고, floating point addition은 associative가 아닙니다.
Fix: 작은 차이(1e-6)를 받아들이거나 `torch.use_deterministic_algorithms(True)`를 설정하고 speed penalty를 감수하세요.

**Bug: loss computation에서 `exp()`가 `inf`를 반환합니다.**
Cause: max-subtraction trick 없이 raw logits가 `exp()`에 전달되었습니다.
Fix: 내부적으로 log-sum-exp를 구현하는 `torch.nn.functional.log_softmax()`를 사용하세요.

**Bug: float32에서 float16으로 바꾼 뒤 training이 diverge합니다.**
Cause: float16은 6e-8보다 작은 gradient magnitude나 65,504보다 큰 activation을 표현할 수 없습니다.
Fix: loss scaling이 있는 mixed precision(AMP)을 사용하거나 bfloat16을 사용하세요.

```figure
logsumexp-stability
```

## 직접 만들기

### Step 1: floating point precision limit 시연

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: naive vs stable softmax 구현

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Step 3: stable log-sum-exp 구현

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Step 4: stable cross-entropy 구현

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Step 5: Gradient checking 구현

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 사용하기

### Mixed precision simulation 구현

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping 구현

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection 구현

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

모든 edge case가 포함된 complete implementations는 `code/numerical.py`를 보세요.

## 산출물

이 lesson은 다음을 만듭니다.
- `code/numerical.py`: stable softmax, log-sum-exp, cross-entropy, gradient checking, mixed precision simulation
- `outputs/prompt-numerical-debugger.md`: training 중 NaN/Inf 및 numerical issues를 진단하는 prompt

이 stable implementations는 Phase 3에서 training loop를 만들 때와 Phase 4에서 attention mechanisms를 구현할 때 다시 등장합니다.

## 연습문제

1. **Catastrophic cancellation.** float32에서 naive formula `E[x^2] - E[x]^2`로 [1000000.0, 1000001.0, 1000002.0]의 variance를 계산하세요. 그런 다음 Welford's online algorithm으로 계산하세요. true variance(0.6667)와 error를 비교하세요.

2. **Precision hunt.** Python에서 `1.0 + x == 1.0`이 되는 가장 작은 positive float32 값 `x`를 찾으세요. 이것이 machine epsilon입니다. `numpy.finfo(numpy.float32).eps`와 일치하는지 확인하세요.

3. **Log-sum-exp edge cases.** `logsumexp_stable` 함수를 (a) 모든 값이 같음, (b) 하나의 값이 나머지보다 훨씬 큼, (c) 모든 값이 매우 음수(-1000)인 경우로 테스트하세요. naive version이 실패하는 곳에서 올바른 결과를 주는지 확인하세요.

4. **Gradient checking a neural network layer.** 단일 linear layer `y = Wx + b`와 analytical backward pass를 구현하세요. `numerical_gradient`로 3x2 weight matrix에 대한 correctness를 검증하세요.

5. **Loss scaling experiment.** float16 학습을 simulation하세요. [1e-9, 1e-3] 범위의 random gradients를 만들고 float16으로 변환해 몇 퍼센트가 zero가 되는지 측정하세요. 그런 다음 loss scaling(1024 곱하기)을 적용하고 float16으로 변환한 뒤 다시 scale back하여 zero fraction을 다시 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| IEEE 754 | "float standard" | binary floating point formats, rounding rules, special values(inf, nan)를 정의하는 국제 표준입니다. 모든 현대 CPU와 GPU가 구현합니다. |
| Machine epsilon | "precision limit" | 주어진 float format에서 1.0 + e != 1.0이 되는 가장 작은 값 e입니다. float32에서는 약 1.19e-7입니다. |
| Catastrophic cancellation | "subtraction으로 인한 precision loss" | 거의 같은 floating point number를 뺄 때 significant digits가 상쇄되고 rounding noise가 결과를 지배합니다. |
| Overflow | "숫자가 너무 큼" | 결과가 표현 가능한 최댓값을 넘어 inf가 됩니다. exp(89)는 float32에서 overflow합니다. |
| Underflow | "숫자가 너무 작음" | 결과가 표현 가능한 가장 작은 양수보다 0에 가까워 0.0이 됩니다. exp(-104)는 float32에서 underflow합니다. |
| Log-sum-exp trick | "먼저 max를 빼기" | overflow와 underflow를 막기 위해 exp(max(x))를 factor out하여 log(sum(exp(x)))를 계산하는 방법입니다. softmax, cross-entropy, log-probability math에 사용됩니다. |
| Stable softmax | "폭발하지 않는 softmax" | exponentiating 전에 max(logits)를 빼는 것입니다. 결과는 수치적으로 동일하고 overflow가 불가능합니다. |
| Gradient checking | "backprop 검증" | backpropagation의 analytical gradients를 finite differences의 numerical gradients와 비교해 구현 버그를 잡는 방법입니다. |
| Mixed precision | "Float16 forward, float32 backward" | speed-critical operations에는 낮은 precision float를, numerically sensitive operations에는 높은 precision float를 사용합니다. 일반적인 speedup은 2-3x입니다. |
| Loss scaling | "gradient underflow 방지" | backprop 전에 loss에 큰 상수를 곱해 gradients를 float16의 표현 범위 안에 두고, weight update 전에 같은 상수로 나눕니다. |
| bfloat16 | "Brain floating point" | 8 exponent bits(float32와 같은 range)와 7 mantissa bits(float16보다 낮은 precision)를 가진 Google의 16-bit format입니다. training에 선호됩니다. |
| Gradient clipping | "gradient norm 제한" | gradient vector의 norm이 threshold를 넘지 않도록 scale합니다. exploding gradients가 weights를 망치는 것을 막습니다. |
| NaN | "Not a Number" | undefined operations(0/0, inf-inf, sqrt(-1))에서 생기는 special float value입니다. 이후 모든 arithmetic에 전파됩니다. |
| Inf | "Infinity" | overflow 또는 division by zero에서 생기는 special float value입니다. 결합되어 NaN을 만들 수 있습니다(inf - inf, inf * 0). |
| Numerical gradient | "brute force derivative" | f(x+h)와 f(x-h)를 평가하고 2h로 나누어 derivative를 근사합니다. 느리지만 검증에는 신뢰할 수 있습니다. |

## 더 읽을거리

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- definitive reference, dense but complete
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- float16 training을 위한 loss scaling을 소개한 NVIDIA paper
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch의 mixed precision 실무 가이드
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google이 TPU에 이 format을 선택한 이유
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- floating point sums의 rounding error를 줄이는 algorithm
