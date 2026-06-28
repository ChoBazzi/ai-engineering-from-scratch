---
name: skill-gradient-computation
description: 일반적인 ML 손실 함수의 그래디언트를 계산하고 적절한 도함수 접근법 선택하기
version: 1.0.0
phase: 1
lesson: 4
tags: [calculus, gradients, backpropagation]
---

# ML을 위한 그래디언트 계산

신경망에서 쓰이는 손실 함수, 활성화 함수, layer 연산의 그래디언트를 계산하기 위한 실용 참고 자료입니다.

## 결정 체크리스트

1. 함수가 단순 primitive(power, exp, log, trig)로 구성되어 있나요? 해석적 도함수와 chain rule을 사용하세요.
2. 함수가 custom 또는 black-box operation인가요? h = 1e-7로 `(f(x+h) - f(x-h)) / (2h)` 수치 미분을 사용하세요.
3. 함수가 PyTorch/JAX의 tensor operations로 만들어졌나요? autograd에 맡기고 numerical check로 검증하세요.
4. 스칼라 loss를 weights 행렬에 대해 미분한 그래디언트가 필요한가요? computation graph를 따라 node 하나씩 chain rule을 적용하세요.
5. non-differentiable operation(argmax, rounding, sampling)이 있나요? straight-through estimator 또는 reparameterization trick을 사용하세요.

## 각 접근법을 사용할 때

| Approach | When to use | Cost |
|---|---|---|
| Analytical (hand-derived) | 단순 함수, autograd output 검증 | runtime 비용 없음 |
| Numerical (finite differences) | 디버깅, gradient checking, black-box functions | n개 파라미터에 2n forward passes |
| Automatic differentiation | 임의의 differentiable computation graph(기본값) | backward pass 한 번 |
| Symbolic (SymPy, Mathematica) | 논문용 closed-form gradients 유도 | compile time only |

## 빠른 참고: 흔한 도함수

| Function | f(x) | f'(x) | ML context |
|---|---|---|---|
| MSE loss | (1/n) sum(y_hat - y)^2 | (2/n)(y_hat - y) | Regression |
| Cross-entropy (binary) | -(y log(p) + (1-y) log(1-p)) | p - y (after sigmoid) | Binary classification |
| Cross-entropy (multi) | -log(p_true_class) | p - one_hot(y) (after softmax) | Multi-class classification |
| Sigmoid | 1 / (1 + e^(-x)) | sigma(x) * (1 - sigma(x)) | Output gates, binary output |
| Tanh | (e^x - e^(-x)) / (e^x + e^(-x)) | 1 - tanh(x)^2 | Hidden activations (legacy) |
| ReLU | max(0, x) | 1 if x > 0, 0 if x < 0 | Default hidden activation |
| Leaky ReLU | max(0.01x, x) | 1 if x > 0, 0.01 if x < 0 | Avoiding dead neurons |
| GELU | x * Phi(x) | Phi(x) + x * phi(x) | Transformers |
| Softmax_i | e^(x_i) / sum(e^(x_j)) | s_i(1 - s_i) for i=j, -s_i*s_j for i!=j | Output layer (Jacobian) |
| Log-softmax | x_i - log(sum(e^(x_j))) | 1 - softmax(x_i) for the i-th entry | Numerically stable CE |
| Linear layer | y = Wx + b | dL/dW = dL/dy * x^T, dL/db = dL/dy | Every layer |
| L2 regularization | lambda * sum(w^2) | 2 * lambda * w | Weight decay |
| L1 regularization | lambda * sum(\|w\|) | lambda * sign(w) | Sparsity |

## 흔한 실수

- batch-averaged losses(MSE, cross-entropy)에서 1/n 계수를 잊는 것. 그래디언트는 batch size만큼 스케일됩니다.
- softmax gradient를 실제로는 Jacobian matrix인데 vector로 계산하는 것. cross-entropy + softmax를 함께 쓰면 그래디언트가 (p - y)로 단순화되어 full Jacobian을 피할 수 있습니다.
- chain rule을 잘못된 순서로 적용하는 것. loss에서 뒤로 작업하세요: dL/dW = dL/dy * dy/dW.
- 수치 도함수에 너무 큰 h(h = 0.1) 또는 너무 작은 h(h = 1e-15)를 쓰는 것. float64에서는 h = 1e-7을 유지하세요.
- ReLU가 정확히 x = 0에서 undefined gradient를 갖는다는 점을 잊는 것. 실제로는 0 또는 0.5로 설정합니다.

## Gradient checking 레시피

```text
For each parameter w:
  numeric_grad = (loss(w + h) - loss(w - h)) / (2h)
  auto_grad = backward pass value
  relative_error = |numeric - auto| / max(|numeric|, |auto|, 1e-8)
  assert relative_error < 1e-5
```

Relative error가 1e-3보다 크면 무언가 잘못된 것입니다. 1e-5와 1e-3 사이이면 조사하세요.
