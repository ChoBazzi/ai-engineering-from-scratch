---
name: skill-convexity-checker
description: optimization problem이 convex인지 판단하고 적절한 solver 선택
version: 1.0.0
phase: 1
lesson: 18
tags: [optimization, convexity, solvers]
---

# Convexity checker(볼록성 검사기)

optimization problem이 convex인지 확인하는 방법과 그 답을 바탕으로 무엇을 해야 하는지 설명합니다.

## 결정 체크리스트

1. objective function이 convex인가요? (Hessian positive semi-definiteness를 확인하거나 composition rules를 사용하세요.)
2. 모든 inequality constraints가 g_i(x) <= 0 형태이고 각 g_i가 convex인가요?
3. 모든 equality constraints가 affine(linear)인가요?
4. 세 가지가 모두 yes이면 problem은 convex입니다. convergence guarantees가 있는 convex solver를 사용하세요.
5. 하나라도 no이면 problem은 non-convex입니다. SGD/Adam을 사용하고 local optima를 받아들이세요.

## function의 convexity를 테스트하는 방법

| Test | Applies to | Method |
|---|---|---|
| Second derivative >= 0 | Scalar functions f(x) | f''(x)를 계산합니다. 모든 x에서 f''(x) >= 0이면 convex입니다. |
| Hessian is PSD | Multivariate functions f(x) | H(x)를 계산합니다. 모든 곳에서 모든 eigenvalues >= 0이면 convex입니다. |
| Definition test | Any function | sampled x, y, t에 대해 f(tx + (1-t)y) <= t*f(x) + (1-t)*f(y)를 확인합니다. |
| Composition rules | Composed functions | 아래 composition table을 보세요. |
| Restriction to a line | Multivariate f | 모든 x, v에 대해 g(t) = f(x + tv)가 t에 대해 convex일 때 그리고 그때만 f는 convex입니다. |

## Composition rules (convexity 보존)

| Operation | Result |
|---|---|
| f + g (both convex) | Convex |
| c * f (c > 0, f convex) | Convex |
| max(f, g) (both convex) | Convex |
| f(Ax + b) where f is convex | Convex |
| g(f(x)) where g is convex non-decreasing and f is convex | Convex |
| g(f(x)) where g is convex non-increasing and f is concave | Convex |
| sum of convex functions | Convex |
| pointwise supremum of convex functions | Convex |

## 흔한 ML objectives: convex인가 아닌가?

| Objective | Convex? | Reason |
|---|---|---|
| MSE: (1/n) sum(y - Xw)^2 | Yes | w에 대한 quadratic, Hessian = (2/n) X^T X는 PSD |
| Logistic loss: sum(log(1 + exp(-y_i * w^T x_i))) | Yes | convex functions의 sum(log-sum-exp family) |
| Hinge loss: sum(max(0, 1 - y_i * w^T x_i)) | Yes | convex(linear) functions의 max |
| L2 regularization: lambda * \|\|w\|\|^2 | Yes | Quadratic, Hessian = 2*lambda*I |
| L1 regularization: lambda * \|\|w\|\|_1 | Yes | absolute values의 sum(convex지만 not differentiable) |
| Ridge regression: MSE + L2 | Yes | 두 convex functions의 sum |
| LASSO: MSE + L1 | Yes | 두 convex functions의 sum |
| Elastic net: MSE + L1 + L2 | Yes | convex functions의 sum |
| SVM (primal): hinge + L2 | Yes | convex functions의 sum |
| Cross-entropy with softmax | Yes (in logits) | Log-sum-exp는 convex |
| Neural network (any loss) | No | Nonlinear activations가 non-convex composition을 만듦 |
| k-means objective | No | Discrete assignment step |
| Matrix factorization: \|\|X - UV^T\|\|^2 | No | U와 V에 대해 bilinear |
| GAN loss | No | Minimax, generator에 대해 non-convex |
| Contrastive loss (InfoNCE) | No | negative samples가 있는 exponentials ratio의 log |

## convexity에 따른 solver 선택

| Problem type | Solver | Convergence guarantee |
|---|---|---|
| Convex, smooth, unconstrained | Gradient descent | global minimum까지 O(1/k) |
| Convex, smooth, unconstrained | L-BFGS | global minimum까지 superlinear |
| Convex, smooth, unconstrained | Newton's method | minimum 근처에서 quadratic(Hessian이 tractable이면) |
| Convex, smooth, constrained | Interior point method | Polynomial time |
| Convex, non-smooth (L1) | Proximal gradient / ISTA | global minimum까지 O(1/k) |
| Convex, non-smooth (L1) | ADMM | Flexible, constraints 처리 |
| Convex, quadratic | Conjugate gradient | n steps에서 exact |
| Non-convex, smooth | SGD / Adam | local minimum으로 수렴 |
| Non-convex, smooth | SGD + restarts | 평균적으로 더 나은 local minimum |
| Non-convex, smooth | Overparameterize + SGD | Flat minima, good generalization |

## 흔한 실수

- loss function이 convex라는 이유만으로 problem이 convex라고 가정함. loss는 최적화하는 parameters에 대해 convex여야 합니다. Cross-entropy는 logits에 대해 convex이지만, inputs에서 logits로 가는 전체 neural network mapping은 non-convex입니다.
- non-convex problem에 Newton's method를 사용함. Hessian에 negative eigenvalues가 있을 수 있어 Newton이 minima가 아니라 saddle points나 maxima 쪽으로 움직일 수 있습니다.
- L1 regularization이 objective를 zero에서 non-differentiable하게 만든다는 점을 잊음. Standard gradient descent는 잘 작동하지 않습니다. proximal gradient descent나 subgradient methods를 사용하세요.
- A^T A를 형성해 condition number를 제곱함. least-squares problem을 풀어야 하고 A가 ill-conditioned이면 normal equations 대신 QR 또는 SVD를 사용하세요.
- 확인하지 않고 problem을 non-convex라고 선언함. 많은 ML problems(linear models, SVMs, logistic regression)는 convex이며 더 강한 solvers의 이점을 얻습니다.

## 빠른 테스트: 내 problem은 convex인가?

```text
1. Write out the objective: minimize f(w) subject to constraints
2. For each term in f(w):
   - Is it quadratic with PSD matrix? -> Convex
   - Is it a norm? -> Convex
   - Is it log-sum-exp? -> Convex
   - Does it involve w nonlinearly (sigmoid(w), w1*w2)? -> Likely non-convex
3. Are all constraints linear or convex inequalities?
4. If ALL terms are convex and constraints are convex/linear -> problem is convex
5. If ANY term is non-convex -> problem is non-convex
```
