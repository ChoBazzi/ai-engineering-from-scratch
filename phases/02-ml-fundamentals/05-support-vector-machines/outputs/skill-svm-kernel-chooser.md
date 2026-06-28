---
name: skill-svm-kernel-chooser
description: Choose the right SVM kernel and tune C and gamma for your problem
version: 1.0.0
phase: 2
lesson: 5
tags: [svm, kernel, classification, hyperparameter-tuning]
---

# SVM Kernel 선택 가이드

SVM은 두 가지 선택으로 정의됩니다. kernel(decision boundary의 shape을 결정)과 regularization parameter(margin width와 classification error 사이의 tradeoff를 제어)입니다. 이를 제대로 고르는 것이 쓸모없는 model과 강한 model을 가릅니다.

## 결정 체크리스트

1. data가 linearly separable이거나 그에 가까운가요?
   - Yes: linear kernel을 사용하세요. 더 빠르고 해석하기 쉽습니다.
   - No: step 2로 가세요.

2. feature와 sample은 각각 얼마나 많나요?
   - Features >> samples(예: TF-IDF가 있는 text): linear kernel을 사용하세요. High-dimensional data는 종종 linearly separable입니다. RBF는 이득 없이 complexity만 더합니다.
   - Samples >> features(예: feature 10-50개의 tabular data): RBF kernel이 기본 선택입니다.

3. decision boundary가 smooth할 것으로 예상되나요?
   - Smooth, continuous boundary: RBF kernel
   - Polynomial-shaped boundary: polynomial kernel(degree 2 또는 3부터 시작)
   - Domain knowledge가 특정 interaction term을 시사함: matching degree의 polynomial kernel

4. dataset은 얼마나 큰가요?
   - 10,000 sample 미만: 어떤 kernel도 작동하며, RBF가 안전한 기본값입니다
   - 10,000 to 100,000: linear kernel 또는 LinearSVC(primal formulation, epoch당 O(n))
   - 100,000 초과: kernel SVM을 사용하지 마세요. linear SVM, gradient boosting, neural network로 전환하세요.

5. feature를 scale했나요?
   - SVM은 feature scaling이 필요합니다. fit하기 전에 항상 standardize(zero mean, unit variance)하세요. Unscaled feature는 margin geometry를 왜곡합니다.

## Kernel 선택 flowchart

```text
Start
  |
  v
Features > 1000 or features >> samples?
  Yes --> Linear kernel (LinearSVC for speed)
  No  --> Dataset < 10k samples?
            Yes --> Try RBF first (best general-purpose kernel)
            No  --> Linear kernel (kernel SVMs are O(n^2) to O(n^3))
```

RBF가 잘 작동하지 않으면 polynomial degree 2-3을 시도하세요. 그것도 실패하면 문제 자체가 SVM에 적합하지 않을 수 있습니다.

## C tuning (regularization)

C는 misclassification에 대한 penalty를 제어합니다. regularization strength와 반비례합니다.

| C value | Effect | When to use |
|---------|--------|-------------|
| 0.001 - 0.01 | Wide margin, 많은 violation 허용 | Noisy data, generalization을 원할 때 |
| 0.1 - 1.0 | Balanced | 좋은 시작 범위 |
| 10 - 1000 | Narrow margin, 적은 violation | Clean data, 높은 accuracy가 필요할 때 |

Tuning strategy:
- C=1.0에서 시작하세요
- log scale로 search하세요: [0.001, 0.01, 0.1, 1, 10, 100, 1000]
- cross-validation으로 best value를 고르세요
- best C가 range의 edge에 있으면 그 방향으로 range를 확장하세요

## gamma tuning (RBF kernel)

Gamma는 단일 training point의 influence가 얼마나 멀리 도달하는지 제어합니다. Gaussian의 width를 정의합니다.

| gamma value | Effect | When to use |
|-------------|--------|-------------|
| Small (0.001) | 각 point가 큰 영역에 영향. Smooth, simple boundary | Underfitting 또는 feature가 적을 때 |
| Medium (auto: 1/n_features) | sklearn default. 합리적인 시작점 | General use |
| Large (10+) | 각 point가 가까운 point에만 영향. Complex, wiggly boundary | Overfitting risk |

Tuning strategy:
- gamma="scale"(1 / (n_features * X.var()), sklearn default)에서 시작하세요
- log scale로 search하세요: [0.001, 0.01, 0.1, 1, 10]
- Low gamma + high C는 overfit되는 경향이 있습니다
- High gamma + low C는 underfit되는 경향이 있습니다

## C와 gamma joint tuning

C와 gamma는 상호작용합니다. 독립적으로가 아니라 항상 함께 tuning하세요.

Recommended approach:
1. Coarse grid search: C in [0.01, 0.1, 1, 10, 100], gamma in [0.001, 0.01, 0.1, 1, 10] (25 combos)
2. best region을 찾으세요
3. best region 주변에서 fine grid search를 수행하세요(예: C in [5, 10, 20, 50], gamma in [0.05, 0.1, 0.2])
4. 전체 과정에서 5-fold cross-validation을 사용하세요

## 흔한 실수

- high-dimensional sparse data에 RBF kernel 사용(linear가 더 낫고 100x 빠름)
- feature scaling을 잊음(가장 흔한 SVM 실수)
- noisy data에서 C를 너무 높게 설정(noise를 학습하는 대신 암기함)
- 50k sample을 넘는 dataset에서 kernel SVM 사용(training time이 prohibitive)
- C와 gamma를 함께 tuning하지 않음(서로 보상하기 때문)
- polynomial degree 5+를 기본으로 사용(공격적으로 overfit됨, 먼저 2 또는 3을 시도)

## 빠른 참조

| Kernel | When to use | Key parameters | Training complexity |
|--------|------------|----------------|-------------------|
| Linear | Text/TF-IDF, 많은 feature, 큰 data | C only | epoch당 O(n) |
| RBF | General-purpose, 10k sample 미만 | C, gamma | O(n^2) to O(n^3) |
| Polynomial | 알려진 polynomial relationship | C, degree, coef0 | O(n^2) to O(n^3) |
| Sigmoid | 거의 유용하지 않음(two-layer neural net과 동등) | C, gamma, coef0 | O(n^2) to O(n^3) |
