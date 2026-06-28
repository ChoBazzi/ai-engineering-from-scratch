---
name: skill-regression
description: data characteristics와 problem constraints에 따라 올바른 regression 접근법 선택하기
version: 1.0.0
phase: 2
lesson: 2
tags: [regression, linear-regression, polynomial-regression, ridge, regularization]
---

# Regression 전략 가이드

Regression은 continuous values를 예측합니다. 올바른 접근법은 features와 target 사이의 관계, features 수, overfitting 위험에 따라 달라집니다.

## 의사결정 체크리스트

1. features와 target 사이의 관계가 대략 linear인가요?
   - 예: ordinary linear regression으로 시작
   - 아니요: polynomial features 또는 nonlinear model 시도

2. samples에 비해 features가 얼마나 많나요?
   - features가 적고 samples가 많음: ordinary linear regression이 잘 동작
   - features가 많고 samples가 적음: regularization (Ridge 또는 Lasso) 사용
   - samples보다 features가 많음: feature selection에는 Lasso (L1), 모든 weights 축소에는 Ridge (L2)

3. interpretability가 필요한가요?
   - 예: features가 적은 linear regression, 또는 automatic feature selection을 위한 Lasso
   - 아니요: polynomial features, 또는 tree-based models나 neural networks로 이동

4. dataset이 작은가요 (10,000 rows 미만)?
   - 속도를 위해 normal equation (closed-form solution) 사용
   - 신뢰할 수 있는 evaluation을 위해 cross-validation이 필수

5. dataset이 큰가요 (수백만 rows)?
   - stochastic gradient descent (SGD) 또는 mini-batch gradient descent 사용
   - normal equation은 O(n^3) matrix inversion 때문에 너무 느림

## 각 접근법을 사용할 때

**Ordinary Linear Regression**: 모든 regression task의 baseline입니다. 여기서 시작하세요. R-squared가 허용 가능하고 model이 단순하면 여기서 멈추세요.

**Polynomial Regression**: scatter plot이 line이 아니라 curve를 보입니다. degree 2로 시작하세요. validation performance로 정당화될 때만 높이세요. Degree > 5는 거의 항상 overfits합니다.

**Ridge Regression (L2)**: correlated features가 많을 때 사용합니다. 모든 weights가 zero 쪽으로 shrink되지만 정확히 zero가 되지는 않습니다. 모든 features가 기여한다고 믿을 때 좋습니다.

**Lasso Regression (L1)**: features가 많고 그중 일부만 중요하다고 의심될 때 사용합니다. Lasso는 관련 없는 feature weights를 정확히 zero로 만들어 automatic feature selection을 수행합니다.

**Elastic Net**: L1과 L2 penalties를 결합합니다. correlated features가 많고 일부 feature selection도 원할 때 사용하세요.

## 흔한 실수

- gradient descent 전에 feature scaling을 건너뛰기 (convergence가 매우 느려짐)
- hyperparameters를 조정하는 데 test set performance 사용하기 (validation set 또는 cross-validation 사용)
- validation error를 확인하지 않고 high-degree polynomials fitting하기 (training R^2는 degree가 올라갈수록 항상 증가)
- residual plots 무시하기 (residuals에 patterns가 있으면 R^2가 misleading할 수 있음)
- R^2를 유일한 metric으로 취급하기 (residual distribution, MAE, domain-specific thresholds 확인)

## 빠른 참조

| Method | When to use | Regularization | Feature selection |
|--------|------------|---------------|-------------------|
| OLS | Baseline, few features | None | Manual |
| Ridge | Many features, all relevant | L2 (shrink) | No |
| Lasso | Many features, few relevant | L1 (zero out) | Automatic |
| Elastic Net | Many correlated features | L1 + L2 | Partial |
| Polynomial | Nonlinear relationship | Add Ridge/Lasso on top | Manual degree choice |
