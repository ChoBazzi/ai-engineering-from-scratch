---
name: skill-evaluation
description: classification 및 regression 모델을 위한 evaluation strategy 체크리스트
version: 1.0.0
phase: 2
lesson: 9
tags: [evaluation, metrics, cross-validation, model-selection]
---

# Model evaluation 전략

모든 ML 모델을 올바르게 평가하기 위한 체크리스트입니다. 가장 흔한 evaluation 실수를 피하려면 이 순서를 따르세요.

## Step 1: 데이터를 올바르게 나누기

- preprocessing(scaling, imputation, encoding) 전에 split하세요
- classification task에는 stratified split을 사용하세요
- 마지막에 정확히 한 번만 사용할 test set을 남겨 두세요
- 작은 데이터셋에는 단일 split 대신 5-fold 또는 10-fold cross-validation을 사용하세요
- time series에는 time-based split을 사용하세요(절대 shuffle하지 않음)

## Step 2: 올바른 metric 고르기

### 분류

| 상황 | 사용할 metric | 이유 |
|-----------|----------------|-----|
| 균형 잡힌 class, 단순 비교 | Accuracy | 해석하기 쉽고 class가 같을 때 의미 있음 |
| false positive 비용이 큼(spam filter, fraud alerts) | Precision | flag된 항목 중 실제 positive가 얼마나 되는지 측정 |
| false negative 비용이 큼(cancer screening, security) | Recall | 실제 positive 중 얼마나 잡아냈는지 측정 |
| precision과 recall의 균형 필요 | F1 Score | 조화평균이며 극단적 불균형에 페널티 부여 |
| threshold 전반에서 모델 비교 | AUC-ROC | threshold와 무관한 ranking 품질 |
| 불균형 데이터 | F1, AUC-ROC, or PR-AUC | class가 불균형하면 accuracy가 오해를 부름 |

### 회귀

| 상황 | 사용할 metric | 이유 |
|-----------|----------------|-----|
| 표준 regression, outlier 허용 | RMSE | target과 같은 단위이며 큰 error에 페널티 부여 |
| outlier에 robust한 평가 | MAE | 모든 error를 동일하게 다루며 outlier에 지배되지 않음 |
| 서로 다른 scale의 모델 비교 | R-squared | 정규화된 0-1 scale(설명된 variance 비율) |
| 비즈니스가 금액 단위를 요구 | MAE or RMSE | error magnitude로 직접 해석 가능 |

## Step 3: baseline 세우기

모델을 평가하기 전에 baseline 성능을 계산하세요.
- Classification: majority class predictor(항상 가장 흔한 class 예측)
- Regression: 항상 training target의 mean 예측
- 이 baseline을 넘지 못하는 모델은 학습하고 있지 않은 것입니다

## Step 4: Cross-validation 수행

- 안정적인 추정을 위해 K-fold(K=5 또는 K=10)를 사용하세요
- classification에는 stratified K-fold를 사용하세요
- fold 전체의 mean과 standard deviation을 보고하세요
- mean=0.85, std=0.02인 모델이 mean=0.87, std=0.10인 모델보다 더 신뢰할 만합니다

## Step 5: 모델을 통계적으로 비교하기

- significance를 확인하지 않고 평균 점수가 가장 높은 모델을 고르지 마세요
- cross-validation fold 전반에 paired t-test를 사용하세요
- |t| < 2.78(K=5, df=4, p<0.05)라면 차이가 우연 때문일 수 있습니다
- 성능 차이가 significant하지 않다면 더 단순한 모델을 고려하세요

## Step 6: 흔한 실수 확인하기

- Data leakage: test data 정보가 training으로 흘러 들어갔나요?(split 전 scaling, target-derived features)
- Class imbalance: accuracy가 낮은 minority-class 성능을 숨기고 있나요?
- Overfitting: training 성능과 validation 성능의 차이가 큰가요?
- Too many evaluations: test set을 한 번 넘게 보았나요?

## Step 7: 최종 성능 보고하기

- train + validation을 합쳐 학습하세요
- held-out test set에서 정확히 한 번 평가하세요
- 가능하면 선택한 metric을 confidence interval과 함께 보고하세요
- baseline 비교를 명시하세요(random/mean보다 얼마나 나은지)
