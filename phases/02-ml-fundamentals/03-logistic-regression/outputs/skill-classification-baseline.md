---
name: skill-classification-baseline
description: complex models로 넘어가기 전에 강력한 classification baseline 세우기
version: 1.0.0
phase: 2
lesson: 3
tags: [classification, logistic-regression, baseline, preprocessing]
---

# Classification baseline 가이드

complex models를 시도하기 전에 logistic regression으로 baseline을 세우세요. 몇 초 안에 train되고, probabilities를 만들며, 완전히 interpretable합니다. 실제 문제 중 놀랄 만큼 많은 수가 그보다 더 복잡한 것을 필요로 하지 않습니다.

## 의사결정 체크리스트

1. decision boundary가 linear일 가능성이 높은가요?
   - 예: logistic regression으로 충분할 가능성이 큼
   - 아니요: 그래도 improvement를 측정할 baseline으로 필요함

2. features가 몇 개인가요?
   - 50 미만: standard logistic regression이 잘 동작
   - 50 to 10,000: L2 regularization (Ridge) 추가
   - 10,000 초과 (예: TF-IDF text features): L1 regularization (Lasso) 또는 LinearSVC 사용

3. dataset이 imbalanced인가요?
   - 5:1 ratio 미만: adjustment 없이도 대체로 괜찮음
   - 5:1 to 50:1: sklearn에서 `class_weight="balanced"` 사용
   - 50:1 초과: class weighting과 적절한 metric (precision, recall, 또는 F1)을 결합

4. features가 서로 다른 scales에 있나요?
   - logistic regression 전에 항상 standardize하세요. gradient-based optimization을 사용하므로 unscaled features는 convergence를 늦추거나 decision boundary를 왜곡합니다.

5. missing values가 있나요?
   - fitting 전에 impute하세요. Logistic regression은 NaNs를 처리할 수 없습니다.
   - numeric columns에는 median imputation, categorical에는 mode를 사용하세요.

## logistic regression으로 충분할 때

- mostly linear feature relationships를 가진 binary classification
- probability outputs가 필요함 (class labels만이 아님)
- interpretability가 필요함 (standardization 후 coefficients가 feature importance의 방향과 상대 크기를 나타냄)
- training data가 작음 (수백에서 낮은 수천 samples)
- real-time serving을 위한 빠른 model이 필요함 (inference에서 single dot product)
- regulatory 또는 compliance requirements가 explainability를 요구함

## upgrade할 때

- feature engineering을 시도했는데도 accuracy가 target보다 훨씬 낮게 plateau됨
- features와 target 사이의 관계가 명확히 nonlinear임 (residual plots 확인)
- large tabular data가 있음 (10k+ rows): gradient boosting (XGBoost 또는 LightGBM) 시도
- polynomial features가 포착할 수 없는 complex interactions가 있음
- image, text, 또는 sequential data가 있음: raw inputs에 대한 logistic regression은 작동하지 않음

## classification baseline을 위한 preprocessing steps

1. preprocessing 전에 먼저 **Train/test split**. 이것이 data leakage를 방지합니다.
2. **Handle missing values**: numeric은 median impute, categorical은 mode impute.
3. **Encode categoricals**: low cardinality (10 values 미만)는 one-hot, 더 높으면 target encoding. target encoding은 training folds에만 fit하세요 (leakage 방지를 위해 out-of-fold encoding 사용).
4. **Scale numerics**: StandardScaler (zero mean, unit variance). train에 fit하고 둘 다 transform.
5. `C=1.0` (default regularization)으로 **Fit logistic regression**.
6. **Evaluate**: confusion matrix, precision, recall, F1. accuracy만 보지 마세요.
7. **Tune threshold**: default 0.5가 최적인 경우는 드뭅니다. 0.1에서 0.9까지 sweep하고 precision/recall 우선순위에 맞는 threshold를 고르세요.

## 흔한 실수

- imbalanced data에서 accuracy만 evaluate하기 (majority class를 예측하는 model이 높은 점수를 받지만 쓸모없음)
- features scaling을 잊기 (unscaled features를 쓰는 logistic regression은 느리게 train되고 더 나쁜 solution으로 converge함)
- decision threshold를 tune하는 데 test set 사용하기 (validation 또는 cross-validation 사용)
- baseline을 건너뛰고 곧바로 XGBoost로 넘어가기 (interpretability를 잃고 reference point도 없음)
- multicollinearity를 확인하지 않기 (highly correlated features는 coefficient variance를 부풀림)

## 빠른 참조

| Scenario | Model | Regularization | Key setting |
|----------|-------|---------------|-------------|
| Few features, interpretable | LogisticRegression | L2 (default) | C=1.0 |
| Many features, some irrelevant | LogisticRegression | L1 | penalty="l1", solver="saga" |
| High-dim sparse (text) | SGDClassifier | L1 or ElasticNet | loss="log_loss" |
| Imbalanced classes | LogisticRegression | L2 | class_weight="balanced" |
| Need probabilities | LogisticRegression | L2 | predict_proba() |
| Need class labels only | LinearSVC | L2 | Faster than LR for large data |
