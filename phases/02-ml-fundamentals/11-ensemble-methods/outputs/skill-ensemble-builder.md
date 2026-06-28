---
name: skill-ensemble-builder
description: 문제에 맞는 ensemble method를 선택하고 configure하기
version: 1.0.0
phase: 2
lesson: 11
tags: [ensemble, bagging, boosting, random-forest, xgboost, stacking]
---

# Ensemble Method 선택 가이드

Ensemble은 여러 model을 결합해 어떤 단일 model보다 더 나은 prediction을 만든다. 질문은 항상 이것이다. 어떤 종류의 ensemble을 언제 써야 하는가?

## 의사결정 체크리스트

1. 현재 model의 주된 문제는 무엇인가?
   - 높은 분산(과대적합): bagging(Random Forest) 사용
   - 높은 편향(과소적합): boosting(Gradient Boosting, XGBoost) 사용
   - 둘 다 있거나 최대 accuracy를 원함: stacking 사용

2. 데이터가 얼마나 많은가?
   - 1,000 row 미만: Random Forest(robust하고 잘못 configure하기 어려움)
   - 1,000-100,000 row: XGBoost 또는 LightGBM(tabular에서 전반적으로 가장 좋음)
   - 100,000 row 초과: LightGBM(가장 빠른 gradient boosting이며 large data를 잘 처리함)

3. tuning에 얼마나 시간을 투자할 수 있는가?
   - 최소: default Random Forest(거의 항상 작동)
   - 중간: `learning_rate=0.1`인 XGBoost, early stopping으로 n_estimators 조정
   - 최대: Bayesian hyperparameter search를 적용한 LightGBM 또는 XGBoost

4. interpretability가 필요한가?
   - 예: single decision tree 또는 feature importance를 제공하는 작은 Random Forest
   - 부분적으로: SHAP value를 사용하는 gradient boosting
   - 아니오: stacking 또는 deep ensemble

5. 데이터에 noise와 outlier가 많은가?
   - 예: Random Forest(bagging은 noise에 robust함)
   - 아니오: gradient boosting(clean data에서 accuracy를 더 끌어올릴 수 있음)

## 각 method를 사용할 때

**Random Forest (Bagging)**: 안전한 첫 선택지다. bootstrap sample에 많은 tree를 학습하고 평균낸다. 편향을 늘리지 않고 분산을 줄인다. moderate data에서는 과대적합하기가 거의 어렵다. 필요한 tuning은 최소다. n_estimators=100-500으로 설정하고 나머지는 default로 둔다.

**AdaBoost**: sample reweighting을 사용하는 sequential boosting이다. 단순한 base learner(decision stump)와 잘 작동한다. 잘못 분류된 point의 weight를 올리므로 outlier와 noisy label에 민감하다. 실무에서는 대부분 gradient boosting으로 대체되었다.

**Gradient Boosting**: 각 새 tree를 지금까지 ensemble의 residual에 맞춘다. 편향을 줄인다. tabular data에서 가장 강력한 method다. learning_rate, n_estimators, max_depth, min_child_weight, subsample tuning이 필요하다.

**XGBoost**: regularization, second-order optimization, systems-level speedup을 더한 gradient boosting이다. missing value를 native하게 처리한다. Kaggle competition과 tabular data production ML의 default 선택지다.

**LightGBM**: level-wise 대신 leaf-wise growth를 사용하는 gradient boosting이다. large dataset에서는 XGBoost보다 빠르다. histogram-based split을 사용한다. 50k row를 넘는 dataset에 가장 좋다.

**CatBoost**: categorical feature를 native하게 처리하는 gradient boosting이다. one-hot encode가 필요 없다. categorical feature가 많을 때 좋다.

**Stacking**: 서로 다양한 base model의 prediction으로 meta-learner를 학습한다. 절대적으로 가장 좋은 accuracy가 필요하고 compute 여유가 있을 때 사용한다. leakage를 피하려면 base model prediction을 항상 cross-validation으로 생성한다.

**Voting**: 가장 단순한 ensemble이다. hard voting(majority class) 또는 soft voting(average probabilities)을 사용한다. meta-learner 없이 서로 다른 model 2-3개를 빠르게 결합하는 방법이다.

## 흔한 실수

- early stopping 없이 gradient boosting 사용(너무 많은 round를 실행하면 과대적합한다)
- learning_rate를 너무 높게 설정(0.3 초과는 보통 instability를 일으킨다)
- gradient boosting에서 max_depth를 tuning하지 않음(unlimited 또는 매우 깊은 tree default는 과대적합한다)
- 모두 같은 type의 model로 stacking(diversity가 stacking의 핵심이다)
- noisy data에서 AdaBoost 사용(outlier의 weight가 round마다 계속 커진다)
- Random Forest가 underfitting을 고쳐주리라 기대(분산을 줄이지, 편향을 줄이지 않는다)

## Method별 tuning 우선순위

**Random Forest:**
1. n_estimators: 100-500(더 많아도 나빠지는 경우는 드물고, 느려질 뿐이다)
2. max_depth: None(tree가 완전히 자라도록 둠) 또는 speed를 위해 10-20으로 제한
3. max_features: classification은 "sqrt", regression은 "log2" 또는 n/3

**XGBoost / LightGBM:**
1. learning_rate: 0.01-0.3(tree를 더 많이 쓸 compute가 있다면 낮을수록 좋다)
2. n_estimators: 추측하지 말고 validation set에서 early stopping 사용
3. max_depth: 3-8(6으로 시작)
4. min_child_weight / min_data_in_leaf: 1-20(높을수록 과대적합을 막는다)
5. subsample: 0.7-1.0
6. colsample_bytree: 0.7-1.0
7. reg_alpha (L1) and reg_lambda (L2): 0-10

## 빠른 참고

| Method | 줄이는 것 | Speed | Tuning effort | Best for |
|--------|---------|-------|--------------|----------|
| Random Forest | Variance | Fast | Low | Noisy data, quick baseline |
| AdaBoost | Bias | Fast | Low | Simple base learner, clean data |
| Gradient Boosting | Bias | Medium | High | Tabular data, competition |
| XGBoost | Both | Fast | High | Production tabular ML |
| LightGBM | Both | Fastest | High | Large dataset(50k+ row) |
| CatBoost | Both | Medium | Medium | 많은 categorical feature |
| Stacking | Both | Slow | High | Maximum accuracy, diverse model |
| Voting | Variance | Fast | None | 2-3개 model의 빠른 결합 |
