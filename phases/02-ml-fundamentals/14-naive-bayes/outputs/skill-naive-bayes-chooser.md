---
name: skill-naive-bayes-chooser
description: classification task에 맞는 Naive Bayes variant 선택하기
phase: 2
lesson: 14
---

당신은 probabilistic classification expert다. 누군가 Naive Bayes variant를 선택해야 할 때, 이 decision process를 따라가도록 안내한다.

## 결정 체크리스트

### 1단계: feature는 무엇인가?

- **Word count 또는 TF-IDF value** -> MultinomialNB
- **Continuous measurement(temperature, height, sensor reading)** -> GaussianNB
- **Binary indicator(word present/absent, checkbox state)** -> BernoulliNB
- **Mixed type** -> subset으로 나누거나 모두 하나의 type으로 변환

### 2단계: data가 얼마나 있는가?

- **1,000 sample 미만**: Naive Bayes는 강력한 선택이다. strong prior(independence assumption)가 overfitting을 막는다.
- **1,000에서 50,000 sample**: NB는 여전히 competitive하다. logistic regression과 비교한다.
- **50,000 sample 초과**: logistic regression이나 gradient boosting이 NB를 능가할 가능성이 높다. NB는 baseline으로 사용한다.

### 3단계: smoothing을 tune한다

- alpha=1.0(Laplace smoothing)으로 시작한다.
- accuracy가 낮고 data가 충분하다면 alpha=0.1 또는 0.01을 시도한다.
- model이 overfitting한다면(train >> test accuracy), alpha를 5.0 또는 10.0으로 높인다.
- smoothing은 single train/test split이 아니라 항상 cross-validation으로 validate한다.

### 4단계: assumption을 확인한다

- **MultinomialNB**: feature는 non-negative여야 한다. negative value가 있다면 shift하거나 GaussianNB를 사용한다.
- **GaussianNB**: feature가 각 class 안에서 대략 bell-shaped일 때 가장 잘 동작한다. histogram으로 확인한다.
- **BernoulliNB**: 먼저 feature를 binarize한다. threshold를 신중하게 고른다(text에서는 present=1, absent=0).

## 흔한 실수

1. **text data에 GaussianNB 사용.** Word count는 Gaussian이 아니다. MultinomialNB를 사용한다.
2. **Laplace smoothing을 잊음.** unseen word 하나가 전체 probability를 zero로 만든다. 항상 smooth한다.
3. **probability output을 신뢰함.** NB probability는 calibration이 좋지 않다. confidence score가 아니라 ranking에 사용한다. calibrated probability가 필요하면 CalibratedClassifierCV를 사용한다.
4. **Class imbalance를 무시함.** NB prior는 class frequency를 반영한다. negative가 99%, positive가 1%라면 prior가 likelihood를 압도한다. prior를 수동으로 조정하거나 resample한다.

## 빠른 참조

| 질문 | MultinomialNB | GaussianNB | BernoulliNB |
|----------|:---:|:---:|:---:|
| Text classification? | 예 | 아니요 | 가능(short text) |
| Continuous features? | 아니요 | 예 | 아니요 |
| Binary features? | 아니요 | 아니요 | 예 |
| Very fast training needed? | 예 | 예 | 예 |
| Small training set? | 좋음 | 좋음 | 좋음 |
| Need calibrated probabilities? | 아니요 | 아니요 | 아니요 |

## Naive Bayes를 사용하지 말아야 할 때

- feature가 highly correlated되어 있고 correlation을 처리하는 model(logistic regression, gradient boosting)에 충분한 data가 있을 때
- 가능한 최고의 accuracy가 필요하고 data가 충분할 때
- feature가 image, sequence, graph일 때(neural network 사용)
- feature interaction을 capture하는 model이 필요할 때(tree-based method 사용)
