# 모델 평가

> 모델은 그것을 측정하는 방식만큼만 좋습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1 (Probability & Distributions, Statistics for ML), Phase 2 Lessons 1-8
**Time:** ~90 minutes

## 학습 목표

- K-fold와 stratified K-fold cross-validation을 처음부터 구현하고 imbalanced data에서 stratification이 왜 중요한지 설명합니다
- precision, recall, F1, AUC-ROC와 regression metric(MSE, RMSE, MAE, R-squared)을 처음부터 계산합니다
- learning curve를 해석해 모델이 high bias 또는 high variance 문제를 겪는지 진단합니다
- data leakage, 잘못된 metric 선택, test set contamination을 포함한 흔한 evaluation mistake를 식별합니다

## 문제

모델을 학습했습니다. 데이터에서 95% accuracy가 나왔습니다. 좋은 모델일까요?

그럴 수도 있고 아닐 수도 있습니다. 데이터의 95%가 한 class에 속한다면, 항상 그 class만 예측하는 모델은 완전히 쓸모없어도 95% accuracy를 얻습니다. 학습한 같은 데이터에서 평가했다면, 모델이 답을 외운 것뿐이므로 95%라는 숫자는 의미가 없습니다. 데이터셋에 time component가 있는데 split 전에 random shuffle했다면, 모델은 미래 데이터를 사용해 과거를 예측하고 있을 수 있습니다.

Model evaluation은 대부분의 ML project가 잘못되는 지점입니다. 잘못된 metric은 나쁜 모델을 좋아 보이게 만듭니다. 잘못된 split은 모델이 cheat하게 합니다. 잘못된 비교는 더 나쁜 모델을 고르게 만듭니다. evaluation을 올바르게 하는 것은 선택 사항이 아닙니다. production에서 작동하는 모델과 실제 데이터를 보는 순간 실패하는 모델의 차이입니다.

## 개념

### 훈련, 검증, 테스트

```mermaid
flowchart LR
    A[Full Dataset] --> B[Train Set 60-70%]
    A --> C[Validation Set 15-20%]
    A --> D[Test Set 15-20%]
    B --> E[Fit Model]
    E --> C
    C --> F[Tune Hyperparameters]
    F --> E
    F --> G[Final Model]
    G --> D
    D --> H[Report Performance]
```

세 split에는 세 가지 목적이 있습니다.

- **Training set**: 모델은 이 데이터에서 학습합니다. training 중 이 example들을 봅니다.
- **Validation set**: hyperparameter를 tuning하고 모델 사이에서 선택하는 데 사용합니다. 모델은 이 데이터로 학습하지 않지만, 우리의 결정은 이 데이터의 영향을 받습니다.
- **Test set**: 마지막에 final performance를 보고하기 위해 정확히 한 번만 건드립니다. test performance를 보고 다시 모델을 바꾸러 돌아간다면, 그것은 더 이상 test set이 아닙니다. 두 번째 validation set이 된 것입니다.

test set은 보고된 성능이 truly unseen data에서 모델이 어떻게 작동할지를 반영한다는 hold-out 보증입니다.

### K-fold cross-validation

작은 데이터셋에서는 단일 train/validation split이 데이터를 낭비하고 noisy estimate를 제공합니다. K-fold cross-validation은 모든 데이터를 training과 validation 양쪽에 사용합니다.

```mermaid
flowchart TB
    subgraph Fold1["Fold 1"]
        direction LR
        V1["Val"] --- T1a["Train"] --- T1b["Train"] --- T1c["Train"] --- T1d["Train"]
    end
    subgraph Fold2["Fold 2"]
        direction LR
        T2a["Train"] --- V2["Val"] --- T2b["Train"] --- T2c["Train"] --- T2d["Train"]
    end
    subgraph Fold3["Fold 3"]
        direction LR
        T3a["Train"] --- T3b["Train"] --- V3["Val"] --- T3c["Train"] --- T3d["Train"]
    end
    subgraph Fold4["Fold 4"]
        direction LR
        T4a["Train"] --- T4b["Train"] --- T4c["Train"] --- V4["Val"] --- T4d["Train"]
    end
    subgraph Fold5["Fold 5"]
        direction LR
        T5a["Train"] --- T5b["Train"] --- T5c["Train"] --- T5d["Train"] --- V5["Val"]
    end
    Fold1 --> R["Average scores"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. 데이터를 같은 크기의 K개 fold로 나눕니다
2. 각 fold에 대해 K-1개 fold로 학습하고 남은 fold로 validate합니다
3. K개의 validation score를 평균냅니다

K=5 또는 K=10이 표준적인 선택입니다. 모든 데이터 포인트는 validation에 정확히 한 번 사용됩니다. 평균 score는 단일 split보다 더 안정적인 estimate입니다.

**Stratified K-fold**: 각 fold에서 class distribution을 보존합니다. 데이터셋이 class A 70%, class B 30%라면 각 fold도 거의 같은 비율을 갖습니다. random split이 모든 minority sample을 한 fold에 넣어버릴 수 있는 imbalanced dataset에서 중요합니다.

### 분류 metric

**Confusion matrix**: 기반이 되는 표입니다. binary classification에서는 다음과 같습니다.

|  | Predicted Positive | Predicted Negative |
|--|---|---|
| Actually Positive | True Positive (TP) | False Negative (FN) |
| Actually Negative | False Positive (FP) | True Negative (TN) |

다른 모든 metric은 이 matrix에서 나옵니다.

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN). 올바른 prediction의 비율입니다. class가 imbalanced할 때 오해를 부릅니다.
- **Precision** = TP / (TP + FP). positive로 예측한 것 중 실제 positive는 얼마나 되나요? false positive 비용이 클 때 사용합니다(예: spam filter가 실제 email을 spam으로 표시).
- **Recall** (sensitivity) = TP / (TP + FN). 실제 positive 중 얼마나 잡아냈나요? false negative 비용이 클 때 사용합니다(예: cancer screening이 tumor를 놓침).
- **F1 score** = 2 * precision * recall / (precision + recall). precision과 recall의 harmonic mean입니다. 어느 하나가 명확히 우세하지 않을 때 둘의 균형을 맞춥니다.
- **AUC-ROC**: Receiver Operating Characteristic curve 아래 면적입니다. 다양한 classification threshold에서 true positive rate와 false positive rate를 그립니다. AUC = 0.5는 random guessing, AUC = 1.0은 완벽한 분리를 뜻합니다. threshold-independent입니다. 즉 선택한 cutoff와 관계없이 모델이 positive를 negative보다 얼마나 잘 위에 rank하는지 측정합니다.

### 회귀 metric

- **MSE** (Mean Squared Error) = mean((y_true - y_pred)^2). 큰 error를 제곱으로 벌점화합니다. outlier에 민감합니다.
- **RMSE** (Root Mean Squared Error) = sqrt(MSE). target variable과 같은 단위를 갖습니다. MSE보다 해석하기 쉽습니다.
- **MAE** (Mean Absolute Error) = mean(|y_true - y_pred|). 모든 error를 선형으로 다룹니다. MSE보다 outlier에 더 robust합니다.
- **R-squared** = 1 - SS_res / SS_tot, where SS_res = sum((y_true - y_pred)^2) and SS_tot = sum((y_true - y_mean)^2). 모델이 설명한 variance의 비율입니다. R^2 = 1.0은 완벽함을 뜻합니다. R^2 = 0.0은 모델이 항상 mean을 예측하는 것보다 낫지 않다는 뜻입니다. 모델이 mean보다 나쁘면 R^2는 음수가 될 수 있습니다.

### Learning curve

training set size의 함수로 training score와 validation score를 그립니다.

- **High bias (underfitting)**: 두 curve가 낮은 score로 수렴합니다. 데이터를 더 추가해도 도움이 되지 않습니다. 더 복잡한 모델이 필요합니다.
- **High variance (overfitting)**: training score는 높지만 validation score는 훨씬 낮습니다. 둘 사이의 gap이 큽니다. 데이터를 더 추가하면 도움이 될 수 있습니다.

### Validation curve

hyperparameter의 함수로 training score와 validation score를 그립니다.

- 낮은 complexity: 두 score가 모두 낮습니다(underfitting)
- 적절한 complexity: 두 score가 모두 높고 서로 가깝습니다
- 높은 complexity: training score는 높게 유지되지만 validation score는 떨어집니다(overfitting)

optimal hyperparameter value는 validation score가 peak에 도달하는 지점입니다.

### 흔한 평가 실수

**Data leakage**: test set의 정보가 training으로 새어 들어갑니다. 예: split 전에 전체 데이터셋에 scaler를 fit하기, time series prediction에 future data 포함하기, target에서 파생된 feature 사용하기. 항상 먼저 split하고 그다음 preprocess하세요.

**Class imbalance**: transaction의 99%는 legitimate이고 1%는 fraud입니다. 항상 "legitimate"라고 예측하는 모델은 99% accuracy를 얻습니다. 대신 precision, recall, F1 또는 AUC-ROC를 사용하세요.

**Wrong metric**: recall을 optimize해야 할 때 accuracy를 optimize하거나(medical diagnosis), 데이터에 heavy outlier가 있을 때 RMSE를 optimize하는 경우입니다(대신 MAE 사용).

**Not using stratified splits**: imbalanced data에서는 random split이 validation fold에 minority sample을 아주 적게 넣어 unstable estimate를 만들 수 있습니다.

**Testing too often**: test performance를 보고 조정할 때마다 test set에 overfit합니다. test set은 single-use입니다.

```figure
precision-recall-threshold
```

## 직접 만들기

### Step 1: train/validation/test split

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### Step 2: K-fold와 stratified K-fold cross-validation

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### Step 3: confusion matrix와 classification metric

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### Step 4: regression metric

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### Step 5: learning curve

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### Step 6: 테스트용 simple classifier와 전체 demo

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")
```

## 사용하기

scikit-learn에서는 evaluation이 workflow에 내장되어 있습니다.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

처음부터 구현한 버전은 cross-validation이 정확히 무엇을 하는지(마법이 아니라 for-loop와 index tracking일 뿐), 각 metric이 어떻게 계산되는지(TP/FP/TN/FN을 세는 것), 그리고 stratification이 왜 중요한지(각 fold에서 class ratio 보존)를 보여줍니다. library 버전은 parallelism, 더 많은 scoring option, pipeline과의 integration을 더합니다.

## 결과물

이 lesson은 다음을 만듭니다.
- `outputs/skill-evaluation.md` - classification과 regression model의 evaluation strategy를 다루는 skill

## 연습 문제

1. precision-recall curve를 구현하세요. 서로 다른 threshold에서 precision vs recall을 plot합니다. average precision(PR curve 아래 면적)을 계산하세요. imbalanced dataset에서 PR curve와 ROC curve를 비교하고 각각이 언제 더 정보가 많은지 설명하세요.
2. nested cross-validation loop를 만드세요. outer loop는 model performance를 평가하고, inner loop는 hyperparameter를 tuning합니다. evaluation에 validation data가 leak되지 않게 두 모델을 공정하게 비교하는 데 사용하세요.
3. model comparison을 위한 permutation test를 구현하세요. label을 shuffle하고, 다시 학습하고, performance를 측정합니다. null distribution을 만들기 위해 100번 반복하세요. 이 distribution에 대해 관측된 model performance의 p-value를 계산하세요.

## 핵심 용어

| Term | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Overfitting | "training data를 외우기" | 모델이 training data의 noise를 포착해 training에서는 잘 작동하지만 unseen data에서는 나쁘게 작동하는 것 |
| Cross-validation | "서로 다른 subset에서 testing하기" | data의 어느 부분을 validation에 사용할지 체계적으로 회전시키고, 모든 회전의 결과를 평균내는 것 |
| Precision | "예측한 positive 중 얼마나 맞았는지" | TP / (TP + FP): positive prediction 중 실제로 positive인 비율 |
| Recall | "실제 positive 중 얼마나 찾았는지" | TP / (TP + FN): 실제 positive 중 올바르게 식별된 비율 |
| AUC-ROC | "모델이 class를 얼마나 잘 분리하는지" | 모든 threshold에 걸친 true positive rate vs false positive rate curve 아래 면적이며, 0.5(random)부터 1.0(perfect)까지입니다 |
| R-squared | "얼마나 많은 variance가 설명되는지" | 1 - (sum of squared residuals / total sum of squares): 모델이 포착한 target variance의 비율 |
| Data leakage | "모델이 cheated했다" | prediction time에는 사용할 수 없는 정보를 training 중 사용해 낙관적인 evaluation으로 이어지는 것 |
| Learning curve | "데이터가 많아질수록 performance가 어떻게 변하는지" | training set size 대비 training 및 validation score를 그린 plot으로, underfitting 또는 overfitting을 드러냅니다 |
| Stratified split | "class ratio를 균형 있게 유지하기" | 각 subset이 full dataset과 같은 class proportion을 갖도록 데이터를 나누는 것 |

## 더 읽을거리

- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) - cross-validation, metric, hyperparameter tuning에 대한 포괄적인 reference
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) - interactive example을 포함한 명확한 설명
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) - 서로 다른 CV strategy가 언제 왜 작동하는지에 대한 엄밀한 논의
