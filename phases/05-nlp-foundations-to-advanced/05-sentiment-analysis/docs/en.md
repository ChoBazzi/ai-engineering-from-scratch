# 감성 분석 (Sentiment Analysis)

> 표준적인 NLP 과제입니다. 고전적인 텍스트 분류에 대해 알아야 할 대부분이 여기서 드러납니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 minutes

## 문제

"The food was not great." 긍정일까요, 부정일까요?

감성 분석은 단순해 보입니다. 리뷰어가 무언가를 좋아했는지, 좋아하지 않았는지를 말했습니다. 문장에 label을 붙이면 됩니다. 이것이 표준 NLP 과제가 된 이유는 쉬워 보이는 모든 사례 안에 어려운 사례가 숨어 있기 때문입니다. 부정 표현은 의미를 뒤집습니다. 풍자는 의미를 반전합니다. "Not bad at all"은 부정적으로 코딩된 단어가 두 개나 있어도 긍정입니다. 이모지는 주변 텍스트보다 더 많은 signal을 담기도 합니다. 도메인 어휘도 중요합니다. 음악 리뷰의 `tight`와 패션 리뷰의 `tight`는 다릅니다.

감성 분석은 고전 NLP를 위한 실전 실험실입니다. 모든 naive baseline이 왜 특정한 failure mode를 갖는지 이해한다면, 더 풍부한 모델이 왜 발명되었는지도 이해하게 됩니다. 이 lesson은 Naive Bayes baseline을 처음부터 만들고, logistic regression을 추가하며, production sentiment를 compliance-grade 문제로 만드는 함정을 이름 붙입니다.

## 개념

고전적인 sentiment는 두 단계 레시피입니다.

1. **표현하기.** 텍스트를 feature vector로 바꿉니다. BoW, TF-IDF, 또는 n-gram을 사용합니다.
2. **분류하기.** labeled example 위에서 linear model(Naive Bayes, logistic regression, SVM)을 fit합니다.

Naive Bayes는 작동하는 가장 단순한 모델입니다. label이 주어졌을 때 모든 feature가 독립이라고 가정합니다. count로부터 `P(word | positive)`와 `P(word | negative)`를 추정합니다. inference에서는 probability들을 곱합니다. "naive"한 독립 가정은 웃길 만큼 틀렸지만, 결과는 놀랄 만큼 강합니다. 이유는 sparse text feature와 중간 규모 데이터에서는 classifier가 각 단어가 어느 쪽으로 기우는지를, 정확한 joint probability보다 더 중요하게 보기 때문입니다.

Logistic regression은 독립 가정을 고칩니다. 각 feature마다 weight를 학습하고, negative weight도 가질 수 있습니다. bigram feature로서 `not good`은 negative weight를 받습니다. Naive Bayes는 label이 붙은 적 없는 bigram에 대해 그렇게 할 수 없습니다.

```figure
sentiment-logits
```

## 직접 만들기

### 1단계: 실제 mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

일부러 작게 만들었습니다. 실제 작업에서는 수만 개의 example(IMDb, SST-2, Yelp polarity)을 씁니다. 수학은 동일합니다.

### 2단계: 처음부터 만드는 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing(alpha=1.0)은 Laplace smoothing입니다. 이것이 없으면 어떤 class에서 보지 못한 word의 probability가 zero가 되고 log가 폭발합니다. 실무에서는 `alpha=0.01`이 흔합니다. `alpha=1.0`은 교육용 default입니다.

### 3단계: 처음부터 만드는 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

여기서는 L2 regularization이 중요합니다. Text feature는 sparse합니다. L2가 없으면 모델은 training example을 외웁니다. `0.01`에서 시작해 tune하세요.

### 4단계: negation 처리하기(failure mode)

"not good"과 "not bad"를 생각해 봅시다. BoW classifier는 `{not, good}`과 `{not, bad}`를 보고, training에서 더 많이 나타난 쪽으로 학습합니다. Bigram classifier는 `not_good`과 `not_bad`를 서로 다른 feature로 보고 학습합니다. 보통은 이것만으로 충분합니다.

Bigram이 없을 때도 작동하는 더 거친 해결책은 **negation scoping**입니다. negation word 뒤에 오는 token들에 다음 punctuation까지 `NOT_` prefix를 붙입니다.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

이제 `good`과 `NOT_good`은 서로 다른 feature입니다. Classifier는 둘에 반대 weight를 줄 수 있습니다. 세 줄짜리 preprocessing만으로도 sentiment benchmark에서 accuracy가 눈에 띄게 오릅니다.

### 5단계: 중요한 evaluation metric

Class가 imbalanced라면 accuracy만으로는 오해하기 쉽습니다. 실제 sentiment corpus는 보통 70-80% positive이거나 70-80% negative입니다. constant-majority classifier는 80% accuracy를 얻지만 쓸모없습니다. 다음을 모두 report하세요.

- **Per-class precision and recall.** Class마다 한 쌍입니다. Macro-average해서 class balance를 존중하는 단일 숫자를 얻습니다.
- **Macro-F1(imbalanced data의 primary metric).** Per-class F1 score의 평균이며 동일한 weight를 줍니다. Class가 imbalanced일 때 accuracy 대신 사용하세요.
- **Weighted-F1(대안).** Macro와 같지만 class frequency로 weight를 줍니다. Imbalance 자체가 business meaning을 가질 때 macro-F1과 함께 report하세요.
- **Confusion matrix.** Raw count입니다. 어떤 scalar metric도 믿기 전에 항상 inspect하세요. 모델이 어떤 class pair를 혼동하는지 드러냅니다.
- **Per-class error sample.** Class마다 wrong prediction 5개를 뽑으세요. 읽으세요. 실제 error를 읽는 것을 대체할 방법은 없습니다.

심하게 imbalanced한 데이터(> 95-5 ratio)에서는 accuracy 대신 **AUROC**와 **AUPRC**를 report하세요. AUPRC는 minority class에 더 민감하며, 보통 여러분이 신경 쓰는 대상(spam, fraud, rare sentiment)이 바로 그 class입니다.

**피해야 할 흔한 bug.** Imbalanced data에서 macro-F1 대신 micro-F1을 report하면 majority class가 지배하기 때문에 숫자가 높아 보입니다. Macro-F1은 minority-class performance를 보게 만듭니다.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 사용하기

scikit-learn은 이것을 여섯 줄로, 올바르게 처리합니다.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

세 가지를 주목하세요. `stop_words=None`은 negation을 유지합니다. `ngram_range=(1, 2)`는 bigram을 추가해 `not_good`이 feature가 되게 합니다. `sublinear_tf=True`는 반복된 word를 완화합니다. 이 세 flag가 SST-2에서 75% accurate baseline과 85% accurate baseline의 차이를 만듭니다.

### Transformer를 써야 할 때

- Sarcasm detection. 고전 모델은 여기서 실패합니다. 예외 없습니다.
- Sentiment가 document 중간에 바뀌는 긴 review.
- Aspect-based sentiment. "Camera was great but battery was terrible." Sentiment를 aspect에 귀속해야 합니다. Transformer나 structured output model만 가능합니다.
- 영어가 아니거나 low-resource인 language. Multilingual BERT는 zero-shot baseline을 공짜로 제공합니다.

위 항목 중 하나라도 필요하다면 phase 7(transformers deep dive)로 넘어가세요. 그렇지 않다면 TF-IDF + bigram + negation handling 위의 Naive Bayes 또는 logistic regression이 2026년 production baseline입니다.

### Reproducibility trap(다시)

Sentiment model을 retrain하는 일은 routine입니다. Re-evaluate하는 일은 그렇지 않습니다. Paper에 report된 accuracy number는 특정 split, 특정 preprocessing, 특정 tokenizer를 사용합니다. 동일한 pipeline을 쓰지 않고 새 모델을 baseline과 비교하면 misleading delta가 나옵니다. Paper의 숫자가 아니라 여러분의 pipeline에서 항상 baseline을 다시 생성하세요.

## 출시하기

`outputs/prompt-sentiment-baseline.md`로 저장하세요.

```markdown
---
name: sentiment-baseline
description: 새 dataset을 위한 sentiment analysis baseline을 설계합니다.
phase: 5
lesson: 05
---

Dataset description(domain, language, size, label granularity, latency budget)이 주어지면 다음을 output하세요.

1. Feature extraction recipe. Tokenizer, n-gram range, stopword policy(보통 keep), negation handling(scoped prefix 또는 bigram)을 명시하세요.
2. Classifier. Baseline에는 Naive Bayes, production에는 logistic regression, domain에 sarcasm / aspect / cross-lingual이 필요할 때만 transformer를 사용하세요.
3. Evaluation plan. Precision, recall, F1, confusion matrix, per-class error sample을 report하세요. Scalar만 report하지 마세요.
4. Post-deployment에서 monitor할 failure mode 하나. Domain drift와 sarcasm이 상위 두 가지입니다.

Sentiment task에서 stopword 제거를 추천하지 마세요. Class가 imbalanced일 때(예: 90% positive) accuracy를 유일한 metric으로 report하지 마세요. Subword가 많은 language는 word-level TF-IDF보다 FastText 또는 transformer embedding이 필요하다고 flag하세요.
```

## 연습문제

1. **쉬움.** scikit-learn pipeline에 preprocessing step으로 `apply_negation`을 추가하고 작은 sentiment dataset에서 F1 delta를 측정하세요.
2. **보통.** Class-weighted logistic regression을 구현하세요(scikit-learn에 `class_weight="balanced"`를 넘기거나 gradient를 직접 유도하세요). Synthetic 90-10 class imbalance에서 효과를 측정하세요.
3. **어려움.** Sentiment model의 residual에 두 번째 classifier를 학습해 sarcasm detector를 만드세요. Experimental setup을 문서화하세요. Accuracy가 chance보다 낮을 때 reader에게 경고하세요(2-class sarcasm의 chance-level은 ~50%이고, 대부분의 첫 시도는 거기에 머뭅니다).

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Polarity | Positive 또는 negative | Binary label입니다. 때로 neutral 또는 fine-grained(5-star)로 확장됩니다. |
| Aspect-based sentiment | Aspect별 polarity | Text에 언급된 특정 entity나 attribute에 sentiment를 귀속합니다. |
| Negation scoping | 가까운 token 뒤집기 | "not" 뒤의 token에 punctuation까지 `NOT_` prefix를 붙입니다. |
| Laplace smoothing | Count에 1 더하기 | Naive Bayes에서 zero-probability feature를 방지합니다. |
| L2 regularization | Weight 줄이기 | Loss에 `lambda * sum(w^2)`를 더합니다. Sparse text feature에 필수입니다. |

## 더 읽을거리

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — foundational survey입니다. 길지만 처음 네 section이 고전 방법을 모두 다룹니다.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — bigram + Naive Bayes가 짧은 text에서 이기기 어렵다는 것을 보인 paper입니다.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`, `TfidfVectorizer`, 그리고 tune할 모든 knob의 reference입니다.
