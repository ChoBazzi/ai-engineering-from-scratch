# 머신러닝을 위한 통계

> 통계는 모델이 실제로 작동하는지, 아니면 운이 좋았을 뿐인지 알려주는 방법입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 minutes

## 학습 목표

- descriptive statistics, Pearson/Spearman correlation, covariance matrices를 처음부터 계산하기
- hypothesis tests(t-test, chi-squared)를 수행하고 p-values와 confidence intervals를 올바르게 해석하기
- distributional assumptions 없이 어떤 metric에도 confidence interval을 만들기 위해 bootstrap resampling 사용하기
- effect size measures를 사용해 statistical significance와 practical significance를 구분하기

## 문제

두 모델을 학습했습니다. Model A는 test set에서 0.87을 기록합니다. Model B는 0.89를 기록합니다. Model B를 배포합니다. 3주 뒤 production metrics가 이전보다 나빠졌습니다. 무슨 일이 일어났을까요?

Model B는 실제로 Model A보다 낫지 않았습니다. 0.02 차이는 noise였습니다. test set이 너무 작았거나 variance가 너무 컸거나 둘 다였습니다. 당신은 improvement로 포장된 randomness를 배포했습니다.

이 일은 계속 일어납니다. Kaggle leaderboard shakeups. 재현되지 않는 논문. 몇백 samples로 winner를 선언하는 A/B tests. root cause는 항상 같습니다. 누군가 statistics를 건너뛰었습니다.

Statistics는 signal과 noise를 구분하는 도구를 제공합니다. 차이가 실제인지, 얼마나 확신해야 하는지, 결과를 신뢰하기 전에 얼마나 많은 data가 필요한지 알려줍니다. 모든 ML pipeline, 모든 model comparison, 모든 experiment에는 statistics가 필요합니다. 없으면 추측하는 것입니다.

## 개념

### Descriptive Statistics: 데이터 요약

무언가를 model하기 전에 data가 어떻게 생겼는지 알아야 합니다. Descriptive statistics는 dataset을 그 모양을 포착하는 몇 개의 숫자로 압축합니다.

**Measures of central tendency**는 "가운데가 어디인가?"에 답합니다.

```text
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

mean은 balance point입니다. median은 halfway mark입니다. 둘이 벌어지면 distribution이 skewed입니다. income distributions는 mean >> median입니다(billionaires로 인한 right skew). training 중 loss distributions는 종종 mean << median입니다(easy samples로 인한 left skew).

**Measures of spread**는 "data가 얼마나 퍼져 있는가?"에 답합니다.

```text
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentiles**는 sorted data를 100개의 동일한 부분으로 나눕니다. 25th percentile(Q1)은 값의 25%가 이 지점 아래에 있다는 뜻입니다. 50th percentile은 median입니다. 75th percentile은 Q3입니다.

```text
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

ML에서는 inference latency, prediction confidence distributions, error distributions 이해를 위해 percentiles가 중요합니다. average error는 낮지만 P99 error가 끔찍한 모델은 safety-critical application에서는 쓸모없을 수 있습니다.

**Sample vs population statistics.** sample에서 variance를 계산할 때 n 대신 (n-1)로 나눕니다. 이것이 Bessel's correction입니다. sample mean이 true population mean이 아니라는 사실을 보정합니다. denominator가 n이면 true variance를 체계적으로 과소추정합니다. (n-1)을 쓰면 estimate가 unbiased입니다.

```text
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

실무에서는 n이 크면(수천 samples) 차이가 무시할 만합니다. n이 작으면(수십 samples) 중요합니다.

### Correlation: 변수들이 함께 움직이는 방식

Correlation은 두 변수 사이의 linear relationship의 strength와 direction을 측정합니다.

**Pearson correlation coefficient**는 linear association을 측정합니다.

```text
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson은 relationship이 linear이고 두 변수가 대략 normally distributed라고 가정합니다. outliers에 민감합니다. 극단값 하나가 r을 0.1에서 0.9로 끌어올릴 수 있습니다.

**Spearman rank correlation**은 monotonic association을 측정합니다.

```text
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**각각을 사용할 때:**

```text
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**황금률:** correlation은 causation을 의미하지 않습니다. ice cream sales와 drowning deaths는 둘 다 여름에 증가하기 때문에 correlated입니다. 모델 accuracy와 parameters 수는 correlated이지만, parameters를 추가한다고 자동으로 accuracy가 올라가지는 않습니다(see: overfitting).

### 공분산 행렬

두 변수의 covariance는 둘이 함께 어떻게 변하는지 측정합니다.

```text
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

d features에 대해 covariance matrix C는 C[i][j] = Cov(feature_i, feature_j)인 d x d matrix입니다. diagonal entries C[i][i]는 각 feature의 variance입니다.

```text
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**PCA와의 연결.** PCA는 covariance matrix를 eigendecompose합니다. eigenvectors는 principal components(maximum variance directions)입니다. eigenvalues는 각 component가 얼마나 많은 variance를 포착하는지 알려줍니다. Lesson 10에서 다룬 내용이지만, 이제 왜 covariance matrix를 분해하는 것이 맞는지 보입니다. covariance matrix는 data의 모든 pairwise linear relationships를 인코딩합니다.

**correlation과의 연결.** correlation matrix는 standardized variables(각각 standard deviation으로 나눈 변수)의 covariance matrix입니다. correlation은 covariance를 normalize해 모든 값이 [-1, 1]에 들어가게 합니다.

### 가설 검정

Hypothesis testing은 uncertainty 아래에서 decision을 내리는 framework입니다. claim으로 시작하고, data를 수집한 뒤, data가 claim과 일관되는지 판단합니다.

**설정:**

```text
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value**는 H0가 참이라고 가정했을 때 관측한 것만큼 extreme한 data를 볼 확률입니다. H0가 참일 확률이 아닙니다. 이것이 statistics에서 가장 흔한 오해입니다.

```text
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence intervals**는 parameter에 대한 plausible values의 range를 제공합니다.

```text
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

confidence interval의 width는 precision을 알려줍니다. 넓은 interval은 uncertainty가 높다는 뜻입니다. 좁은 interval은 estimate가 precise하다는 뜻입니다(data가 biased라면 accurate하다는 뜻은 아닙니다).

### t-test 설명

t-test는 means를 비교합니다. 여러 종류가 있습니다.

**One-sample t-test:** population mean이 hypothesized value와 다른가?

```text
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (independent):** 두 group means가 다른가?

```text
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** measurements가 pair로 올 때(같은 model을 같은 data splits에서 평가):

```text
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

ML에서 paired t-test는 흔합니다. 두 모델을 같은 10 cross-validation folds에서 실행하고 score를 pairwise로 비교합니다.

### Chi-squared test 설명

chi-squared test는 observed frequencies가 expected frequencies와 맞는지 확인합니다. categorical data에 유용합니다.

```text
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 모델을 위한 A/B Testing

ML에서 A/B testing은 web A/B testing과 같지 않습니다. Model comparison에는 고유한 어려움이 있습니다.

```text
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**절차:**

```text
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 통계적 유의성과 실무적 유의성

결과가 statistically significant하지만 practically meaningless일 수 있습니다. data가 충분히 많으면 사소한 차이도 statistically significant가 됩니다.

```text
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size**는 sample size와 독립적으로 차이가 얼마나 큰지 정량화합니다.

```text
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

항상 p-value와 effect size를 모두 보고하세요. p-value는 차이가 실제인지 알려줍니다. effect size는 그 차이가 중요한지 알려줍니다.

### 다중 비교 문제

많은 hypotheses를 test하면 일부는 우연히 "significant"가 됩니다. alpha = 0.05에서 20가지를 test하면 실제 효과가 없어도 false positive 1개를 기대합니다.

```text
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction:** alpha를 test 수로 나눕니다.

```text
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

ML에서는 model을 여러 metrics로 비교하거나, 많은 hyperparameter configurations를 test하거나, 여러 datasets에서 평가할 때 중요합니다.

### Bootstrap methods 설명

Bootstrapping은 replacement를 사용해 data를 resampling함으로써 statistic의 sampling distribution을 추정합니다. underlying distribution에 대한 가정이 필요 없습니다.

**알고리즘:**

```text
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval(percentile method) 계산:**

```text
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**ML에서 bootstrap이 중요한 이유:**

```text
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Model comparison을 위한 bootstrap:**

```text
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

이는 paired t-test보다 robust합니다. distributional assumptions가 없기 때문입니다.

### Parametric tests와 non-parametric tests

**Parametric tests**는 특정 distribution(보통 normal)을 가정합니다.

```text
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric tests**는 distributional assumptions가 없습니다.

```text
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**non-parametric을 사용할 때:**

```text
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**parametric을 사용할 때:**

```text
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

ML experiments에서는 보통 n이 작습니다(5 또는 10 cross-validation folds). 따라서 Wilcoxon signed-rank 같은 non-parametric tests가 t-tests보다 더 적절한 경우가 많습니다.

### 중심극한정리: 실무적 함의

CLT는 underlying population distribution과 무관하게 sample means의 distribution이 n이 커질수록 normal distribution에 접근한다고 말합니다.

```text
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**ML에서 중요한 이유:**

```text
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT가 하지 않는 것:**

```text
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 논문에서 흔한 통계 실수

1. **training set에서 test하기.** overfitting을 보장합니다. training 중 모델이 보지 않는 data를 항상 hold out하세요.

2. **confidence intervals 없음.** uncertainty 없이 단일 accuracy number를 보고하면 결과를 재현하거나 검증하기 어렵습니다.

3. **multiple comparisons 무시.** 50 configurations를 test하고 correction 없이 최고 결과를 보고하면 false positive rates가 부풀어 오릅니다.

4. **statistical significance와 practical significance 혼동.** 0.01% accuracy improvement에 대한 p-value 0.001은 의미가 없습니다.

5. **imbalanced data에 accuracy 사용.** negative class가 99%인 dataset에서 99% accuracy는 모델이 아무것도 배우지 않았다는 뜻일 수 있습니다. precision, recall, F1, 또는 AUC를 사용하세요.

6. **Cherry-picking metrics.** 모델이 이기는 metric만 보고하는 것. 정직한 evaluation은 관련된 모든 metrics를 보고합니다.

7. **train/test split 사이 정보 leakage.** split하기 전에 normalization하거나 future data를 사용해 past를 예측하는 것.

8. **variance estimates 없는 작은 test sets.** 100 samples로 평가하고 2% improvement를 주장하는 것은 signal이 아니라 noise입니다.

9. **data가 independent가 아닌데 independence를 가정.** 같은 patient의 medical images, 같은 document의 여러 sentences. group 안의 observations는 correlated입니다.

10. **P-hacking.** p < 0.05가 나올 때까지 다른 tests, subsets, exclusion criteria를 시도하는 것. 결과는 search의 artifact입니다.

## 직접 만들기

다음을 구현합니다.

1. **Descriptive statistics from scratch** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Correlation functions** (Pearson and Spearman, with the covariance matrix)
3. **Hypothesis tests** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Bootstrap confidence intervals** (for any statistic, no assumptions needed)
5. **A/B test simulator** (generate data, test, check for Type I and Type II errors)
6. **Statistical vs practical significance demo** (large n이 모든 것을 "significant"하게 만든다는 것을 보여줌)

모두 `math`와 `random`만 사용해 처음부터 만듭니다. numpy도 scipy도 없습니다.

## 핵심 용어

| 용어 | 정의 |
|---|---|
| Mean | values 합을 count로 나눈 값입니다. outliers에 민감합니다. |
| Median | sorted data의 가운데 값입니다. outliers에 robust합니다. |
| Standard deviation | variance의 square root입니다. 원래 units로 spread를 측정합니다. |
| Percentile | data의 주어진 percentage가 그 아래에 놓이는 값입니다. |
| IQR | Interquartile range. Q3 minus Q1. 가운데 50%의 spread입니다. |
| Pearson correlation | 두 변수 사이의 linear association을 측정합니다. Range [-1, 1]. |
| Spearman correlation | ranks를 사용해 monotonic association을 측정합니다. |
| Covariance matrix | 모든 features 사이의 pairwise covariances matrix입니다. |
| Null hypothesis | effect나 difference가 없다는 default assumption입니다. |
| p-value | null hypothesis가 true일 때 이 정도로 extreme한 data가 나올 확률입니다. |
| Confidence interval | 주어진 confidence level에서 parameter의 plausible values range입니다. |
| t-test | means가 significantly 다른지 test합니다. t-distribution을 사용합니다. |
| Chi-squared test | observed frequencies가 expected frequencies와 다른지 test합니다. |
| Effect size | sample size와 독립적인 difference magnitude입니다. Cohen's d가 흔합니다. |
| Bonferroni correction | false positives를 제어하기 위해 significance threshold를 tests 수로 나눕니다. |
| Bootstrap | sampling distributions를 추정하기 위한 replacement resampling입니다. |
| Type I error | False positive. H0가 true인데 reject하는 것. |
| Type II error | False negative. H0가 false인데 reject하지 못하는 것. |
| Statistical power | false H0를 올바르게 reject할 확률입니다. Power = 1 minus Type II error rate. |
| Central limit theorem | sample size가 커질수록 sample means가 normal distribution으로 수렴합니다. |
| Parametric test | data에 특정 distribution(보통 normal)을 가정합니다. |
| Non-parametric test | distributional assumptions가 없습니다. ranks 또는 signs에서 작동합니다. |
