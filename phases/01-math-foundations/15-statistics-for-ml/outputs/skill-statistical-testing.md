---
name: skill-statistical-testing
description: ML 모델 비교와 실험 평가에 맞는 올바른 통계 검정을 선택
version: 1.0.0
phase: 1
lesson: 15
tags: [statistics, hypothesis-testing, model-comparison]
---

# ML을 위한 통계 검정

모델을 비교하거나 A/B 실험을 실행하거나 결과를 검증할 때 올바른 검정을 고르는 방법입니다.

## 의사결정 체크리스트

1. 무엇을 비교하나요? Means, proportions, distributions, 또는 correlations?
2. 그룹은 몇 개인가요? One sample vs reference, two groups, 또는 multiple groups?
3. 관측값이 paired(same test set, same folds)인가요, independent인가요?
4. 데이터가 normally distributed인가요? n < 30이고 분명히 normal이 아니면 non-parametric을 사용하세요.
5. 데이터가 continuous, ordinal, 또는 categorical인가요?
6. 몇 개의 test를 실행하나요? 하나보다 많으면 correction을 적용하세요.

## 의사결정 트리

```text
Comparing means?
  Two groups?
    Paired (same data splits)? --> Paired t-test (or Wilcoxon signed-rank if non-normal)
    Independent? --> Welch's t-test (or Mann-Whitney U if non-normal)
  Multiple groups?
    Paired? --> Repeated measures ANOVA (or Friedman test)
    Independent? --> One-way ANOVA (or Kruskal-Wallis)

Comparing proportions?
  Two groups? --> Chi-squared test or Fisher's exact test (small n)
  Multiple groups? --> Chi-squared test

Comparing distributions?
  Is one distribution a reference? --> Kolmogorov-Smirnov test
  Are both empirical? --> Two-sample KS test

Measuring association?
  Both continuous, roughly normal? --> Pearson correlation
  Ordinal or non-normal? --> Spearman rank correlation
  Categorical x Categorical? --> Chi-squared test of independence

Running many tests?
  Apply Bonferroni correction: alpha_adjusted = alpha / number_of_tests
  Or use Holm-Bonferroni (less conservative, still controls family-wise error)
```

## 각 검정을 사용할 때

| 검정 | 데이터 타입 | 가정 | ML use case |
|---|---|---|---|
| Paired t-test | Continuous, paired | Normal differences | 같은 k-fold splits에서 모델 2개 비교 |
| Wilcoxon signed-rank | Continuous/ordinal, paired | 없음(non-parametric) | 모델 2개 비교, 작은 k(5-10 folds) |
| Welch's t-test | Continuous, independent | 대략 normal | 서로 다른 dataset 두 개에서 모델 비교 |
| Mann-Whitney U | Continuous/ordinal, independent | 없음 | latency distributions 비교 |
| ANOVA | Continuous, 3+ groups | Normal, equal variance | 여러 model architectures 비교 |
| Kruskal-Wallis | Continuous/ordinal, 3+ groups | 없음 | 여러 모델, non-normal metrics 비교 |
| Chi-squared | Categorical counts | Expected count >= 5 | class distributions, confusion matrices 비교 |
| Fisher's exact | Categorical counts | Small samples | rare event comparison |
| KS test | Continuous | 없음 | predictions가 expected distribution을 따르는지 확인 |
| Bootstrap CI | Any statistic | 없음 | AUC, F1, 어떤 metric이든 confidence interval |
| McNemar's test | Paired binary | 없음 | 같은 test set에서 classifier 두 개 비교 |

## 모델 비교 recipe

1. 실험 전에 metric과 significance level(alpha = 0.05)을 정의하세요.
2. 두 모델을 같은 k-fold cross-validation splits(k = 5 또는 10)에서 실행하세요.
3. paired scores를 수집하세요: (a_1, b_1), (a_2, b_2), ..., (a_k, b_k).
4. differences를 계산하세요: d_i = b_i - a_i.
5. paired test를 실행하세요(k <= 10이면 Wilcoxon, k > 10 또는 normal diffs이면 paired t-test).
6. 보고하세요: p-value, mean difference, 95% confidence interval, effect size(Cohen's d).
7. p < alpha이고 effect size가 의미 있다면 차이는 실제이며 행동할 가치가 있습니다.

## 흔한 실수

- data가 paired인데 independent test 사용. 두 모델이 같은 test folds에서 평가되었다면 paired test를 사용해야 합니다. Independent tests는 pairing을 버리고 statistical power를 잃습니다.
- effect size 없이 p < 0.05만 보고. 통계적으로 유의한 0.1% accuracy improvement는 배포할 가치가 없습니다. 항상 Cohen's d 또는 raw mean difference를 계산하세요.
- 서로 다른 test sets에서 모델 비교. test set은 두 모델에 대해 반드시 동일해야 합니다. 다른 test sets는 비교를 무의미하게 만듭니다.
- 20개 비교를 실행하고 Bonferroni correction 없이 최고 결과만 보고. alpha = 0.05에서 20개 test를 하면 우연히 1개의 false positive를 기대하게 됩니다.
- imbalanced data에 accuracy 사용. 99% majority class에서는 trivial classifier도 99%를 달성합니다. F1, precision-recall AUC, 또는 Matthews correlation coefficient를 사용하세요.
- cross-validation folds를 independent samples처럼 취급. 이들은 training data를 공유하므로 independence assumption을 위반합니다. corrected resampled t-test가 이를 고려합니다.

## 빠른 참조: effect size 해석

| Cohen's d | 해석 |
|---|---|
| 0.2 | 작은 효과 |
| 0.5 | 중간 효과 |
| 0.8 | 큰 효과 |
| > 1.0 | 매우 큰 효과 |

| 보고할 것 | 이유 |
|---|---|
| p-value | 차이가 실제인가? |
| Confidence interval | 차이가 얼마나 클 수 있는가? |
| Effect size (Cohen's d) | 차이가 의미 있는가? |
| Sample size (n or k folds) | 결과를 신뢰할 수 있는가? |
