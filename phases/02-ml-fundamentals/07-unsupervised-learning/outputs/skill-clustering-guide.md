---
name: skill-clustering-guide
description: 데이터 모양, 노이즈, 제약 조건에 따라 적절한 클러스터링 알고리즘 선택하기
version: 1.0.0
phase: 2
lesson: 7
tags: [clustering, k-means, dbscan, hierarchical, gmm, unsupervised]
---

# 클러스터링 알고리즘 선택 가이드

클러스터링에는 단 하나의 최선 알고리즘이 없습니다. 올바른 선택은 클러스터 모양, 클러스터 수를 알고 있는지, 데이터에 노이즈가 얼마나 있는지, 데이터셋이 얼마나 큰지에 따라 달라집니다.

## 결정 체크리스트

1. 클러스터 수를 알고 있나요?
   - 예: K-Means 또는 GMM
   - 아니요: DBSCAN(클러스터를 자동으로 찾음) 또는 hierarchical(서로 다른 수준에서 dendrogram 자르기)

2. 클러스터의 모양은 어떤가요?
   - 대체로 구형(blob 형태): K-Means
   - 크기가 다른 타원형: GMM
   - 임의의 모양(초승달, 고리, 사슬): DBSCAN
   - 중첩 또는 계층 구조: hierarchical clustering

3. 데이터에 노이즈나 이상치가 있나요?
   - 예: DBSCAN(noise point를 명시적으로 레이블링) 또는 GMM(확률이 낮은 점을 이상치로 간주)
   - 아니요: K-Means로 충분합니다

4. soft assignment(확률)가 필요한가요?
   - 예: GMM은 각 클러스터에 대해 P(cluster | data point)를 제공합니다
   - 아니요: K-Means 또는 DBSCAN은 hard assignment를 제공합니다

5. 데이터셋 크기는 얼마나 큰가요?
   - 10,000 미만: 어떤 알고리즘이든 동작합니다
   - 10,000에서 1,000,000: K-Means(빠름), Mini-Batch K-Means(더 빠름)
   - 1,000,000 초과: Mini-Batch K-Means 또는 BIRCH. Hierarchical은 너무 느립니다.

## 각 접근법을 사용할 때

**K-Means**: 기본 출발점입니다. 빠르고(O(n * k * iterations)), 단순하며, 많은 문제에서 충분히 좋습니다. K를 고를 때 elbow method나 silhouette score를 사용하세요. 한계: 구형 클러스터를 가정하고, 초기화에 민감하며(K-Means++를 사용하거나 여러 번 실행), 서로 다른 클러스터 크기를 잘 다루지 못합니다.

**DBSCAN**: 임의의 모양을 가진 클러스터를 발견하고 이상치를 자동 탐지하는 데 가장 좋습니다. 매개변수는 eps(이웃 반지름)와 min_samples(최소 밀도) 두 가지입니다. K를 지정할 필요가 없습니다. 한계: 클러스터 밀도가 크게 다르면 어려움을 겪고 eps 튜닝이 까다로울 수 있습니다. eps를 추정하려면 k-distance plot을 사용하세요. 각 점의 k번째 최근접 이웃까지의 거리를 계산하고 정렬한 뒤 elbow를 찾습니다.

**Hierarchical (Agglomerative)**: 병합 트리를 만듭니다. 여러 세밀도에서 클러스터 구조를 탐색하고 싶을 때 유용합니다(서로 다른 높이에서 dendrogram 자르기). Ward's linkage는 조밀한 클러스터에 가장 잘 맞습니다. Single linkage는 길쭉한 클러스터를 찾지만 노이즈에 민감합니다. 한계: O(n^2) memory와 O(n^3) time이 필요해 큰 데이터셋에는 비현실적입니다.

**GMM (Gaussian Mixture Models)**: 확률적 assignment를 사용하는 soft clustering입니다. 각 클러스터를 고유한 평균과 공분산을 가진 Gaussian distribution으로 모델링합니다. 클러스터가 타원형이거나 겹칠 때 K-Means보다 낫습니다. component 수를 고를 때 BIC (Bayesian Information Criterion)를 사용하세요. 한계: Gaussian distribution을 가정하고, 비볼록 모양에서는 실패할 수 있으며, 초기화에 민감합니다.

## 클러스터 품질 평가(레이블 없음)

| 지표 | 측정하는 것 | 범위 | 사용할 때 |
|--------|-----------------|-------|----------|
| Silhouette score | 응집도와 분리도 | -1에서 1(높을수록 좋음) | K 값이나 알고리즘 비교 |
| Inertia (within-cluster SS) | 클러스터의 조밀함 | 0에서 inf(낮을수록 좋음) | K-Means의 elbow method |
| BIC / AIC | 복잡도 페널티를 포함한 모델 적합도 | 낮을수록 좋음 | GMM component 수 선택 |
| Calinski-Harabasz index | 클러스터 간 분산과 클러스터 내 분산의 비율 | 높을수록 좋음 | 빠른 비교 |
| Davies-Bouldin index | 클러스터 간 평균 유사도 | 낮을수록 좋음 | 겹치는 클러스터에 페널티 부여 |

## 흔한 실수

- feature scaling 없이 K-Means 실행하기(스케일이 큰 feature가 거리 계산을 지배함)
- 실제 데이터는 고차원인데 2D 데이터만 눈으로 보고 K를 고르기(silhouette score 사용)
- 비구형 클러스터에 K-Means 사용하기(초승달 또는 고리 모양 데이터에는 DBSCAN 필요)
- DBSCAN eps를 너무 크게(모든 것이 한 클러스터) 또는 너무 작게(모든 것이 noise) 설정하기
- 클러스터 레이블을 ground truth처럼 다루기(클러스터링은 탐색적이며 도메인 지식으로 검증해야 함)
- 20,000개가 넘는 점을 가진 데이터셋에 hierarchical clustering 실행하기(memory와 time이 폭증함)

## 빠른 참조

| 알고리즘 | 클러스터 모양 | K 찾기 | 노이즈 처리 | Soft assignments | 확장성 |
|-----------|--------------|---------|---------------|-----------------|-------------|
| K-Means | 구형 | 아니요(K를 직접 설정) | 아니요 | 아니요 | 수백만 |
| Mini-Batch K-Means | 구형 | 아니요 | 아니요 | 아니요 | 수천만 |
| DBSCAN | 임의 | 예 | 예 | 아니요 | 수십만 |
| Hierarchical | 임의(linkage에 따라 다름) | 유연함(dendrogram 자르기) | linkage에 따라 다름 | 아니요 | 20k 미만 |
| GMM | 타원형 | 아니요(K를 직접 설정) | 부분적(낮은 확률) | 예 | 100k 미만 |
| HDBSCAN | 임의 | 예 | 예 | 부분적 | 수십만 |
