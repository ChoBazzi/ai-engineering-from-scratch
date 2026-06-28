# 노름과 거리

> distance function은 "비슷하다"의 의미를 정의합니다. 잘못 고르면 그 뒤의 모든 것이 깨집니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 minutes

## 학습 목표

- L1, L2, cosine, Mahalanobis, Jaccard, edit distance functions를 처음부터 구현하기
- 주어진 ML task에 적절한 distance metric을 선택하고 다른 대안이 실패하는 이유 설명하기
- L1과 L2 norms를 LASSO 및 Ridge regularization과 그 기하학적 constraint regions에 연결하기
- 같은 dataset이 다른 metrics 아래에서 서로 다른 nearest neighbors를 만든다는 것을 시연하기

## 문제

두 vector가 있습니다. word embeddings일 수도 있고 user profiles일 수도 있고 pixel arrays일 수도 있습니다. 알아야 할 것은 이것입니다. 얼마나 가까운가?

답은 어떤 distance function을 고르느냐에 완전히 달려 있습니다. 두 data point는 한 metric에서는 nearest neighbors이고 다른 metric에서는 멀리 떨어져 있을 수 있습니다. KNN classifier, recommendation engine, vector database, clustering algorithm, loss function은 모두 이 선택에 의존합니다. 잘못 고르면 모델은 잘못된 것을 최적화합니다.

보편적으로 최선인 distance는 없습니다. L2는 spatial data에 맞습니다. Cosine similarity는 NLP를 지배합니다. Jaccard는 set을 처리합니다. Edit distance는 string을 처리합니다. Mahalanobis는 correlation을 고려합니다. Wasserstein은 probability mass를 이동합니다. 각각은 "similar"가 무엇인지에 대한 다른 가정을 인코딩합니다.

이 lesson은 주요 distance function을 처음부터 만들고, 각각을 언제 써야 하는지 보여주며, 같은 data가 metric에 따라 완전히 다른 nearest neighbors를 만든다는 것을 시연합니다.

## 개념

### Norms: vector magnitude 측정

norm은 vector의 "size"를 측정합니다. 두 vector 사이의 모든 distance function은 차이의 norm으로 쓸 수 있습니다: d(a, b) = ||a - b||. 따라서 norms를 이해하는 것은 distances를 이해하는 것입니다.

### L1 norm(Manhattan distance) 설명

L1 norm은 모든 component의 absolute value를 합산합니다.

```text
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

city grid에서 axis를 따라만 움직일 수 있을 때 얼마나 걸어야 하는지를 측정하므로 Manhattan distance라고 부릅니다. 대각선은 없습니다.

```text
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

L1을 사용할 때:
- High-dimensional sparse data(text features, one-hot encodings)
- outliers에 robust해야 할 때(하나의 huge difference가 지배하지 않음)
- feature selection problems(L1 regularization은 sparsity를 촉진)

L1 regularization(Lasso)과의 연결: loss function에 ||w||_1을 더하면 absolute weight values의 합을 penalize합니다. 이는 작은 weights를 정확히 0으로 밀어 automatic feature selection을 수행합니다. L1 penalty는 weight space에서 diamond-shaped constraint regions를 만들고, diamond의 corner는 일부 weights가 0인 axis 위에 있습니다.

loss functions와의 연결: Mean Absolute Error(MAE)는 predictions와 targets 사이의 평균 L1 distance입니다. 모든 error를 선형으로 penalize하므로 MSE보다 outliers에 robust합니다.

### L2 norm(Euclidean distance) 설명

L2 norm은 straight-line distance입니다. squared components 합의 square root입니다.

```text
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

기하 시간에 배운 distance입니다. n dimensions의 Pythagoras입니다.

```text
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

L2를 사용할 때:
- Low-to-medium dimensional continuous data
- feature scales가 comparable할 때
- Physical distances(spatial data, sensor readings)
- pixel level의 image similarity

L2 regularization(Ridge)과의 연결: loss function에 ||w||_2^2를 더하면 큰 weights를 penalize합니다. L1과 달리 weights를 0으로 밀지 않습니다. 모든 weights를 비례적으로 0 쪽으로 줄입니다. L2 penalty는 circular constraint regions를 만들므로 axis 위 corner가 없습니다. weights는 작아지지만 거의 정확히 0이 되지는 않습니다.

loss functions와의 연결: Mean Squared Error(MSE)는 L2 distances squared의 평균입니다. 제곱은 작은 error보다 큰 error를 훨씬 강하게 penalize합니다.

```text
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Norms: 일반 family

L1과 L2는 Lp norm의 special cases입니다.

```text
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

p의 값에 따라 서로 다른 모양의 "unit balls"(origin에서 distance 1인 모든 points의 집합)가 만들어집니다.

```text
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity norm(Chebyshev distance) 설명

p가 infinity에 가까워질수록 Lp norm은 최대 absolute component로 수렴합니다.

```text
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

두 point 사이의 distance는 가장 많이 다른 단일 dimension으로 결정됩니다. 다른 모든 dimensions는 무시됩니다.

```text
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

L-infinity를 사용할 때:
- 단일 dimension의 worst-case deviation이 중요할 때
- Game boards(체스의 king은 L-infinity로 움직임: 어느 방향이든 한 칸 비용이 1)
- Manufacturing tolerances(모든 dimension이 spec 안에 있어야 함)

### Cosine Similarity와 Cosine Distance

Cosine similarity는 두 vector 사이의 angle을 측정하며 magnitude는 무시합니다.

```text
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

범위는 -1(opposite directions)부터 +1(same direction)까지입니다. perpendicular vectors의 cosine similarity는 0입니다.

Cosine distance는 이를 distance로 바꿉니다: cosine_distance = 1 - cosine_similarity. 범위는 0(identical direction)부터 2(opposite direction)까지입니다.

```text
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

cosine이 NLP와 embeddings를 지배하는 이유: text에서는 document length가 similarity에 영향을 주면 안 됩니다. cats에 대한 문서가 다른 cats 문서보다 두 배 길어도 여전히 "similar"해야 합니다. Cosine similarity는 magnitude(length)를 무시하고 direction만 봅니다. 같은 word distribution을 가진 길이가 다른 두 문서는 같은 방향을 가리키며 cosine similarity 1.0을 얻습니다.

cosine similarity를 사용할 때:
- Text similarity(TF-IDF vectors, word embeddings, sentence embeddings)
- magnitude가 noise이고 direction이 signal인 모든 domain
- Recommendation systems(user preference vectors)
- Embedding search(vector databases는 거의 항상 cosine 또는 dot product 사용)

### Dot product similarity와 cosine similarity

두 vector의 dot product는 다음과 같습니다.

```text
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity는 두 magnitude로 normalized된 dot product입니다. 두 vector가 이미 unit-normalized(magnitude = 1)이면 dot product와 cosine similarity는 동일합니다.

```text
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

차이점: dot product는 magnitude information을 포함합니다. magnitude가 큰 vector는 더 높은 dot product score를 얻습니다. 일부 retrieval system에서는 "popular" items를 더 높게 ranking하고 싶으므로 이것이 중요합니다. magnitude는 implicit quality 또는 importance signal처럼 작동합니다.

```text
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

실무에서는:
- pure directional similarity를 원하면 cosine similarity를 사용하세요
- magnitude가 의미 있는 정보를 담고 있으면 dot product를 사용하세요
- 많은 vector databases(Pinecone, Weaviate, Qdrant)는 둘 중 선택하게 해줍니다
- embeddings가 L2-normalized라면 선택은 중요하지 않습니다

### Mahalanobis distance 설명

Euclidean distance는 모든 dimension을 동일하게 취급합니다. 그러나 features가 correlated이거나 scale이 다르면 L2는 misleading results를 줍니다.

Mahalanobis distance는 data의 covariance structure를 고려합니다.

```text
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

여기서 S는 data의 covariance matrix입니다.

직관적으로 Mahalanobis distance는 먼저 data를 decorrelate하고 normalize(whitening)한 뒤 그 transformed space에서 L2 distance를 계산합니다. S가 identity matrix(uncorrelated, unit variance features)이면 Mahalanobis distance는 Euclidean distance로 줄어듭니다.

```text
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Mahalanobis distance를 사용할 때:
- Outlier detection(mean에서 Mahalanobis distance가 큰 point는 outlier)
- features의 scales와 correlations가 다른 classification
- reliable covariance matrix를 추정할 충분한 data가 있을 때
- 제조 quality control(multivariate process monitoring)

### Jaccard Similarity (sets용)

Jaccard similarity는 두 set 사이의 overlap을 측정합니다.

```text
J(A, B) = |A intersect B| / |A union B|
```

범위는 0(no overlap)부터 1(identical sets)까지입니다. Jaccard distance = 1 - Jaccard similarity입니다.

```text
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Jaccard를 사용할 때:
- tags, categories, features의 set 비교
- word presence 기반 document similarity(frequency가 아님)
- Near-duplicate detection(MinHash approximation of Jaccard)
- binary feature vectors(presence/absence data) 비교
- segmentation models 평가(Intersection over Union = Jaccard)

### Edit distance(Levenshtein distance) 설명

Edit distance는 한 string을 다른 string으로 바꾸는 데 필요한 최소 single-character operations 수를 셉니다. operations는 insert, delete, substitute입니다.

```text
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

dynamic programming으로 계산합니다. entry (i, j)가 string A의 처음 i개 characters와 string B의 처음 j개 characters 사이의 edit distance인 matrix를 채웁니다.

```text
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

edit distance를 사용할 때:
- Spell checking and correction
- DNA sequence alignment(weighted operations 포함)
- Fuzzy string matching
- 지저분한 text data의 deduplication

### KL Divergence (distance는 아니지만 distance처럼 사용됨)

KL divergence는 한 probability distribution이 다른 distribution과 얼마나 다른지 측정합니다. Lesson 09에서 다뤘지만, 사람들이 이것을 "distance"처럼 사용하기 때문에 이 논의에 속합니다.

```text
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

중요한 성질: KL divergence는 symmetric이 아닙니다.

```text
D_KL(P || Q) != D_KL(Q || P)
```

따라서 distance metric의 기본 요구사항을 만족하지 않습니다. triangle inequality도 만족하지 않습니다. distance가 아니라 divergence입니다.

Forward KL(D_KL(P || Q))은 "mean-seeking"입니다. Q는 P의 모든 mode를 덮으려 합니다.
Reverse KL(D_KL(Q || P))은 "mode-seeking"입니다. Q는 P의 단일 mode에 집중합니다.

KL divergence를 보는 곳:
- VAEs(ELBO의 KL term이 latent distribution을 prior 쪽으로 밀어냄)
- Knowledge distillation(student가 teacher distribution에 맞추려 함)
- RLHF(KL penalty가 fine-tuned model을 base model에 가깝게 유지)
- Policy gradient methods(policy updates 제한)

### Wasserstein distance(Earth Mover's Distance) 설명

Wasserstein distance는 한 probability distribution을 다른 distribution으로 바꾸는 데 필요한 최소 "work"를 측정합니다. 한 distribution이 흙더미이고 다른 distribution이 구멍이라면, 얼마나 많은 흙을 얼마나 멀리 옮겨야 하는지라고 생각하세요.

```text
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

1D distributions에서는 cumulative distribution functions의 absolute difference 적분으로 단순화됩니다.

```text
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Wasserstein이 중요한 이유:
- true metric입니다(symmetric, triangle inequality 만족)
- distributions가 겹치지 않아도 gradients를 제공합니다(KL divergence는 infinity로 감)
- 이 성질 덕분에 original GAN의 training instability를 해결한 Wasserstein GANs(WGANs)의 중심이 되었습니다

```text
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Wasserstein을 사용할 때:
- GAN training(WGAN, WGAN-GP)
- 겹치지 않을 수 있는 distributions 비교
- Optimal transport problems
- Image retrieval(color histograms 비교)

### 왜 task마다 다른 distance가 필요한가

| Task | Best distance | Why |
|------|--------------|-----|
| Text similarity | Cosine | Magnitude is noise, direction is meaning |
| Image pixel comparison | L2 | Spatial relationships matter, features are comparable scale |
| Sparse high-dim features | L1 | Robust, does not amplify rare large differences |
| Set overlap (tags, categories) | Jaccard | Data is naturally set-valued, not vectorial |
| String matching | Edit distance | Operations map to human editing intuition |
| Outlier detection | Mahalanobis | Accounts for feature correlations and scales |
| Comparing distributions | KL divergence | Measures information lost by using Q instead of P |
| GAN training | Wasserstein | Provides gradients even when distributions do not overlap |
| Embeddings (vector DB) | Cosine or dot product | Embeddings are trained to encode meaning in direction |
| Recommendation | Dot product | Magnitude can encode popularity or confidence |
| DNA sequences | Weighted edit distance | Substitution costs vary by nucleotide pair |
| Manufacturing QC | L-infinity | Worst-case deviation in any dimension matters |

### Loss Functions와의 연결

Loss functions는 predictions vs targets에 적용된 distance functions입니다.

```text
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Regularization과의 연결

Regularization은 weights에 대한 norm penalty를 loss function에 더합니다.

```text
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

L1이 sparsity를 만들고 L2가 그렇지 않은 이유: 2D weight space에서 constraint region을 그려보세요. L1은 diamond이고 L2는 circle입니다. loss function의 contours(ellipses)는 한 weight가 0인 corner에서 diamond에 닿을 가능성이 큽니다. circle에는 smooth point에서 닿으므로 두 weights가 모두 nonzero입니다.

### Nearest neighbor search 설명

모든 distance function은 nearest neighbor search problem을 암시합니다. query point가 주어졌을 때 dataset에서 가장 가까운 points를 찾는 문제입니다.

exact nearest neighbor search는 n points와 d dimensions를 가진 dataset에서 query당 O(n * d)입니다. 큰 dataset에서는 너무 느립니다.

Approximate Nearest Neighbor(ANN) algorithms는 작은 accuracy 손실을 massive speed gain과 교환합니다.

```text
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW(Hierarchical Navigable Small World)는 현대 vector databases에서 지배적인 algorithm입니다. 각 node가 approximate nearest neighbors와 연결되는 multi-layer graph를 만듭니다. Search는 top layer(sparse, long jumps)에서 시작해 bottom layer(dense, short jumps)로 내려갑니다.

```figure
norm-unit-balls
```

## 직접 만들기

### Step 1: 모든 norm과 distance functions

complete implementation은 `code/distances.py`를 보세요. 모든 function은 basic Python math만 사용해 처음부터 만들어집니다.

### Step 2: 같은 data, 다른 distances, 다른 neighbors

`distances.py`의 demo는 dataset을 만들고 query point를 선택한 뒤 distance metric에 따라 nearest neighbor가 어떻게 바뀌는지 보여줍니다. L1에서 "closest"인 point가 L2나 cosine에서는 closest가 아닐 수 있습니다.

### Step 3: Embedding similarity search 구현

코드에는 cosine similarity와 L2 distance를 사용해 query와 가장 비슷한 "documents"를 찾는 mock embedding similarity search가 포함되어 있으며 ranking이 달라질 수 있음을 보여줍니다.

## 사용하기

가장 흔한 실용적 사용: vector database에서 similar items 찾기.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

`model.encode(text)`를 호출한 뒤 vector database를 검색할 때 내부에서는 이런 일이 일어납니다. embedding model이 text를 vector로 mapping합니다. vector database는 query vector와 저장된 모든 vector 사이의 cosine similarity(또는 dot product)를 계산하고, 모든 vector를 확인하지 않기 위해 ANN algorithms를 사용합니다.

## 연습문제

1. (1, 2, 3)과 (4, 0, 6) 사이의 L1, L2, L-infinity distances를 계산하세요. 어떤 point pair에 대해서도 L-inf <= L2 <= L1이 항상 성립함을 확인하세요. 이 ordering이 보장되는 이유를 증명하세요.

2. cosine similarity는 높지만(> 0.9) L2 distance는 큰(> 10) 두 vector를 만드세요. 기하학적으로 무슨 일이 일어나는지 설명하세요. 그런 다음 cosine similarity는 낮지만(< 0.3) L2 distance는 작은(< 0.5) 두 vector를 만드세요.

3. dataset과 query point를 받아 L1, L2, cosine, Mahalanobis distance 각각에서 nearest neighbor를 반환하는 함수를 구현하세요. 네 metric이 모두 어떤 point가 nearest인지 disagree하는 dataset을 찾으세요.

4. CDF method를 사용해 [0.5, 0.5, 0, 0]과 [0, 0, 0.5, 0.5] 사이의 Wasserstein distance를 손으로 계산하세요. 그런 다음 [0.25, 0.25, 0.25, 0.25]와 [0, 0, 0.5, 0.5] 사이도 계산하세요. 어느 쪽이 더 크고 왜인가요?

5. approximate Jaccard similarity를 위한 MinHash를 구현하세요. random sets 100개를 생성하고 모든 pairs에 대해 exact Jaccard를 계산한 뒤, 50, 100, 200 hash functions를 사용한 MinHash approximation과 비교하세요. approximation error를 plot하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| Norm | "vector의 size" | vector를 non-negative scalar로 mapping하고 triangle inequality, absolute homogeneity, zero only for zero vector를 만족하는 function |
| L1 norm | "Manhattan distance" | component absolute values의 합. optimization에서 sparsity를 만듭니다. outliers에 robust합니다 |
| L2 norm | "Euclidean distance" | squared components 합의 square root. Euclidean space의 straight-line distance |
| Lp norm | "Generalized norm" | absolute components의 p-th powers 합의 p-th root. L1과 L2는 special cases |
| L-infinity norm | "Max norm" 또는 "Chebyshev distance" | 최대 absolute component value. p가 infinity로 갈 때 Lp의 limit |
| Cosine similarity | "vectors 사이의 angle" | magnitudes로 normalized한 dot product. -1부터 +1까지. vector length를 무시 |
| Cosine distance | "1 minus cosine similarity" | cosine similarity를 distance로 변환. 0부터 2까지 |
| Dot product | "Unnormalized cosine" | component-wise products의 합. cosine similarity에 두 magnitude를 곱한 것과 같음 |
| Mahalanobis distance | "Correlation-aware distance" | data covariance matrix로 whitened(decorrelated and normalized)된 space에서의 L2 distance |
| Jaccard similarity | "Set overlap" | intersection size를 union size로 나눈 값. vectors가 아니라 sets용 |
| Edit distance | "Levenshtein distance" | 한 string을 다른 string으로 바꾸는 최소 insertions, deletions, substitutions |
| KL divergence | "distributions 사이의 distance" | true distance가 아님(not symmetric). Q로 P를 encode할 때 생기는 extra bits를 측정 |
| Wasserstein distance | "Earth mover's distance" | 한 distribution에서 다른 distribution으로 mass를 운반하는 최소 work. true metric |
| Approximate nearest neighbor | "ANN search" | exact search보다 훨씬 빠르게 approximate closest points를 찾는 algorithms(HNSW, LSH, IVF) |
| HNSW | "vector DB algorithm" | Hierarchical Navigable Small World graph. fast approximate nearest neighbor search를 위한 multi-layer graph |
| L1 regularization | "Lasso" | weights의 L1 norm을 loss에 더함. weights를 0으로 몰아 sparsity 생성 |
| L2 regularization | "Ridge" 또는 "weight decay" | weights의 squared L2 norm을 loss에 더함. sparsity 없이 weights를 0 쪽으로 shrink |
| Elastic Net | "L1 + L2" | L1과 L2 regularization 결합. correlated feature groups를 더 잘 처리 |

## 더 읽을거리

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - billion-scale ANN search를 위한 Meta library
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - Earth Mover's distance를 GAN에 도입한 paper
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - foundational ANN algorithm
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - embeddings에서 cosine similarity가 기본이 된 Word2Vec paper
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn의 distance metrics와 neighbor algorithms 실무 가이드
