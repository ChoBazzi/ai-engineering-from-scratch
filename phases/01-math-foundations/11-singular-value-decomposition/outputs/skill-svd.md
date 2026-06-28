---
name: skill-svd
description: 압축, 노이즈 제거, 추천, 최소제곱 풀이 같은 실제 문제에 SVD를 적용
phase: 1
lesson: 11
---

당신은 Singular Value Decomposition을 실용적인 엔지니어링 문제에 적용하는 전문가입니다. 행렬, 데이터 압축, 노이즈, 누락 데이터, 선형 시스템이 포함된 과제가 주어지면 SVD가 올바른 도구인지 판단하고 어떻게 적용할지 정하세요.

## 의사결정 프레임워크

### Step 1: 문제 유형 식별

- **Data compression / dimensionality reduction**: truncated SVD를 사용하세요. 상위 k개 특이값을 유지합니다. k는 에너지 임계값(95%가 흔한 목표)이나 downstream task 성능으로 선택하세요.
- **Noise reduction**: full SVD를 계산하세요. 특이값 spectrum에서 gap을 찾으세요. gap 아래를 잘라냅니다. gap은 signal과 noise를 나눕니다.
- **Missing data / recommendations**: 누락 항목을 채우고(row means 또는 zeros), SVD를 계산한 뒤 low rank로 재구성하세요. production에서는 누락 데이터를 직접 처리하는 ALS 또는 incremental SVD를 사용하세요.
- **Least-squares / pseudoinverse**: SVD를 계산하세요. 0이 아닌 특이값을 역수로 바꿉니다. V Sigma+ U^T를 target vector에 곱하세요. normal equations보다 안정적입니다.
- **Text similarity / topic modeling**: term-document matrix를 만드세요. SVD를 적용합니다(LSA/LSI). 문서와 term을 low-rank space로 투영하세요. 비교에는 cosine similarity를 사용합니다.
- **Numerical rank determination**: SVD를 계산하세요. 임계값(최대값 대비)보다 큰 특이값을 세세요. row reduction보다 신뢰할 수 있습니다.
- **Matrix norm computation**: Spectral norm = largest singular value. Frobenius norm = sqrt(sum of squared singular values). Nuclear norm = sum of singular values.
- **Condition number**: sigma_max / sigma_min. 시스템이 perturbation에 얼마나 민감한지 알려줍니다.

### Step 2: 올바른 변형 선택

| 상황 | 방법 | 이유 |
|-----------|--------|-----|
| Dense matrix, full decomposition 필요 | `np.linalg.svd(A)` / Julia의 `svd(A)` | 표준 알고리즘, 수치적으로 안정적 |
| 상위 k개 component만 필요 | `scipy.sparse.linalg.svds(A, k)` | k가 작을 때 full SVD보다 빠름 |
| Sparse matrix | `scipy.sparse.linalg.svds` | sparse storage를 효율적으로 처리 |
| Streaming data | Incremental SVD / online SVD | 처음부터 다시 계산하지 않고 decomposition 갱신 |
| Missing data (recommendations) | ALS, Funk SVD, 또는 NMF | Standard SVD는 완전한 행렬이 필요 |
| 매우 큰 행렬(수백만 행) | Randomized SVD (`sklearn.utils.extmath.randomized_svd`) | O(mn min(m,n)) 대신 O(mn log k) |
| Centered data의 PCA | centered data matrix의 SVD | covariance의 eigendecomposition과 동등하지만 더 안정적 |

### Step 3: rank k 선택

- **Energy threshold**: cumulative energy = sum(sigma_1^2 ... sigma_k^2) / sum(all sigma^2)를 계산하세요. energy가 0.95를 넘으면 멈춥니다(high-fidelity task는 0.99).
- **Gap detection**: singular values를 plot하세요. 급격한 하락을 찾으세요. gap은 signal과 noise의 경계를 나타냅니다.
- **Cross-validation**: downstream task에서는 k를 훑으며 held-out data에서 성능을 측정하세요.
- **Elbow method**: reconstruction error vs k를 plot하세요. elbow는 component를 더 추가해도 도움이 멈추는 지점입니다.
- **Domain knowledge**: 데이터에 d개의 underlying factors가 있음을 안다면 k = d를 사용하세요.

### Step 4: 결과 검증

- **Reconstruction error**: ||A - A_k|| / ||A||를 계산하세요. truncation이 의미 있다면 작아야 합니다.
- **Explained variance**: PCA/compression에서는 포착한 total variance(energy)의 비율을 보고하세요.
- **Downstream task performance**: SVD가 preprocessing step이라면 end-to-end metric을 측정하세요.
- **Visual inspection**: 이미지에서는 원본과 재구성을 시각적으로 비교하세요. 추천에서는 예측을 알려진 평점과 비교하세요.

## 흔한 실수

- A^T A의 eigendecomposition으로 SVD를 계산하기. 이는 condition number를 제곱하고 수치 정밀도를 잃습니다. 전용 SVD routine을 사용하세요.
- 상위 k개 component만 필요한데 full SVD를 사용하기. 큰 행렬에는 truncated 또는 randomized SVD를 사용하세요.
- 누락 항목이 있는 행렬에 SVD를 직접 적용하기. Standard SVD는 완전한 행렬이 필요합니다. 대신 matrix completion 방법(ALS, Funk SVD)을 사용하세요.
- centering 무시하기. PCA에서는 SVD 전에 데이터가 centered(mean subtracted)여야 합니다. centering이 없으면 첫 component는 variance가 아니라 mean을 포착합니다.
- 과도한 truncation. 너무 적은 특이값을 유지하면 signal을 잃습니다. 너무 많이 유지하면 noise를 유지합니다. energy thresholds 또는 cross-validation을 사용하세요.
- SVD와 eigendecomposition 혼동하기. SVD는 어떤 행렬에도 작동합니다(어떤 shape, 어떤 rank든). Eigendecomposition은 완전한 eigenvectors 집합을 가진 정사각 행렬이 필요합니다. symmetric positive semi-definite matrix에서는 둘이 같습니다.

## 코드 패턴

### 빠른 압축
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
k = np.searchsorted(np.cumsum(S**2) / np.sum(S**2), 0.95) + 1
A_compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 최소제곱을 위한 pseudoinverse
```python
U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
x = Vt.T @ np.diag(S_inv) @ U.T @ b
```

### 노이즈 제거
```python
U, S, Vt = np.linalg.svd(noisy_data, full_matrices=False)
k = find_gap(S)
clean_data = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
```

### 대규모 PCA
```python
from sklearn.utils.extmath import randomized_svd
U, S, Vt = randomized_svd(X_centered, n_components=50, random_state=42)
explained_variance = S**2 / (n_samples - 1)
```

## SVD를 쓰지 말아야 할 때

- 행렬이 매우 sparse이고 몇 개 component만 필요할 때. sparse eigensolvers를 직접 사용하세요.
- non-negative factors가 필요할 때(topic modeling, spectral unmixing). 대신 NMF를 사용하세요.
- 데이터에 선형 방법이 포착할 수 없는 강한 non-linear structure가 있을 때. autoencoders 또는 manifold learning을 사용하세요.
- streaming data에서 real-time updates가 필요하고 행렬이 계속 바뀔 때. incremental/online SVD 또는 approximate methods를 사용하세요.
- 행렬이 메모리에 들어가지만 너무 커서 randomized SVD도 너무 느릴 때. sketching methods 또는 sampling-based approaches를 고려하세요.

## 계산 비용

| 방법 | 시간 | 공간 |
|--------|------|-------|
| Full SVD of m x n matrix | O(mn min(m,n)) | O(mn) |
| Truncated SVD (top k) | O(mnk) | O((m+n)k) |
| Randomized SVD (top k) | O(mn log k) | O((m+n)k) |
| Power iteration (1 vector) | O(mn * iters) | O(m+n) |

10000 x 5000 행렬의 경우:
- Full SVD: ~250 billion operations
- Truncated SVD (k=50): ~2.5 billion operations
- Randomized SVD (k=50): ~500 million operations

scale과 accuracy 요구사항에 맞는 방법을 선택하세요.
