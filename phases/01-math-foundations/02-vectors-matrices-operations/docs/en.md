# 벡터, 행렬, 연산

> 모든 신경망은 몇 가지 단계가 더 붙은 행렬 곱셈일 뿐입니다.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## 학습 목표

- 원소별 연산, 행렬 곱셈, 전치, 행렬식, 역행렬을 갖춘 Matrix 클래스를 만듭니다.
- 원소별 곱셈과 행렬 곱셈을 구분하고 각각이 언제 적용되는지 설명합니다.
- 처음부터 만든 Matrix 클래스만 사용해 단일 dense 신경망 layer(`relu(W @ x + b)`)를 구현합니다.
- 브로드캐스팅 규칙과 신경망 프레임워크에서 bias addition이 작동하는 방식을 설명합니다.

## 문제

신경망을 만들고 싶다고 합시다. 코드를 읽다가 다음을 봅니다:

```text
output = activation(weights @ input + bias)
```

여기서 `@`는 행렬 곱셈입니다. `weights`는 행렬이고 `input`은 벡터입니다. 이 연산들이 무엇을 하는지 모르면 이 줄은 마법처럼 보입니다. 알고 있다면 이 줄은 layer의 전체 forward pass를 세 연산으로 표현한 것입니다.

모델이 처리하는 모든 이미지는 픽셀값 행렬입니다. 모든 word embedding은 벡터입니다. 모든 신경망의 모든 layer는 행렬 변환입니다. 변수를 이해하지 못하고 코드를 쓸 수 없듯, 행렬 연산에 익숙하지 않으면 AI 시스템을 만들 수 없습니다.

이 레슨은 그 유창함을 처음부터 쌓습니다.

## 개념

### 벡터: 순서가 있는 숫자 목록

벡터는 방향과 크기를 가진 숫자 목록입니다. AI에서 벡터는 데이터 포인트, 특징, 파라미터를 나타냅니다.

```text
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D 벡터 `[3, 4]`는 평면 위 좌표 (3, 4)를 가리킵니다. 길이(크기)는 5입니다(3-4-5 삼각형).

### 행렬: 숫자의 격자

행렬은 2D 격자입니다. 행과 열로 이루어집니다. m x n 행렬은 m개 행과 n개 열을 가집니다.

```text
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

신경망에서 가중치 행렬은 입력 벡터를 출력 벡터로 변환합니다. 입력 784개와 출력 128개를 가진 layer는 128x784 가중치 행렬을 사용합니다.

### shape가 중요한 이유

행렬 곱셈에는 엄격한 규칙이 있습니다: `(m x n) @ (n x p) = (m x p)`. 안쪽 차원이 일치해야 합니다.

```text
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

PyTorch에서 shape mismatch 오류가 나면 보통 이 때문입니다.

### 연산 지도

| 연산 | 하는 일 | 신경망에서의 사용 |
|-----------|-------------|-------------------|
| Addition | 원소별 결합 | 출력에 bias 더하기 |
| Scalar multiply | 모든 원소 스케일링 | Learning rate * gradients |
| Matrix multiply | 벡터 변환 | Layer forward pass |
| Transpose | 행과 열 뒤집기 | Backpropagation |
| Determinant | 단일 숫자 요약 | invertibility 확인 |
| Inverse | 변환 되돌리기 | 선형 시스템 풀기 |
| Identity | 아무것도 하지 않는 행렬 | Initialization, residual connections |

### 원소별 곱셈과 행렬 곱셈

이 구분은 초보자를 계속 헷갈리게 합니다.

원소별 곱셈: 같은 위치끼리 곱합니다. 두 행렬은 같은 shape여야 합니다.

```text
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

행렬 곱셈: 행과 열의 내적입니다. 안쪽 차원이 일치해야 합니다.

```text
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

서로 다른 연산이고, 서로 다른 결과를 내며, 규칙도 다릅니다.

### 브로드캐스팅

출력 행렬에 bias 벡터를 더할 때 shape가 맞지 않습니다. 브로드캐스팅은 더 작은 배열을 맞게 늘립니다.

```text
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

현대 프레임워크는 모두 이 작업을 자동으로 합니다. 이를 이해하면 shape가 틀린 것처럼 보이는데 코드가 실행될 때의 혼란을 줄일 수 있습니다.

```figure
vector-projection
```

## 직접 만들기

### Step 1: Vector 클래스

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Step 2: 핵심 연산을 갖춘 Matrix 클래스

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Step 3: 작동 확인하기

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Step 4: 신경망과 연결하기

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

이것이 단일 dense layer입니다: `output = relu(W @ x + b)`. 모든 신경망의 모든 dense layer가 정확히 이 일을 합니다.

## 활용하기

NumPy는 위의 모든 일을 더 적은 줄과 훨씬 빠른 속도로 처리합니다.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python의 `@` 연산자는 `__matmul__`을 호출합니다. NumPy는 C와 Fortran으로 작성된 최적화 BLAS 루틴으로 이를 구현합니다. 같은 수학이지만 100배 빠릅니다.

NumPy의 브로드캐스팅:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy는 1D bias를 두 행 전체에 자동으로 브로드캐스트합니다. 모든 신경망 프레임워크에서 bias addition이 이렇게 작동합니다.

## 결과물

이 레슨은 기하학적 직관으로 행렬 연산을 가르치기 위한 prompt를 만듭니다. `outputs/prompt-matrix-operations.md`를 보세요.

여기서 만든 Matrix 클래스는 Phase 3, Lesson 10에서 만들 미니 신경망 프레임워크의 기반입니다.

## 연습 문제

1. **역행렬을 검증하세요.** `A @ A.inverse_2x2()`를 곱해 단위 행렬이 나오는지 확인하세요. 서로 다른 2x2 행렬 세 개로 시도하세요. 행렬식이 0이면 어떤 일이 일어나나요?

2. **3x3 역행렬을 구현하세요.** adjugate method를 사용해 3x3 행렬의 역행렬을 계산하도록 Matrix 클래스를 확장하세요. NumPy의 `np.linalg.inv`와 비교해 테스트하세요.

3. **두 layer 네트워크를 만드세요.** Matrix 클래스만 사용해(NumPy 없이) 두 layer 신경망을 만드세요: input (3) -> hidden (4) -> output (2). 무작위 가중치를 초기화하고 forward pass를 실행한 뒤 모든 shape가 올바른지 확인하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Vector | "화살표" | 순서가 있는 숫자 목록. AI에서는 고차원 공간의 점입니다. |
| Matrix | "숫자 표" | 선형 변환. 한 공간의 벡터를 다른 공간으로 매핑합니다. |
| Matrix multiply | "그냥 숫자 곱하기" | 첫 번째 행렬의 모든 행과 두 번째 행렬의 모든 열 사이의 내적입니다. 순서가 중요합니다. |
| Transpose | "뒤집기" | 행과 열을 바꿉니다. m x n 행렬을 n x m으로 만듭니다. backpropagation에서 중요합니다. |
| Determinant | "행렬에서 나온 어떤 숫자" | 행렬이 면적(2D) 또는 부피(3D)를 얼마나 스케일하는지 측정합니다. 0이면 변환이 한 차원을 눌러 없앤다는 뜻입니다. |
| Inverse | "행렬 되돌리기" | 변환을 되돌리는 행렬입니다. 행렬식이 0이 아닐 때만 존재합니다. |
| Identity matrix | "지루한 행렬" | 1을 곱하는 것과 같은 행렬입니다. residual connections(ResNets)에 사용됩니다. |
| Broadcasting | "마법 같은 shape 수정" | 작은 배열을 누락된 차원을 따라 반복해 큰 배열과 맞게 늘리는 것. |
| Element-wise | "일반 곱셈" | 같은 위치끼리 곱합니다. 두 배열은 같은 shape이거나 broadcast 가능해야 합니다. |

## 더 읽을거리

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 여기서 다룬 모든 연산에 대한 시각적 직관
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy가 따르는 정확한 규칙
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - ML 특화 선형대수 간결 참고 자료
