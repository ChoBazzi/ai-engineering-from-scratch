# JAX 소개

> PyTorch는 텐서를 변경합니다. TensorFlow는 그래프를 만듭니다. JAX는 순수 함수를 컴파일합니다. 마지막 방식은 딥러닝을 생각하는 방법 자체를 바꿉니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, 기본 NumPy
**Time:** ~90 minutes

## 학습 목표

- JAX의 functional API(jax.numpy, jax.grad, jax.jit, jax.vmap)를 사용해 pure-function 신경망 코드를 작성합니다
- PyTorch의 eager mutation과 JAX의 functional compilation model 사이의 핵심 설계 차이를 설명합니다
- jit compilation과 vmap vectorization을 적용해 naive Python 대비 학습 루프를 가속합니다
- JAX에서 단순한 네트워크를 학습하고 명시적 상태 관리를 PyTorch의 객체지향 접근 방식과 대조합니다

## 문제

이제 PyTorch로 신경망을 만드는 방법을 알고 있습니다. `nn.Module`을 정의하고, `.backward()`를 호출하고, optimizer를 step합니다. 잘 동작합니다. 수백만 명이 이 방식을 사용합니다.

하지만 PyTorch에는 DNA에 박힌 제약이 있습니다. Python에서 연산을 하나씩 eager하게 추적한다는 점입니다. 모든 `tensor + tensor`는 별도의 kernel launch입니다. 모든 training step은 같은 Python 코드를 다시 해석합니다. 2,048개의 TPU에 걸쳐 5,400억 파라미터 모델을 학습해야 하기 전까지는 잘 동작합니다. 그 순간 overhead가 치명적이 됩니다.

Google DeepMind는 Gemini를 JAX로 학습합니다. Anthropic은 Claude를 JAX로 학습했습니다. 이것은 작은 작업이 아닙니다. 지구상에서 가장 큰 신경망 학습 실행들입니다. 그들이 JAX를 선택한 이유는 JAX가 학습 루프를 Python 호출의 나열이 아니라 컴파일 가능한 프로그램으로 다루기 때문입니다.

JAX는 세 가지 초능력을 가진 NumPy입니다: 자동 미분, XLA로의 JIT compilation, 자동 vectorization입니다. 하나의 예제를 처리하는 함수를 작성합니다. 그러면 JAX는 배치를 처리하고, 그래디언트를 계산하고, machine code로 컴파일하며, 여러 device에서 실행되는 함수를 제공합니다. 원래 함수를 바꾸지 않고 모두 가능합니다.

## 개념

### JAX 철학

JAX는 functional framework입니다. 클래스도, mutable state도, `.backward()` 메서드도 없습니다. 대신:

| PyTorch | JAX |
|---------|-----|
| state를 가진 `nn.Module` 클래스 | Pure function: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | XLA를 통한 JIT compilation |
| `for x in batch:` 수동 루프 | `jax.vmap(f)` auto-vectorization |
| `DataParallel` / `FSDP` | `jax.pmap(f)` auto-parallelism |
| Mutable `model.parameters()` | 배열의 immutable pytree |

이것은 스타일 선호가 아닙니다. 컴파일러 제약입니다. JIT compilation은 pure function을 요구합니다. 같은 입력은 항상 같은 출력을 만들고 side effect가 없어야 합니다. 이 제약이 100배 속도 향상을 가능하게 합니다.

### jax.numpy: 익숙한 표면

JAX는 accelerator 위에서 NumPy API를 다시 구현합니다:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

같은 함수 이름, 같은 broadcasting 규칙, 같은 slicing semantics입니다. 하지만 배열은 GPU/TPU 위에 있고 모든 연산은 컴파일러가 trace할 수 있습니다.

중요한 차이 하나가 있습니다. JAX 배열은 immutable입니다. `a[0] = 5`는 없습니다. 대신 `a = a.at[0].set(5)`를 사용합니다. 일주일 정도는 어색하지만 곧 이해됩니다. immutability가 `grad`, `jit`, `vmap` 같은 transformation을 composable하게 만들기 때문입니다.

### jax.grad: 함수형 자동 미분

PyTorch는 그래디언트를 텐서에 붙입니다(`.grad`). JAX는 그래디언트를 함수에 붙입니다.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad`는 함수를 받아 그래디언트를 계산하는 새 함수를 반환합니다. `.backward()` 호출은 없습니다. 텐서에 저장되는 computation graph도 없습니다. 그래디언트는 호출하고, 합성하고, JIT-compile할 수 있는 또 하나의 함수일 뿐입니다.

이 방식은 자유롭게 합성됩니다:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

2차 도함수, 3차 도함수, Jacobian, Hessian 모두 `grad`를 합성해서 구합니다. PyTorch도 이것을 할 수 있지만(`torch.autograd.functional.hessian`) 덧붙여진 기능에 가깝습니다. JAX에서는 이것이 기초입니다.

제약은 `grad`가 pure function에서만 동작한다는 점입니다. 내부에 print 문을 둘 수 없습니다(실행이 아니라 tracing 중에 실행됩니다). 외부 상태를 변경할 수 없습니다. 명시적 key 관리 없이 난수를 생성할 수 없습니다.

### jit: XLA로 컴파일하기

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

첫 호출에서 JAX는 함수를 trace합니다. 연산을 실행하지 않고 어떤 연산이 일어나는지 기록합니다. 그런 다음 그 trace를 TPU와 GPU를 위한 Google의 컴파일러인 XLA(Accelerated Linear Algebra)에 넘깁니다. XLA는 연산을 fuse하고, 불필요한 메모리 복사를 제거하며, 최적화된 machine code를 생성합니다.

이후 호출은 Python을 완전히 건너뜁니다. 컴파일된 코드는 accelerator에서 C++ 속도로 실행됩니다.

JIT가 도움이 되는 경우:
- Training step(같은 계산이 수천 번 반복됨)
- 추론(같은 모델, 다른 입력)
- 비슷한 shape의 입력으로 두 번 이상 호출되는 함수

JIT가 해가 되는 경우:
- 값에 의존하는 Python control flow가 있는 함수(`x`가 traced array인 `if x > 0`)
- 일회성 계산(compilation overhead가 실행 시간을 초과함)
- 디버깅(tracing이 실제 실행을 가림)

control flow 제약은 실제입니다. `jax.lax.cond`가 `if/else`를 대체합니다. `jax.lax.scan`이 `for` 루프를 대체합니다. 이것들은 선택 사항이 아닙니다. 컴파일의 대가입니다.

### vmap: 자동 Vectorization

하나의 예제를 처리하는 함수를 작성합니다:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`은 이를 배치를 처리하도록 끌어올립니다:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)`은 `params`는 batch하지 않고(공유), `x`의 axis 0을 따라 batch하라는 뜻입니다. 수동 `for` 루프도, reshape도, batch dimension을 직접 끼워 넣는 일도 없습니다. JAX가 batch dimension을 알아내고 전체 계산을 vectorize합니다.

이것은 syntactic sugar가 아닙니다. `vmap`은 Python 루프보다 10-100배 빠르게 실행되는 fused vectorized code를 생성합니다. 그리고 `jit`, `grad`와 합성됩니다:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

예제별 그래디언트입니다. 한 줄입니다. PyTorch에서는 hack 없이는 거의 불가능합니다.

### pmap: Device 간 Data Parallelism

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap`은 사용 가능한 모든 device(GPU/TPU)에 함수를 복제하고 배치를 나눕니다. 함수 안에서 `jax.lax.pmean`과 `jax.lax.psum`이 device 간 그래디언트를 동기화합니다.

Google은 `pmap`(그리고 그 후속인 `shard_map`)을 사용해 수천 개의 TPU v5e 칩에 걸쳐 Gemini를 학습합니다. 프로그래밍 모델은 단순합니다. single-device 버전을 작성하고, `pmap`으로 감싸면 끝입니다.

### Pytrees: 범용 데이터 구조

JAX는 "pytrees" 위에서 동작합니다. list, tuple, dict, array의 중첩 조합입니다. 모델 파라미터는 pytree입니다:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

모든 JAX transformation인 `grad`, `jit`, `vmap`은 pytree를 순회하는 방법을 알고 있습니다. `jax.tree.map(f, tree)`는 모든 leaf에 `f`를 적용합니다. optimizer가 모든 파라미터를 한 번에 업데이트하는 방식입니다:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

`.parameters()` 메서드는 없습니다. 파라미터 등록도 없습니다. tree 구조 자체가 모델입니다.

### 함수형 vs 객체지향

PyTorch는 state를 객체 안에 저장합니다:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX는 명시적 state를 가진 pure function을 사용합니다:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params는 인자로 전달됩니다. 아무것도 저장되지 않습니다. 아무것도 변경되지 않습니다. 그래서 모든 함수가 testable, composable, compilable해집니다. 동시에 params를 직접 관리해야 한다는 뜻이기도 합니다. 아니면 Flax나 Equinox 같은 라이브러리를 사용합니다.

### JAX 생태계

JAX는 primitive를 제공합니다. 라이브러리는 ergonomics를 제공합니다:

| 라이브러리 | 역할 | 스타일 |
|---------|------|-------|
| **Flax** (Google) | 신경망 레이어 | 명시적 state를 가진 `nn.Module` |
| **Equinox** (Patrick Kidger) | 신경망 레이어 | Pytree 기반, Pythonic |
| **Optax** (DeepMind) | Optimizer + LR schedule | Composable gradient transforms |
| **Orbax** (Google) | Checkpointing | pytree 저장/복원 |
| **CLU** (Google) | Metric + logging | 학습 루프 유틸리티 |

Optax는 표준 optimizer 라이브러리입니다. gradient transformation(Adam, SGD, clipping)을 파라미터 업데이트와 분리해 쉽게 합성할 수 있게 합니다:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### JAX와 PyTorch를 언제 사용할까

| 요인 | JAX | PyTorch |
|--------|-----|---------|
| TPU 지원 | First-class(Google이 둘 다 만듦) | 커뮤니티 유지(torch_xla) |
| GPU 지원 | 좋음(XLA를 통한 CUDA) | 최고 수준(native CUDA) |
| 디버깅 | 어려움(tracing + compilation) | 쉬움(eager, line-by-line) |
| 생태계 | 연구 중심(Flax, Equinox) | 거대함(HuggingFace, torchvision 등) |
| 채용 | niche(Google/DeepMind/Anthropic) | mainstream(어디에나 있음) |
| 대규모 학습 | 우수함(XLA, pmap, mesh) | 좋음(FSDP, DeepSpeed) |
| 프로토타이핑 속도 | 느림(functional overhead) | 빠름(mutate and go) |
| 프로덕션 추론 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 사용 주체 | DeepMind(Gemini), Anthropic(Claude) | Meta(Llama), OpenAI(GPT), Stability AI |

솔직한 답은 이렇습니다. JAX를 써야 하는 구체적인 이유가 없다면 PyTorch를 사용하세요. 그 이유란 TPU 접근, 예제별 그래디언트 필요, 거대한 규모의 multi-device training, 또는 Google/DeepMind/Anthropic에서 일하는 경우입니다.

### JAX의 난수

JAX에는 global random state가 없습니다. 모든 random operation에는 명시적 PRNG key가 필요합니다:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

처음에는 번거롭습니다. 하지만 device와 compilation을 가로질러 재현성을 보장합니다. PyTorch의 `torch.manual_seed`가 multi-GPU setting에서 보장할 수 없는 속성입니다.

```figure
batchnorm-effect
```

## 만들어 보기

### 1단계: 설정과 데이터

JAX와 Optax를 사용해 MNIST에서 3층 MLP를 학습합니다. 입력 784개, 256개와 128개 뉴런의 hidden layer 두 개, 출력 클래스 10개입니다.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 2단계: 파라미터 초기화

클래스는 없습니다. pytree를 반환하는 함수만 있습니다:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization을 수동으로 수행합니다. 하나의 seed에서 세 개의 PRNG key를 나눕니다. 모든 weight는 중첩 dict 안의 immutable array입니다.

### 3단계: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Pure function입니다. Params가 들어가고 prediction이 나옵니다. `self`도, 저장된 state도 없습니다. `loss_fn`은 cross-entropy를 처음부터 계산합니다. softmax, log, negative mean입니다.

### 4단계: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad`는 한 번의 pass에서 loss 값과 그래디언트를 모두 반환합니다. `@jax.jit` decorator는 두 함수를 XLA로 컴파일합니다. 첫 호출 이후 각 training step은 Python을 거치지 않고 실행됩니다.

### 5단계: 학습 루프

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs입니다. 테스트 정확도는 약 97%입니다. 첫 epoch는 느립니다(JIT compilation). epoch 2-10은 빠릅니다.

없는 것에 주목하세요. `.zero_grad()`도, `.backward()`도, `.step()`도 없습니다. 전체 업데이트는 하나의 합성된 함수 호출입니다. 그래디언트가 계산되고, Adam으로 변환되고, 파라미터에 적용됩니다. 모두 `train_step` 안에서 일어납니다.

## 사용하기

### Flax: Google 표준

Flax는 가장 흔한 JAX 신경망 라이브러리입니다. `nn.Module`을 다시 추가하지만, 명시적 state management를 사용합니다:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

PyTorch와 같은 구조지만 `params`는 모델과 분리되어 있습니다. `model.init()`이 params를 만듭니다. `model.apply(params, x)`가 forward pass를 실행합니다. 모델 객체에는 state가 없습니다.

### Equinox: Pythonic 대안

Equinox(Patrick Kidger 제작)는 모델을 pytree로 표현합니다:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

모델 자체가 pytree입니다. `.apply()`가 필요 없습니다. 파라미터는 모델의 leaf일 뿐입니다. 이것은 JAX의 사고방식에 더 가깝습니다.

### Optax: 합성 가능한 Optimizer

Optax는 gradient transformation을 업데이트와 분리합니다:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, learning rate warmup, weight decay가 모두 transform chain으로 합성됩니다. 각 transform은 그래디언트를 보고, 수정하고, 다음 transform에 전달합니다. 거대한 단일 optimizer 클래스는 없습니다.

## 내보내기

**설치:**

```bash
pip install jax jaxlib optax flax
```

GPU 지원:

```bash
pip install jax[cuda12]
```

TPU용(Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**성능 관련 주의점:**

- 첫 JIT 호출은 느립니다(compilation). benchmark 전에 warm up하세요.
- JIT 내부에서 JAX 배열에 대한 Python 루프를 피하세요. `jax.lax.scan` 또는 `jax.lax.fori_loop`를 사용하세요.
- `jax.debug.print()`는 JIT 내부에서 동작합니다. 일반 `print()`는 동작하지 않습니다.
- `jax.profiler` 또는 TensorBoard로 프로파일링하세요. XLA compilation은 bottleneck을 숨길 수 있습니다.
- JAX는 기본적으로 GPU 메모리의 75%를 미리 할당합니다. 비활성화하려면 `XLA_PYTHON_CLIENT_PREALLOCATE=false`를 설정하세요.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**이 레슨이 만드는 것:**
- `outputs/prompt-jax-optimizer.md` -- 올바른 JAX optimizer 설정을 선택하기 위한 prompt
- `outputs/skill-jax-patterns.md` -- JAX의 functional pattern을 다루는 skill

## 연습 문제

1. MLP에 dropout을 추가하세요. JAX에서 dropout은 PRNG key가 필요합니다. forward pass를 통해 key를 전달하고 각 dropout layer마다 split하세요. 적용 전후의 테스트 정확도를 비교하세요.

2. `jax.vmap`을 사용해 32개 MNIST 이미지 배치에 대한 예제별 그래디언트를 계산하세요. 각 예제의 gradient norm을 계산하세요. 어떤 예제가 가장 큰 그래디언트를 가지며, 왜 그런가요?

3. 수동 forward 함수를 임의 개수의 layer에서 동작하는 generic `mlp_forward(params, x)`로 바꾸세요. `jax.tree.leaves`를 사용해 depth를 자동으로 결정하세요.

4. `@jax.jit` 사용 여부에 따라 training step을 benchmark하세요. 각각 100 step의 시간을 측정하세요. 여러분의 하드웨어에서 속도 향상은 얼마나 큰가요? 첫 호출의 compilation overhead는 얼마인가요?

5. `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`를 합성해 gradient clipping을 구현하세요. clipping 적용 전후로 학습하세요. 효과를 보기 위해 학습 중 gradient norm을 그리세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|----------------------|
| XLA | "JAX를 빠르게 만드는 것" | Accelerated Linear Algebra. computation graph에서 연산을 fuse하고 최적화된 GPU/TPU 커널을 생성하는 컴파일러 |
| JIT | "Just-in-time compilation" | JAX가 첫 호출에서 함수를 trace하고 XLA로 컴파일한 다음 이후 호출에서 컴파일된 버전을 실행합니다 |
| Pure function | "Side effect 없음" | 출력이 입력에만 의존하는 함수. global state, mutation, 명시적 key 없는 randomness가 없습니다 |
| vmap | "Auto-batching" | 하나의 예제를 처리하는 함수를 다시 작성하지 않고 배치를 처리하는 함수로 변환합니다 |
| pmap | "Auto-parallelism" | 함수를 여러 device에 복제하고 입력 배치를 나눕니다 |
| Pytree | "배열의 중첩 dict" | JAX가 순회하고 변환할 수 있는 list, tuple, dict, array의 임의 중첩 구조 |
| Tracing | "계산 기록" | 실제 결과를 계산하지 않고 abstract value로 함수를 실행해 computation graph를 만듭니다 |
| Functional autodiff | "함수의 grad" | 텐서에 그래디언트 저장소를 붙이는 것이 아니라 함수를 변환해 도함수를 계산합니다 |
| Optax | "JAX의 optimizer library" | Adam, SGD, clipping, scheduling 같은 gradient transformation을 chain으로 합성하는 composable 라이브러리 |
| Flax | "JAX의 nn.Module" | state를 명시적으로 유지하면서 layer abstraction을 추가하는 Google의 JAX용 신경망 라이브러리 |

## 더 읽을거리

- JAX documentation: https://jax.readthedocs.io/ -- grad, jit, vmap에 대한 훌륭한 tutorial을 포함한 공식 문서
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 설계 철학을 설명하는 원 논문
- Flax documentation: https://flax.readthedocs.io/ -- Google의 JAX용 신경망 라이브러리
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax의 Pythonic 대안
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 표준 optimizer 라이브러리
- "You Don't Know JAX" (Colin Raffel, 2020) -- T5 저자 중 한 명이 쓴 JAX gotcha와 pattern 실전 가이드
