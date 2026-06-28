---
name: skill-jax-patterns
description: JAX의 functional programming pattern -- grad, jit, vmap, pmap을 언제 어떻게 사용할지
version: 1.0.0
phase: 3
lesson: 12
tags: [jax, functional-programming, autodiff, compilation, vectorization]
---

# JAX 함수형 패턴

JAX는 pure function을 변환합니다. 아래의 모든 pattern은 한 가지 규칙을 따릅니다. side effect 없이 입력을 받아 출력을 반환하는 함수를 작성합니다. 그런 다음 변환합니다.

## 네 가지 Transform

### grad -- 함수 미분하기

```python
grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```

사용할 때: 최적화를 위한 그래디언트가 필요할 때.
제약: 함수는 scalar를 반환해야 합니다. non-scalar 출력에는 `jax.jacobian`을 사용하세요.

### jit -- 함수 컴파일하기

```python
fast_fn = jax.jit(f)
```

사용할 때: 같은 shape의 입력으로 함수가 두 번 이상 호출될 때.
제약: traced value에 의존하는 Python control flow가 없어야 합니다. 조건문에는 `jax.lax.cond`, 루프에는 `jax.lax.scan`을 사용하세요.

### vmap -- 함수 vectorize하기

```python
batch_fn = jax.vmap(f, in_axes=(None, 0))
```

사용할 때: 하나의 예제를 위한 함수를 작성했고 배치에서도 동작해야 할 때.
`in_axes`는 어떤 인자 축을 따라 batch할지 지정합니다. `None`은 batch하지 않음(broadcast)을 의미합니다.

### pmap -- device 간 parallelize하기

```python
parallel_fn = jax.pmap(f, axis_name='devices')
```

사용할 때: 여러 GPU/TPU가 있고 data parallelism을 원할 때.
함수 안에서 `jax.lax.pmean(x, 'devices')`는 device 간 평균을 냅니다.

## 합성 규칙

Transform은 합성됩니다. 순서가 중요합니다:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))
```

오른쪽에서 왼쪽으로 읽습니다. loss_fn의 그래디언트를 구하고, 예제 위에서 vectorize한 다음, 결과를 컴파일합니다.

유효한 합성:
- `jit(grad(f))` -- 컴파일된 그래디언트 계산
- `jit(vmap(f))` -- 컴파일된 batched computation
- `vmap(grad(f))` -- 예제별 그래디언트
- `pmap(jit(f))` -- 병렬 컴파일 계산
- `grad(jit(f))` -- 컴파일된 함수의 그래디언트(jit(grad(f))와 같음)

## 파라미터 관리 패턴

JAX 파라미터는 pytree입니다(배열의 중첩 dict):

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}
```

모든 파라미터를 한 번에 업데이트:
```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

파라미터 수 세기:
```python
n_params = sum(p.size for p in jax.tree.leaves(params))
```

## PRNG Key 관리

JAX는 명시적 random key를 요구합니다:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```

여러 random operation에는 한 번 split하세요:
```python
keys = jax.random.split(key, n)
```

key를 재사용하지 마세요. 항상 사용 전에 split하세요.

## 흔한 실수

1. **jit 안에서 배열 변경**: JAX 배열은 immutable입니다. `x[i] = v` 대신 `x.at[i].set(v)`를 사용하세요.

2. **jit 안에서 Python print 사용**: `print`는 실행 중이 아니라 tracing 중에 실행됩니다. `jax.debug.print("{}", x)`를 사용하세요.

3. **traced value에 대한 jit 내부 Python if/for**: `jax.lax.cond`, `jax.lax.switch`, `jax.lax.scan`, `jax.lax.fori_loop`을 사용하세요.

4. **`.block_until_ready()`를 잊음**: JAX는 async dispatch를 사용합니다. benchmark에는 실제 완료를 기다리도록 `.block_until_ready()`를 호출하세요.

5. **PRNG key 재사용**: 같은 key를 쓰는 두 operation은 같은 "random" 값을 만듭니다. 항상 split하세요.

6. **jitted function의 global state**: global variable은 trace 시점에 capture됩니다. tracing 이후의 변경은 보이지 않습니다. 모든 것을 인자로 전달하세요.

## 의사결정 체크리스트

1. 함수가 두 번 이상 호출되나요? `@jax.jit`를 추가하세요.
2. 그래디언트가 필요한가요? `jax.grad` 또는 `jax.value_and_grad`로 감싸세요.
3. 하나의 예제를 처리하지만 배치가 있나요? `jax.vmap`으로 감싸세요.
4. 여러 device가 있나요? `jax.pmap`으로 감싸세요.
5. randomness를 사용하나요? PRNG key를 명시적으로 전달하세요.
6. 배열 값에 대한 Python control flow가 있나요? `jax.lax` primitive로 바꾸세요.

## JAX를 사용할 때

JAX를 사용할 때:
- 예제별 그래디언트가 필요함(differential privacy, Fisher information)
- TPU에서 학습함(JAX가 native framework)
- higher-order derivative가 필요함(Hessian, Jacobian)
- 전체 training step을 단일 kernel로 컴파일하고 싶음
- 팀이 Google DeepMind 또는 Anthropic에 있음

PyTorch를 사용할 때:
- 가장 큰 생태계를 원함(HuggingFace, torchvision, Lightning)
- raw speed보다 쉬운 디버깅을 우선함
- TorchServe/Triton으로 NVIDIA GPU에 배포함
- 채용 중임(PyTorch 개발자가 더 많음)
- 새 아키텍처를 빠르게 반복하고 싶음
