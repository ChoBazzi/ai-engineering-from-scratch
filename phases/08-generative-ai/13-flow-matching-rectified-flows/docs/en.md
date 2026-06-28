# Flow Matching과 Rectified Flow

> 확산 모델은 노이즈에서 데이터로 이어지는 휘어진 경로를 따라가기 때문에 샘플링에 20-50단계가 필요합니다. Flow matching(Lipman et al., 2023)과 rectified flow(Liu et al., 2022)는 곧은 경로를 학습했습니다. 경로가 더 곧으면 필요한 단계가 줄고, 추론이 더 빨라집니다. Stable Diffusion 3, Flux.1, AudioCraft 2는 모두 2024년에 flow matching으로 전환했습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 minutes

## 문제

DDPM의 역과정은 `N(0, I)`에서 데이터 분포로 돌아가는 1000단계 확률적 이동입니다. DDIM은 이를 20-50개의 결정론적 단계로 줄였습니다. 우리가 원하는 것은 더 적은 단계, 이상적으로는 한 단계입니다. 걸림돌은 역과정을 푸는 ODE가 stiff하고 경로가 휘어져 있다는 점입니다.

노이즈에서 데이터로 가는 경로가 *직선*이 되도록 모델을 학습할 수 있다면, `t=1`에서 `t=0`으로 가는 단일 Euler 단계만으로도 작동합니다. Flow matching은 이를 직접 구성합니다. `x_1 ∼ N(0, I)`에서 `x_0 ∼ data`로 가는 직선 보간을 정의하고, 벡터장 `v_θ(x, t)`가 그 시간 미분과 일치하도록 학습한 뒤, 추론 때 적분합니다.

Rectified flow(Liu 2022)는 한 걸음 더 나아갑니다. reflow 절차로 경로를 반복적으로 곧게 펴서 점점 더 선형에 가까운 ODE를 만듭니다. reflow를 두 번 반복하면 2단계 샘플러가 50단계 DDPM 품질과 맞먹습니다.

## 개념

![Flow matching: noise와 data 사이의 직선 보간](../assets/flow-matching.svg)

### 직선 flow

다음처럼 정의합니다.

```text
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

여기서 `x_0 ~ data`이고 `x_1 ~ N(0, I)`입니다. 이 직선을 따르는 시간 미분은 상수입니다.

```text
dx_t / dt = x_1 - x_0
```

신경망 벡터장 `v_θ(x_t, t)`를 정의하고 이 미분과 일치하도록 학습합니다.

```text
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

이것이 **conditional flow matching** 손실(Lipman 2023)입니다. 학습은 시뮬레이션 없이 진행됩니다. ODE를 펼쳐 계산하지 않고 `(x_0, x_1, t)`를 샘플링해 회귀만 합니다.

### 샘플링

추론 때는 학습된 벡터장을 시간상 *역방향*으로 적분합니다.

```text
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

`x_1 ~ N(0, I)`에서 시작해 `t=0`까지 Euler 단계로 내려갑니다.

### Rectified flow(Liu 2022)

직선 flow는 작동하지만 학습된 경로가 *실제로 완전히 직선은 아닙니다*. 여러 `x_0`가 같은 `x_1`에 대응될 수 있어서 경로가 휘어집니다. Rectified flow의 reflow 단계는 다음과 같습니다.

1. 무작위 짝짓기로 flow 모델 v_1을 학습합니다.
2. v_1을 `x_1`에서 도착점 `x_0`까지 적분해 N개의 `(x_1, x_0)` 쌍을 샘플링합니다.
3. 그 짝지어진 예제로 v_2를 학습합니다. 이제 쌍들이 "ODE에 맞춰진" 상태이므로 두 점 사이의 직선 보간이 실제로 더 평평합니다.
4. 반복합니다.

실무에서는 reflow를 2번 반복하면 거의 선형에 도달해 2-4단계 추론이 가능해집니다. SDXL-Turbo, SD3-Turbo, LCM은 모두 flow matching 모델에서 증류된 모델입니다.

### 2024년에 이미지에서 이 방식이 승리한 이유

세 가지 이유가 있습니다.

1. **시뮬레이션 없는 학습** - 학습 중 ODE unroll이 없어 구현이 단순합니다.
2. **더 나은 손실 기하** - 직선 경로는 일관된 signal-to-noise를 갖지만, DDPM ε-loss는 스케줄 양끝에서 SNR이 나쁩니다.
3. **더 빠른 추론** - SDXL-Turbo 품질에서 4-8단계, consistency distillation을 쓰면 1단계가 가능합니다.

## Flow matching vs DDPM - 정확한 연결

Gaussian-conditional 경로를 쓰는 flow matching은 *특정 노이즈 스케줄을 가진* 확산입니다. `x_t = α(t) x_0 + σ(t) x_1` 스케줄을 고르면 flow matching은 `v = α'·x_0 - σ'·x_1`인 Stratonovich 재정식화 확산을 복원합니다. Gaussian 경로에서는 둘이 대수적으로 동등합니다.

Flow matching이 추가한 것은 타깃의 *명확성*(평범한 velocity), 더 깔끔한 손실, 그리고 비-Gaussian 보간자를 실험할 수 있는 자유입니다.

## 직접 만들기

`code/main.py`는 두 모드 Gaussian mixture에서 1차원 flow matching을 구현합니다. 벡터장 `v_θ(x, t)`는 직선 타깃으로 학습되는 작은 MLP입니다. 추론 때는 1, 2, 4, 20개의 Euler 단계를 적분하고 샘플 품질을 비교합니다.

### 1단계: 학습 손실

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 2단계: 다단계 추론

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 3단계: 단계 수 비교

4단계 샘플러가 이미 20단계 품질과 비슷해지는 것을 기대할 수 있습니다. 이는 지연 시간 측면에서 큰 차이입니다.

## 함정

- **시간 매개변수화.** Flow matching은 데이터 쪽을 `t=0`, 노이즈 쪽을 `t=1`로 두고 `t ∈ [0, 1]`을 사용합니다. DDPM은 데이터 쪽을 `t=0`, 노이즈 쪽을 `t=T`로 두고 `t ∈ [0, T]`를 사용합니다. 방향은 같지만 스케일이 다릅니다. 논문에서도 이 점을 자주 잘못 씁니다.
- **스케줄 선택.** Rectified flow의 직선은 "그" flow-matching 스케줄이지만, 더 나은 스케일 커버리지를 위해 cosine 또는 logit-normal t-sampling(SD3가 사용)을 쓸 수 있습니다.
- **Reflow 비용.** reflow용 짝지어진 데이터셋을 생성하려면 샘플마다 전체 추론 패스가 필요합니다. 1-2단계 추론이 정말 필요할 때만 reflow를 사용하세요.
- **Classifier-free guidance는 여전히 적용됩니다.** 선형 결합에서 ε를 v로 바꾸기만 하면 됩니다. `v_cfg = (1+w) v_cond - w v_uncond`.

## 활용하기

| 사용 사례 | 2026 스택 |
|----------|-----------|
| 최고 품질의 텍스트-이미지 | Flow matching: SD3, Flux.1-dev |
| 1-4단계 텍스트-이미지 | 증류된 flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| 실시간 추론 | Flow-matched base에서 consistency distillation(LCM, PCM) |
| 오디오 생성 | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| 비디오 생성 | Flow matching과 diffusion 혼합(Sora, Veo, Stable Video) |
| 과학/물리(입자 궤적, 분자) | Flow matching + equivariant vector field |

2025-2026년에 논문이 "diffusion보다 빠르다"고 말한다면 거의 항상 flow matching + distillation입니다.

## 출시하기

`outputs/skill-fm-tuner.md`를 저장하세요. 이 skill은 diffusion 스타일 모델 명세를 flow-matching 학습 설정으로 변환합니다. 스케줄 선택, 시간 샘플링 분포(uniform / logit-normal), optimizer, reflow 계획, 목표 단계 수, 평가 프로토콜을 포함합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하고 1단계와 20단계의 MSE를 실제 데이터 분포와 비교하세요.
2. **보통.** uniform `t` sampling을 logit-normal(중간 t에 샘플링을 집중)로 바꾸세요. 모델 품질이 개선되나요?
3. **어려움.** reflow를 한 번 구현하세요. 첫 번째 모델을 적분해 짝지어진 (x_0, x_1)을 만들고, 그 쌍으로 두 번째 모델을 학습한 뒤, 1단계 샘플 품질을 비교하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------|
| Flow matching | "직선 diffusion" | 보간자를 따라 `v_θ(x, t)`가 `x_1 - x_0`와 일치하도록 학습합니다. |
| Rectified flow | "Reflow" | 학습된 flow를 곧게 펴는 반복 절차입니다. |
| Velocity field | "v_θ" | 모델의 출력, 즉 `x_t`를 이동시킬 방향입니다. |
| Straight-line interpolant | "경로" | `x_t = (1-t)·x_0 + t·x_1`; 타깃 미분이 단순합니다. |
| Euler sampler | "1차 ODE solver" | 가장 단순한 적분기입니다. 경로가 곧을 때 잘 작동합니다. |
| Logit-normal t | "SD3 sampling" | gradient가 가장 강한 중간값 쪽으로 `t` 샘플링을 집중합니다. |
| Consistency distillation | "1단계 sampler" | 어떤 `x_t`든 직접 `x_0`로 매핑하도록 student를 학습합니다. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; 같은 기법, 새 변수입니다. |

## 프로덕션 노트: Flux.1-schnell은 가장 빠른 flow matching입니다

Flow matching의 프로덕션상 승리는 Flux.1-schnell입니다. 이는 Flux-dev급 품질을 유지하면서 1-4 추론 단계로 증류된 flow-matched DiT입니다. Niels의 "Run Flux on an 8GB machine" notebook은 기준이 되는 배포 레시피입니다. T5 + CLIP 인코딩, 양자화된 MMDiT denoise(schnell은 4단계, dev는 50단계), VAE decode로 구성됩니다. 비용 계산은 다음과 같습니다.

| 변형 | 단계 수 | L4에서 1024² latency | 총 FLOPs(상대값) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

프로덕션 규칙은 이렇습니다. **flow-matched base + distillation = 빠른 텍스트-이미지의 2026년 기본값.** 모든 주요 벤더가 이 조합을 제공합니다. SD3-Turbo(SD3 + flow + distillation), Flux-schnell(Flux-dev + rectified-flow straightening), CogView-4-Flash가 예입니다. 순수 diffusion base는 레거시 체크포인트에만 남아 있습니다.

## 더 읽을거리

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) - rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) - flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) - SD3, 대규모 rectified flow.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) - FM + diffusion을 포괄하는 일반 프레임워크.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) - diffusion / flow의 1단계 distillation.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) - turbo variant.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) - 프로덕션 flow matching.
