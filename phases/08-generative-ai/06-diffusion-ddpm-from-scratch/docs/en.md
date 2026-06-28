# Diffusion 모델 — DDPM을 처음부터 만들기

> Ho, Jain, Abbeel (2020)은 분야가 놓지 못할 레시피를 제시했습니다. 데이터를 천 개의 작은 단계에 걸쳐 noise로 파괴합니다. 하나의 neural net을 학습해 noise를 예측하게 합니다. 추론에서는 그 과정을 거꾸로 돌립니다. 오늘날 주류 이미지, 비디오, 3D, 음악 모델은 모두 이 루프 위에서 동작하며, 그 위에 flow matching이나 consistency trick을 얹을 수 있습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 문제

우리는 `p_data(x)`의 sampler를 원합니다. GAN은 자주 발산하는 minimax game을 합니다. VAE는 Gaussian decoder에서 흐릿한 샘플을 만듭니다. 정말 원하는 것은 (a) 하나의 안정적인 loss(saddle point도 minimax도 없음), (b) `log p(x)`의 lower bound(따라서 likelihood가 있음), (c) SOTA 품질에 맞는 샘플을 동시에 주는 학습 목표입니다.

Sohl-Dickstein et al. (2015)은 이론적 답을 냈습니다. Gaussian noise를 점진적으로 추가하는 Markov chain `q(x_t | x_{t-1})`를 정의하고, reverse chain `p_θ(x_{t-1} | x_t)`를 학습해 denoise합니다. Ho, Jain, Abbeel (2020)은 loss가 한 줄로 단순화될 수 있음을 보였습니다. 즉 noise를 예측하면 됩니다. 그리고 수학을 깔끔하게 정리했습니다. 2020년에는 호기심의 대상이었습니다. 2021년에는 state-of-the-art 샘플을 만들었습니다. 2022년에는 Stable Diffusion이 되었습니다. 2026년에는 기반 substrate입니다.

## 개념

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Forward process `q`.** `T`개의 작은 단계로 Gaussian noise를 추가합니다. 수학이 다루기 쉬운 이유인 closed form은 누적 단계 역시 Gaussian이라는 점입니다.

```text
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

여기서 `α̅_t = ∏_{s=1..t} (1 - β_s)`이며, `β_t` schedule에 의해 정해집니다. `β_t`를 T=1000 단계 동안 1e-4에서 0.02까지 선형으로 고르면 `x_T`는 대략 `N(0, I)`가 됩니다.

**Reverse process `p_θ`.** 추가된 noise를 예측하는 neural net `ε_θ(x_t, t)`를 학습합니다. `x_t`가 주어지면 다음 식으로 denoise합니다.

```text
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

여기서 `σ_t`는 `sqrt(β_t)`이거나 학습된 variance입니다. 식은 보기 흉하지만 단순한 대수입니다. posterior `q(x_{t-1} | x_t, x_0)`가 주어졌을 때 `x_{t-1}`를 풀고, `x_0`를 noise 예측으로 얻은 추정치로 치환한 것입니다.

**Training loss.**

```text
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

데이터에서 `x_0`를 샘플링하고, 무작위 `t`를 고른 뒤, `ε ~ N(0, I)`를 샘플링합니다. closed form으로 noisy `x_t`를 한 번에 계산하고 noise에 대해 회귀합니다. 하나의 loss, minimax 없음, KL 없음, reparameterization trick 없음.

**Sampling.** `x_T ~ N(0, I)`에서 시작합니다. `t = T`부터 `1`까지 reverse step을 반복합니다. 끝입니다.

## 왜 작동하는가

세 가지 직관이 있습니다.

1. **Denoising은 쉽고 generating은 어렵습니다.** `t=T`에서는 데이터가 순수 noise이므로 net은 사소한 문제를 풉니다. `t=0`에서는 net이 몇 픽셀만 정리하면 됩니다. 중간 `t`에서는 문제가 어렵지만, 모든 noise level에서 같은 weight로 많은 gradient가 흘러갑니다.

2. **Score matching의 다른 모습입니다.** Vincent (2011)는 noise를 예측하는 것이 `∇_x log q(x_t | x_0)`, 즉 *score*를 추정하는 것과 같음을 증명했습니다. Reverse SDE는 이 score를 사용해 밀도 gradient를 올라가며 high-probability 영역으로 향하는 guided random walk를 수행합니다.

3. **ELBO가 단순 MSE로 줄어듭니다.** 전체 variational lower bound에는 timestep별 KL term이 있습니다. DDPM의 parameterization에서는 그 KL term들이 특정 계수가 붙은 noise prediction MSE로 단순화됩니다. Ho는 그 계수를 제거해 "simple" loss라고 불렀고, 품질은 오히려 *개선*되었습니다.

```figure
diffusion-denoise
```

## 직접 만들기

`code/main.py`는 1-D DDPM을 구현합니다. 데이터는 두 mode를 가진 mixture입니다. "net"은 `(x_t, t)`를 받아 예측 noise를 출력하는 작은 MLP입니다. 학습은 한 줄 loss입니다. 샘플링은 reverse chain을 반복합니다.

### 1단계: forward schedule(닫힌 형식)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 2단계: `x_t`를 한 번에 sample

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 3단계: training step 하나

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 4단계: reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

40개 timestep과 24-unit MLP를 쓰는 1-D 문제에서는 약 200 epoch 안에 두 mode mixture를 학습합니다.

## 시간 conditioning

Net은 자신이 어느 timestep을 denoise하는지 알아야 합니다. 표준 선택지는 두 가지입니다.

- **Sinusoidal embedding.** Transformer positional encoding과 같습니다. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`입니다. MLP를 통과시킨 뒤 net 안으로 broadcast합니다.
- **FiLM / group-norm conditioning.** 각 block에서 embedding을 채널별 scale/bias(FiLM)로 projection합니다.

우리 장난감 코드는 sinusoidal → concat을 사용합니다. Production U-Net은 FiLM을 사용합니다.

## 함정

- **Schedule이 매우 중요합니다.** Linear `β`는 DDPM 기본값이지만 cosine schedule(Nichol & Dhariwal, 2021)은 같은 compute에서 더 나은 FID를 줍니다. 품질이 정체되면 schedule을 바꾸세요.
- **Timestep embedding은 취약합니다.** 원시 `t`를 float로 넘기는 방식은 장난감 1-D에서는 작동하지만 이미지에서는 실패합니다. 항상 제대로 된 embedding을 쓰세요.
- **V-prediction vs ε-prediction.** 좁은 구간(매우 작은 t 또는 매우 큰 t)에서는 `ε`의 signal-to-noise가 좋지 않습니다. V-prediction(`v = α·ε - σ·x`)이 더 안정적입니다. SDXL, SD3, Flux가 사용합니다.
- **Classifier-free guidance.** 추론 시 conditional `ε`와 unconditional `ε`를 모두 계산한 뒤 `ε_cfg = (1 + w) · ε_cond - w · ε_uncond`를 사용합니다. 일반적으로 `w ≈ 3-7`입니다. Lesson 08에서 다룹니다.
- **1000 step은 많습니다.** Production은 DDIM(20-50 step), DPM-Solver(10-20 step), 또는 distillation(1-4 step)을 씁니다. Lesson 12를 보세요.

## 활용하기

| 역할 | 2026년의 일반적인 stack |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT (Lesson 07) |
| Video latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

Diffusion은 범용 generative backbone입니다. Flow matching(Lesson 13)은 2024-2026년의 경쟁자로, 보통 같은 품질에서 추론 속도가 더 빠릅니다.

## 출시하기

`outputs/skill-diffusion-trainer.md`를 저장하세요. 이 skill은 dataset + compute budget을 받아 schedule(linear/cosine/sigmoid), prediction target(ε/v/x), step 수, guidance scale, sampler family, eval protocol을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 T를 40에서 10으로 바꾸세요. 샘플 품질(출력의 시각적 histogram)은 어떻게 나빠지나요? 두 mode 구조는 어느 T에서 붕괴하나요?
2. **중간.** ε-prediction에서 v-prediction으로 바꾸세요. Reverse step을 다시 유도하세요. 최종 샘플 품질을 비교하세요.
3. **어려움.** Classifier-free guidance를 추가하세요. Class label `c ∈ {0, 1}`에 condition하고, 학습 중 10% 확률로 label을 drop한 뒤, sampling 때 `ε = (1+w)·ε_cond - w·ε_uncond`를 사용하세요. `w = 0, 1, 3, 7`에서 conditional-mode-hit rate를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-------------------|-----------|
| Forward process | "Adding noise" | 데이터를 파괴하는 고정 Markov chain `q(x_t \| x_{t-1})`입니다. |
| Reverse process | "Denoising" | 데이터를 재구성하는 학습된 chain `p_θ(x_{t-1} \| x_t)`입니다. |
| β schedule | "The noise ladder" | 단계별 variance입니다. linear, cosine, sigmoid가 있습니다. |
| α̅ | "Alpha bar" | 누적 곱 `∏(1 - β)`입니다. `x_0`에서 closed-form `x_t`를 줍니다. |
| Simple loss | "MSE on noise" | `\|\|ε - ε_θ(x_t, t)\|\|²`입니다. 모든 variational derivation이 이것으로 접힙니다. |
| ε-prediction | "Predict noise" | 출력이 추가된 noise입니다. 표준 DDPM입니다. |
| V-prediction | "Predict velocity" | 출력이 `α·ε - σ·x`입니다. t 전반에서 conditioning이 더 좋습니다. |
| DDPM | "The paper" | Ho et al. 2020. linear β, 1000 step, U-Net. |
| DDIM | "Deterministic sampler" | Non-Markov sampler, 20-50 step, 같은 학습 목표를 사용합니다. |
| Classifier-free guidance | "CFG" | Conditional noise prediction과 unconditional noise prediction을 섞어 conditioning을 증폭합니다. |

## 프로덕션 노트: diffusion inference는 step-count 문제입니다

DDPM 논문은 T=1000 reverse step을 사용합니다. Production에서 이렇게 배포하는 사람은 없습니다. 실제 inference stack은 다음 세 전략 중 하나를 고릅니다. 각각은 "latency가 어디에서 오는가"라는 production framing에 깔끔하게 대응합니다.

1. **더 빠른 sampler, 같은 model.** DDIM(20-50 step), DPM-Solver++(10-20), UniPC(8-16). Reverse loop의 drop-in replacement이며, 학습된 `ε_θ` weight는 그대로 둡니다. 지연 시간을 20-50× 줄입니다.
2. **Distillation.** Student를 학습해 더 적은 step으로 teacher를 맞춥니다. Progressive Distillation(2 → 1), Consistency Models(arbitrary → 1-4), LCM, SDXL-Turbo, SD3-Turbo가 있습니다. 지연 시간을 추가로 5-10× 줄이지만 재학습이 필요합니다.
3. **Caching and compilation.** `torch.compile(unet, mode="reduce-overhead")`, TensorRT-LLM의 diffusion backend, `xformers`/SDPA attention, bf16 weight입니다. step당 지연 시간을 약 2× 줄이며 (1), (2)와 함께 쌓을 수 있습니다.

Production diffusion server에서 budget 대화는 LLM에 대한 production literature와 같습니다. Latency는 `num_steps × step_cost + VAE_decode`, throughput은 `batch_size × (num_steps × step_cost)^-1`입니다. TTFT는 작고(한 step), TPOT에 해당하는 값은 전체 응답 시간입니다. 사용자 관점에서 이미지 생성은 "한 번에 전체가 나오는" 작업이기 때문입니다.

## 더 읽을거리

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — 시대를 앞선 diffusion 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, 더 적은 step.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine schedule, learned variance.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — unified notation, 가장 깔끔한 recipe.
