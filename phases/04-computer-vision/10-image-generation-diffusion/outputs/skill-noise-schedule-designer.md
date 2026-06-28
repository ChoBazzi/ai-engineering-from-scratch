---
name: skill-noise-schedule-designer
description: T와 target corruption level이 주어지면 linear, cosine, sigmoid beta schedule과 SNR plot을 만듭니다
version: 1.0.0
phase: 4
lesson: 10
tags: [computer-vision, diffusion, noise-schedule, training]
---

# Noise Schedule Designer

Beta schedule은 각 diffusion step에서 signal이 얼마나 유지되는지 제어합니다. 나쁜 schedule은 모든 downstream decision에서 training efficiency와 sample quality의 상한을 낮춥니다.

## 사용할 때

- 새 diffusion training run을 시작하며 T와 beta를 고를 때.
- Blurry sample을 만드는 diffusion model(schedule이 너무 aggressive함)이나 structure 학습에 실패하는 model(schedule이 너무 mild함)을 debugging할 때.
- 서로 다른 schedule을 보고하는 paper들의 design을 비교할 때.

## 입력

- `T`: timestep 수, 보통 100-1000.
- `type`: linear | cosine | sigmoid.
- `target_alpha_bar_final`: t=T에서 유지할 signal 비율, 기본값 0.001(99.9% corrupted).
- Optional `image_resolution` — 더 큰 image는 더 천천히 corrupt하는 schedule(cosine 또는 shifted schedule)에서 이득을 봅니다.

## Schedule formulas

### Linear
```text
beta_t = beta_start + (beta_end - beta_start) * (t - 1) / (T - 1)
```
기본값: beta_start=1e-4, beta_end=0.02(DDPM paper).

### Cosine (Nichol & Dhariwal, 2021)
```text
alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
```
s = 0.008. Signal을 더 오래 유지합니다. 낮은 step count에서 더 좋습니다.

### Sigmoid
```text
alpha_bar_t = 1 / (1 + exp(k * (t/T - 0.5)))
```
k = 6 to 12. 좋은 중간 지점입니다. 일부 SDXL variant가 사용합니다.

## 단계

1. Formula에 따라 beta를 계산합니다.
2. `alphas`, `alphas_cumprod`, `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`를 precompute합니다.
3. SNR_t = alpha_bar_t / (1 - alpha_bar_t)를 계산하고, SNR-over-time summary를 만듭니다.
4. `alphas_cumprod[T-1]`가 `target_alpha_bar_final`의 10% 이내인지 확인합니다. 아니면 beta_end(linear), s(cosine), k(sigmoid)를 조정하고 다시 시도합니다.
5. 세 checkpoint를 보고합니다.
   - `t=T*0.25` — early corruption
   - `t=T*0.5` — midway
   - `t=T*0.75` — near-final

## Report

```text
[schedule]
  type:   <name>
  T:      <int>
  beta_start: <float>   beta_end: <float>

[signal retention]
  t=0.25T:  alpha_bar=<X>  SNR=<X>
  t=0.5T:   alpha_bar=<X>  SNR=<X>
  t=0.75T:  alpha_bar=<X>  SNR=<X>
  t=T:      alpha_bar=<X>  SNR=<X>

[warnings]
  - <if alpha_bar collapses before 0.75T>
  - <if beta_end produces NaN in log-SNR>
```

## 규칙

- `alpha_bar_t <= 0`인 schedule은 절대 출력하지 마세요. 1e-5 아래의 값은 clamp하고 warning을 냅니다.
- Low-step-count sampling(< 30 steps)에는 cosine을 기본 recommendation으로 삼습니다.
- `quality_target == research`이면 linear를 기본값으로 삼습니다. DDPM baseline은 linear schedule로 보고됩니다.
- `image_resolution > 256`이면 high resolution에서 더 많은 signal을 유지하도록 schedule shifting(Chen, 2023)을 권장합니다.
