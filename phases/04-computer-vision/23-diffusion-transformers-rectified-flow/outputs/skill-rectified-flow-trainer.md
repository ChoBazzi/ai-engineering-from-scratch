---
name: skill-rectified-flow-trainer
description: AdaLN DiT와 Euler sampling으로 완전한 rectified-flow 학습 루프 작성
version: 1.0.0
phase: 4
lesson: 23
tags: [diffusion, rectified-flow, DiT, training]
---

# Rectified Flow 학습기

어떤 image tensor dataset에서든 작은 DiT를 rectified flow로 성공적으로 학습할 수 있는 깔끔하고 최소한의 학습 루프를 만드세요.

## 사용할 때

- SD3 / FLUX 학습 objective를 작은 규모로 재현할 때.
- 같은 데이터에서 rectified flow와 DDPM을 벤치마크할 때.
- 비표준 도메인(의료, 위성)을 위한 커스텀 rectified-flow 모델을 만들 때.

## 입력

- `model`: `(x, t)`를 받아 predicted velocity를 반환하는 `nn.Module`.
- `dataset`: 모델 도메인의 clean image iterable.
- `optimizer`: `lr=1e-4`, `weight_decay=0.01`, `betas=(0.9, 0.99)`를 쓰는 AdamW.
- `scheduler`: warmup이 있는 cosine, 기본값은 1000 warmup steps.

## 학습 단계

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # uniform in [0, 1]
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # velocity target
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## 샘플링(Euler)

```python
@torch.no_grad()
def sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

## 팁

- `torch.rand`의 uniform `t`를 사용하세요. logit-normal이나 SD3 스타일의 weighted sampling of `t`는 약간 도움이 되지만 시작할 때 필수는 아닙니다.
- 모델 가중치의 EMA는 표준 관행입니다. decay 0.9999로 `ema_model`을 유지하세요.
- 조건부 모델의 classifier-free guidance: 학습 중 10% 확률로 conditioning을 empty/null embedding으로 바꾸고, 추론 시 `v_uncond + w * (v_cond - v_uncond)`를 `w` 약 3-5로 섞습니다.
- LDM 스타일 학습(FLUX, SD3)에서는 전체 루프가 VAE latent space에서 실행됩니다. 위의 clean `x0`는 실제로 `VAE.encode(image)`입니다.
- 32x32 toy dataset의 일반적인 수렴: 2000-5000 steps. 실제 latent SD3 학습: 수십만 steps.

## 보고서

```text
[rectified flow training]
  steps:        <int>
  final loss:   <float>
  ema decay:    <float>
  vae?:         yes | no
  cfg dropout:  <fraction>

[sampling]
  default steps: 20
  schnell / turbo target: 4
  full quality reference: 50+ (for comparison only)
```

## 규칙

- RGB `uint8` 데이터에서 image-space velocity target으로 rectified flow를 학습하지 마세요. 먼저 zero mean, unit variance로 정규화하세요.
- timestep-bucket별 학습 loss를 항상 기록하세요. 이른 timestep(0 근처)의 loss가 늦은 timestep(1 근처)보다 높다면 velocity parameterisation이 잘못 연결되었을 가능성이 큽니다.
- 같은 학습 루프 안에서 rectified-flow velocity target과 DDPM noise target을 섞지 마세요. 하나를 선택하세요.
- Ampere+ GPU에서는 bfloat16 학습을 사용하세요. float16은 velocity magnitude 때문에 rectified flow에서 가끔 NaN gradient를 만듭니다.
