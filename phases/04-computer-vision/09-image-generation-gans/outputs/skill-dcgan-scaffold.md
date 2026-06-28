---
name: skill-dcgan-scaffold
description: z_dim, image_size, num_channels에서 training loop와 sample saver를 포함한 완전한 DCGAN scaffold를 작성합니다
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN Scaffold

세 parameter가 주어지면 target image resolution에 맞게 architecture size가 잡힌 runnable DCGAN project skeleton을 출력합니다.

## 사용할 때

- 작은 dataset에서 새 generative experiment를 시작할 때.
- 동작하는 minimal example로 DCGAN fundamentals를 가르칠 때.
- Conditional GAN을 prototype할 때(label injection은 같은 scaffold 안에서 일어납니다).

## 입력

- `image_size`: 32, 64, 128 중 하나(2의 거듭제곱이어야 함).
- `num_channels`: 1(grayscale) 또는 3(RGB).
- `z_dim`: 보통 64 또는 128.
- `with_spectral_norm`: yes | no; 기본값 yes.

## Architecture sizing

G의 transposed conv block 수와 D의 strided conv block 수는 `image_size`에 따라 달라집니다.

| image_size | G blocks | D blocks |
|------------|----------|----------|
| 32         | 4        | 4        |
| 64         | 5        | 5        |
| 128        | 6        | 6        |

각 추가 block은 spatial dimension을 두 배로 키우거나(G) 절반으로 줄입니다(D). Feature count는 32에서 시작해 `feat_base * 2^block_index`로 scale됩니다.

## Output files

- `model.py` — Generator + Discriminator class
- `train.py` — training loop, loss, optimiser setup
- `sample.py` — sample grid saver
- `config.json` — hyperparameter
- `README.md` — 10-line quickstart

## Report

```text
[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [list]
  D blocks:         <N>, channels: [list]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 1 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md
```

## 규칙

- 항상 G의 output에 `nn.Tanh()`를 사용하고 training 중 data를 [-1, 1]로 scale합니다.
- 항상 D에 `LeakyReLU(0.2)`를 사용합니다.
- `with_spectral_norm == yes`이면 D의 모든 conv를 `spectral_norm()`으로 감싸고 D에서 BatchNorm을 제거합니다. G의 BatchNorm은 유지합니다.
- image_size > 128인 scaffold는 절대 출력하지 않습니다. 그 이상에서는 DCGAN이 불안정해집니다. 사용자를 StyleGAN 또는 diffusion model로 안내하세요.
