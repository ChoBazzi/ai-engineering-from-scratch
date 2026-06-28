# 오토인코더와 변분 오토인코더(VAE)

> 일반 autoencoder는 압축한 뒤 복원합니다. 기억할 뿐입니다. 생성하지는 못합니다. code가 Gaussian처럼 보이도록 강제하는 한 가지 trick을 더하면 sampler가 생깁니다. `z = μ + σ·ε`의 reparameterization이라는 그 단 하나의 trick 때문에, 2026년에 사용하는 모든 latent-diffusion과 flow-matching 이미지 모델은 입력단에 VAE를 갖습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 minutes

## 문제

784-pixel MNIST digit을 16개 숫자의 code로 압축한 뒤 reconstruct한다고 해 봅시다. 일반 autoencoder는 reconstruction MSE를 잘 낮추겠지만 code space는 울퉁불퉁한 엉망이 됩니다. code space에서 임의의 점을 골라 decode하면 noise가 나옵니다. sampler가 없습니다. 압축 모델에 생성 모델 옷을 입힌 것뿐입니다.

실제로 원하는 것은 다음입니다. (a) code space가 sample할 수 있는 깨끗하고 매끄러운 distribution이어야 합니다. 예를 들어 isotropic Gaussian `N(0, I)`입니다. (b) 어떤 sample을 decode해도 그럴듯한 digit이 나와야 합니다. (c) encoder와 decoder가 여전히 잘 압축해야 합니다. 세 목표, 하나의 architecture, 하나의 loss입니다.

Kingma의 2013년 VAE는 encoder가 *distribution* `q(z|x) = N(μ(x), σ(x)²)`를 출력하도록 학습해서 이 문제를 풉니다. KL penalty로 그 distribution을 prior `N(0, I)` 쪽으로 당기고, decoding 전에 `q(z|x)`에서 `z`를 sample합니다. inference 시점에는 encoder를 버리고 `z ~ N(0, I)`를 sample한 뒤 decode합니다. KL penalty가 code space를 구조화하도록 강제합니다.

2026년에 VAE가 standalone으로 배포되는 경우는 드뭅니다. raw image quality에서는 diffusion에 밀렸습니다. 하지만 모든 latent-diffusion model(SD 1/2/XL/3, Flux, AudioCraft)의 기본 encoder입니다. VAE를 배우면 여러분이 쓰는 모든 image pipeline의 보이지 않는 첫 layer를 배우는 셈입니다.

## 개념

![Autoencoder vs VAE: reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Code space는 구조화되어 있지 않습니다.

**VAE encoder.** 두 vector `μ(x)`와 `log σ²(x)`를 출력합니다. 이 둘이 `q(z|x) = N(μ, diag(σ²))`를 정의합니다.

**Reparameterization trick.** `q(z|x)`에서 sampling하는 연산은 differentiable하지 않습니다. sample을 `z = μ + σ·ε`로 다시 쓰세요. 여기서 `ε ~ N(0, I)`입니다. 이제 `z`는 `(μ, σ)`와 parameter가 아닌 noise의 deterministic function이므로 gradient가 `μ`와 `σ`를 통해 흐릅니다.

**Loss.** Evidence Lower BOund(ELBO), 두 항으로 구성됩니다.

```text
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstruction은 `x̂`를 `x` 쪽으로 밀고, KL은 `q(z|x)`를 prior 쪽으로 밉니다. 둘은 trade-off 관계입니다. 작은 β(<1)는 더 선명한 sample과 덜 Gaussian인 code space를 만듭니다. 큰 β(>1)는 더 깨끗한 code space와 더 흐릿한 sample을 만듭니다. β-VAE(Higgins 2017)가 이 knob을 유명하게 만들었고 disentanglement 연구를 촉발했습니다.

**Sampling.** inference에서는 `z ~ N(0, I)`를 뽑아 decoder에 forward합니다. forward pass 한 번입니다. diffusion처럼 iterative sampling이 아닙니다.

```figure
vae-latent-grid
```

## 직접 만들기

`code/main.py`는 numpy나 torch 없이 tiny VAE를 구현합니다. 입력은 8-D의 2-component Gaussian mixture에서 뽑은 8-dimensional synthetic data입니다. Encoder와 decoder는 single hidden-layer MLP입니다. tanh activation, forward pass, loss, 손으로 쓴 backward pass를 구현합니다. production용이 아니라 pedagogy용입니다.

### 1단계: encoder 순전파

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`σ` 대신 `log σ²`를 써서 network output이 unconstrained가 되게 합니다. σ에 softplus를 쓰는 것은 함정입니다. σ ≈ 0에서 gradient가 죽습니다.

### 2단계: reparameterize하고 decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 3단계: ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

두 distribution이 모두 Gaussian이므로 exact closed-form KL이 있습니다. 수치 적분하지 마세요. 2026년에도 monte-carlo KL estimate를 넣은 코드를 배포하는 사람이 있습니다. 아무 이유 없이 3배 느립니다.

### 4단계: 생성

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

이것이 생성 모델입니다. 다섯 줄입니다.

## 함정

- **Posterior collapse.** KL term이 `q(z|x) → N(0, I)`를 너무 강하게 밀어 `z`가 `x`에 대한 정보를 담지 못합니다. 해결: β-annealing(β=0에서 시작해 1까지 ramp), free bits, inactive dimension의 KL skip.
- **흐릿한 sample.** Gaussian decoder likelihood는 MSE reconstruction을 의미하고, 이는 L2에서 Bayes-optimal인 mean입니다. plausible digit 집합의 mean은 흐릿한 digit입니다. 해결: discrete decoder(VQ-VAE, NVAE)를 쓰거나, VAE를 encoder로만 쓰고 latent 위에 diffusion을 쌓습니다. Stable Diffusion이 이렇게 합니다.
- **β가 너무 크고 너무 이른 경우.** posterior collapse를 보세요. β≈0.01에서 시작해 ramp하세요.
- **Latent dim이 너무 작은 경우.** MNIST는 16-D, ImageNet 256²는 256-D, ImageNet 1024²는 2048-D가 작동합니다. Stable Diffusion의 VAE는 512×512×3을 64×64×4로 압축합니다(spatial area에서 32x downsample factor, channel에서 32x).

## 사용하기

2026년 VAE stack:

| 상황 | 선택 |
|-----------|------|
| diffusion용 image-latent encoder | Stable Diffusion VAE(`sd-vae-ft-ema`) 또는 Flux VAE |
| Audio-latent encoder | Encodec(Meta), SoundStream, DAC(Descript) |
| Video latents | Sora의 spatiotemporal patches, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents(transformer modelling용) | VQ-VAE, RVQ(ResidualVQ) |
| generation용 continuous latents | Plain VAE, 그다음 latent space에서 flow/diffusion model condition |

Latent-diffusion model은 encoder와 decoder 사이에 diffusion model이 들어간 VAE입니다. VAE는 coarse compression을 맡고, diffusion model이 무거운 일을 합니다. video(VAE + video-diffusion DiT)와 audio(Encodec + MusicGen transformer)도 같은 pattern입니다.

## 출시하기

`outputs/skill-vae-trainer.md`를 저장하세요.

Skill은 dataset profile + latent-dim target + downstream use(reconstruction, sampling, latent-diffusion input)를 받아 architecture choice(plain/β/VQ/RVQ), β schedule, latent dim, decoder likelihood(Gaussian vs categorical), evaluation plan(recon MSE, dimension별 KL, `q(z|x)`와 `N(0, I)` 사이의 Fréchet distance)을 출력합니다.

## 연습문제

1. **쉬움.** `code/main.py`의 `β`를 `0.01`, `0.1`, `1.0`, `5.0`으로 바꾸세요. 최종 reconstruction MSE와 KL을 기록하세요. synthetic data에서 Pareto-best인 β는 무엇인가요?
2. **중간.** Gaussian decoder likelihood를 Bernoulli likelihood(cross-entropy loss)로 바꾸세요. 같은 synthetic data의 binarized version에서 sample quality를 비교하세요.
3. **어려움.** `code/main.py`를 mini VQ-VAE로 확장하세요. continuous `z`를 K=32 entry를 가진 codebook의 nearest-neighbour lookup으로 바꾸세요. reconstruction MSE를 비교하고 codebook entry가 몇 개 사용되는지 보고하세요(codebook collapse는 실제로 생깁니다).

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Autoencoder | Encode-decode network | `x → z → x̂`, MSE를 학습합니다. generative하지 않습니다. |
| VAE | sampler가 있는 AE | Encoder가 distribution을 출력하고 KL penalty가 code space를 shaping합니다. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z\|x) \|\| p(z)]`. `q = p(z\|x)`일 때 tight합니다. |
| Reparameterization | `z = μ + σ·ε` | stochastic node를 deterministic + pure noise로 다시 씁니다. sampling을 통한 backprop을 가능하게 합니다. |
| Prior | `p(z)` | latent의 target distribution. 보통 `N(0, I)`입니다. |
| Posterior collapse | "KL term이 이김" | Encoder가 `x`를 무시하고 prior를 출력합니다. decoder가 hallucinate해야 합니다. |
| β-VAE | 조절 가능한 KL weight | `loss = recon + β·KL`. β가 높을수록 더 disentangled되지만 더 흐릿합니다. |
| VQ-VAE | Discrete latent | continuous `z`를 가장 가까운 codebook vector로 바꿉니다. transformer modelling을 가능하게 합니다. |

## 프로덕션 노트: VAE는 diffusion server의 가장 뜨거운 path입니다

Stable Diffusion / Flux / SD3 pipeline에서 VAE는 request마다 두 번 호출됩니다. img2img / inpainting이면 encode할 때 한 번, decode할 때 한 번입니다. 1024²에서 decoder pass는 전체 pipeline에서 activation-memory peak가 가장 큰 단일 지점인 경우가 많습니다. `128×128×16` latent를 다시 `1024×1024×3`으로 upsample하기 때문입니다. 실용적 결과는 두 가지입니다.

- **decode를 slice 또는 tile하세요.** `diffusers`는 `pipe.vae.enable_slicing()`와 `pipe.vae.enable_tiling()`을 제공합니다. Tiling은 작은 seam artifact를 감수하고 memory를 `O(H·W)` 대신 `O(tile²)`로 바꿉니다. consumer GPU에서 1024²+를 다룰 때 필수입니다.
- **bf16 decoder, final resize에는 fp32 numerics.** SD 1.x VAE는 fp32로 release되었고 1024²+에서 fp16으로 cast하면 *조용히 NaN을 생성합니다*. SDXL은 `madebyollin/sdxl-vae-fp16-fix`를 제공합니다. 항상 fp16-fix variant를 선호하거나 bf16을 사용하세요.

## 더 읽을거리

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion. encoder로서의 VAE.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, audio VAE standard.
