# GAN — Generator 대 Discriminator

> 2014년 Goodfellow의 trick은 density를 완전히 건너뛰는 것이었습니다. 두 network. 하나는 fake를 만들고, 하나는 그것을 잡아냅니다. fake가 real과 구분되지 않을 때까지 싸웁니다. 작동하면 안 될 것처럼 보입니다. 실제로 자주 작동하지 않습니다. 하지만 작동할 때는 narrow domain에서 여전히 문헌상 가장 선명한 sample을 만듭니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 minutes

## 문제

VAE는 흐릿한 sample을 만듭니다. MSE decoder loss가 *mean* image에 대해 Bayes-optimal이기 때문입니다. 여러 plausible digit의 mean은 흐릿한 digit입니다. 우리는 어떤 target 하나에 대한 pixel-wise proximity가 아니라 *plausibility*를 보상하는 loss가 필요합니다. plausibility의 closed-form은 없습니다. 직접 학습해야 합니다.

Goodfellow의 아이디어는 real image와 fake를 구분하는 classifier `D(x)`를 학습하는 것입니다. 그리고 `D`를 속이도록 generator `G(z)`를 학습합니다. `G`의 loss signal은 현재 `D`가 어떤 것을 real처럼 보인다고 생각하는지입니다. 이 signal은 `G`가 좋아질수록 계속 바뀌며 moving target을 쫓습니다. 두 network가 모두 converge하면 `G`는 한 번도 `log p(x)`를 적지 않고 data distribution을 학습한 것입니다.

이것이 adversarial training입니다. 수학적으로는 minimax game입니다.

```text
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

2026년에 GAN은 더 이상 SOTA generator가 아닙니다(diffusion과 flow matching이 그 왕관을 가져갔습니다). 하지만 StyleGAN 2/3은 여전히 배포된 face model 중 가장 선명하고, GAN discriminator는 diffusion training에서 *perceptual loss*로 쓰이며, adversarial training은 real-time diffusion을 배포하게 해 주는 빠른 1-step distillation(SDXL-Turbo, SD3-Turbo, LCM)을 구동합니다.

## 개념

![GAN training: minimax 안의 generator와 discriminator](../assets/gan.svg)

**Generator `G(z)`.** noise vector `z ~ N(0, I)`를 sample `x̂`로 mapping합니다. decoder 모양의 network(dense 또는 transposed conv)입니다.

**Discriminator `D(x)`.** sample을 scalar probability(또는 score)로 mapping합니다. Real → 1, fake → 0입니다.

**Loss.** 두 alternating update입니다.

- **`D` 학습:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. real=1, fake=0인 binary cross-entropy입니다.
- **`G` 학습:** `loss_G = -log D(G(z))`. Goodfellow가 사용한 *non-saturating* form입니다. 원래의 `log(1 - D(G(z)))`는 `D`가 confident할 때 saturate되어 gradient를 죽입니다.

**Training loop.** `D` 한 step, `G` 한 step. 반복합니다.

**왜 작동하는가.** `G`가 `p_data`와 완전히 일치하면 `D`는 chance보다 잘할 수 없고 모든 곳에서 0.5를 출력합니다. `G`는 더 이상 gradient를 받지 않습니다. equilibrium입니다.

**왜 깨지는가.** mode collapse(`G`가 `D`가 분류하지 못하는 mode 하나를 찾아 그것만 계속 찍어냄), vanishing gradient(`D`가 너무 빨리 학습해 `log D`가 saturate), training instability(learning rate, batch size, 무엇이든)가 생깁니다.

## GAN을 작동하게 만든 변형들

| 연도 | 혁신 | 해결한 것 |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU. 첫 stable architecture. |
| 2017 | WGAN, WGAN-GP | BCE를 Wasserstein distance + gradient penalty로 교체. vanishing gradient를 완화. |
| 2017 | Spectral normalization | discriminator에 Lipschitz bound 적용. 2026년 discriminator에도 여전히 사용. |
| 2018 | Progressive GAN | 낮은 resolution부터 학습하고 layer를 추가. 첫 megapixel 결과. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. fixed-domain photorealism의 state of the art. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant. 2026년에도 face gold standard. |
| 2022 | StyleGAN-XL | Conditional, class-aware, larger scale. |
| 2024 | R3GAN | 더 강한 regularization으로 재정리. trick 없이 1024²에서 작동. |

```figure
gan-minimax
```

## 직접 만들기

`code/main.py`는 1-D data, 즉 두 Gaussian의 mixture에서 tiny GAN을 학습합니다. Generator와 discriminator는 single-hidden-layer MLP입니다. forward, backward, minimax loop를 손으로 구현합니다. 목표는 두 가지 핵심 failure mode(mode collapse + vanishing gradient)가 일어나는 순간을 보는 것입니다.

### 1단계: non-saturating loss

vanilla Goodfellow loss `log(1 - D(G(z)))`는 D가 G의 fake를 높은 confidence로 fake라고 분류하면 0으로 갑니다. 그 시점에서 G의 gradient는 사실상 0입니다. G는 개선될 수 없습니다. non-saturating form `-log D(G(z))`는 반대의 asymptote를 갖습니다. D가 confident할수록 값이 커져 G에 강한 signal을 줍니다.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 2단계: generator step 하나당 discriminator step 하나

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

G에는 fresh fake를 써야 합니다. 그렇지 않으면 gradient가 stale합니다.

### 3단계: mode collapse 관찰

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

정석적인 증상은 두 real mode 중 하나가 더 이상 생성되지 않는 것입니다. discriminator는 그것이 fake로 나타나는 것을 보지 못하므로 더 이상 교정하지 않습니다.

## 함정

- **Discriminator가 너무 강함.** D의 learning rate를 2-5배 낮추거나 instance/layer noise를 추가하세요. D가 >95% accuracy에 도달하면 G는 죽습니다.
- **Generator가 mode 하나를 암기함.** D input에 noise를 추가하거나 minibatch-discriminator layer를 쓰거나 WGAN-GP로 전환하세요.
- **Batch norm의 statistics leakage.** Real batch와 fake batch가 같은 BN layer를 통과하면 statistics가 섞입니다. 대신 instance norm 또는 spectral norm을 사용하세요.
- **Inception-score gaming.** FID와 IS는 sample 수가 적으면 noisy합니다. eval에는 ≥10k samples를 사용하세요.
- **Conditional task에서 one-shot sampling은 거짓말입니다.** 쓸 만한 output을 얻으려면 여전히 CFG scale, truncation trick, re-sampling이 필요합니다.

## 사용하기

2026년 GAN stack:

| 상황 | 선택 |
|-----------|------|
| Photoreal human faces, fixed pose | StyleGAN3(가장 선명하고 가장 작음) |
| Anime / stylized faces | StyleGAN-XL 또는 Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN(Phase 8 · 04) 또는 ControlNet(Phase 8 · 08) |
| Fast 1-step text-to-image | diffusion의 adversarial distillation(SDXL-Turbo, SD3-Turbo) |
| diffusion trainer 내부의 perceptual loss | image crop 위의 small GAN discriminator |
| multi-modal, open-ended인 모든 것 | 하지 마세요. diffusion 또는 flow matching을 쓰세요. |

GAN은 선명하지만 좁습니다. domain이 열리는 순간, 즉 photo, arbitrary text prompt, video로 가면 diffusion으로 전환하세요. adversarial trick은 standalone generator가 아니라 component(perceptual loss, distillation)로 살아남았습니다.

## 출시하기

`outputs/skill-gan-debugger.md`를 저장하세요. Skill은 실패한 GAN run(loss curve, sample grid, dataset size)을 받아 likely cause의 ranked list, one-line fix, rerun protocol을 출력합니다.

## 연습문제

1. **쉬움.** 기본 설정으로 `code/main.py`를 실행하세요. 그다음 `D_LR = 5 * G_LR`로 설정하고 다시 실행하세요. G의 loss가 얼마나 빨리 constant로 collapse하나요?
2. **중간.** Goodfellow BCE loss를 WGAN loss로 바꾸세요. `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`를 쓰고 D의 weight를 `[-0.01, 0.01]`로 clip하세요. training이 더 stable한가요? wall-clock convergence를 비교하세요.
3. **어려움.** 1-D 예제를 2-D data(ring 위의 8개 Gaussian mixture)로 확장하세요. step 1k, 5k, 10k에서 generator가 8개 mode 중 몇 개를 포착하는지 추적하세요. minibatch discrimination을 구현하고 다시 측정하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Generator | "G" | Noise-to-sample network, `G: z → x̂`. |
| Discriminator | "D" | Classifier `D: x → [0, 1]`, real vs fake. |
| Minimax | "The game" | joint objective의 `min_G max_D`. |
| Non-saturating loss | "The fix" | G에 `log(1 - D(G(z)))` 대신 `-log D(G(z))`를 사용합니다. |
| Mode collapse | "G가 하나만 외웠다" | 다양한 data에도 불구하고 generator가 적은 수의 distinct output만 생성합니다. |
| WGAN | "Wasserstein" | BCE를 Earth-Mover distance + gradient penalty로 대체합니다. gradient가 더 smooth합니다. |
| Spectral norm | "Lipschitz trick" | D의 weight norm을 제한해 slope를 bound합니다. training을 안정화합니다. |
| StyleGAN | "작동하는 것" | Mapping network + AdaIN. face 분야 best-in-class이며 2026년에도 그렇습니다. |

## 프로덕션 노트: one-shot inference는 GAN의 지속적인 장점입니다

GAN은 open-domain generation의 sample quality에서는 더 이상 이기지 못하지만 inference cost에서는 여전히 이깁니다. production-inference 문헌의 vocabulary로 보면 GAN에는 다음이 있습니다.

- **prefill도 decode stage도 없습니다.** 단일 `G(z)` forward pass입니다. TTFT ≈ total latency입니다.
- **KV-cache 압박이 없습니다.** 유일한 state는 weight입니다. batch size는 cache가 아니라 activation memory에 의해 제한됩니다.
- **Continuous batching이 trivial합니다.** 모든 request가 같은 고정 FLOPs를 쓰므로 server의 target occupancy에 맞춘 static batch가 보통 optimal입니다. in-flight scheduler가 필요 없습니다.

이것이 GAN distillation(SDXL-Turbo, SD3-Turbo, ADD, LCM)이 2026년 fast text-to-image의 지배적 기술인 이유입니다. diffusion base의 distribution을 유지하면서 20-50-step diffusion pipeline을 1-4개의 GAN-style forward pass로 collapse합니다. adversarial loss는 느린 generator를 빠른 generator로 바꾸는 training-time knob으로 살아남았습니다.

## 더 읽을거리

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 원래 GAN 논문.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 첫 stable architecture.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
