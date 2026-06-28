# 조건부 GAN과 Pix2Pix

> 2014-2017년의 첫 큰 돌파구는 GAN이 무엇을 만들지 제어하는 것이었습니다. label, image, sentence를 붙입니다. Pix2Pix는 image version을 구현했고, narrow image-to-image task에서는 여전히 모든 범용 text-to-image model보다 낫습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 minutes

## 문제

Unconditional GAN은 임의의 얼굴을 sample합니다. demo에는 유용하지만 production에는 쓸모가 없습니다. 원하는 것은 *sketch를 photo로 mapping*, *map을 aerial photo로 mapping*, *daytime scene을 nighttime으로 mapping*, *grayscale image를 colorize*하는 것입니다. 이 모든 경우 input image `x`가 주어지고 어떤 semantic correspondence를 가진 `y`를 출력해야 합니다. 하나의 `x`마다 plausible한 `y`는 여러 개입니다. mean-squared error는 그것들을 뭉개진 결과로 평평하게 만듭니다. adversarial loss는 그렇지 않습니다. "looks real"은 선명하기 때문입니다.

Conditional GAN(Mirza & Osindero, 2014)은 condition `c`를 `G`와 `D` 양쪽의 input으로 추가합니다. Pix2Pix(Isola et al., 2017)는 이를 특화했습니다. condition은 전체 input image이고, generator는 U-Net이며, discriminator는 *patch-based* classifier(PatchGAN)이고, loss는 adversarial + L1입니다. 이 recipe는 2026년에도 narrow image-to-image domain에서 from-scratch text-to-image model을 능가합니다. *paired data*로 학습하기 때문입니다. 필요한 signal을 정확히 가지고 있습니다.

## 개념

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G.** `G(x, z) → y`. Pix2Pix에서 `z`는 G 내부의 dropout입니다(input noise가 없습니다. Isola는 explicit noise가 무시된다는 것을 발견했습니다).

**Conditional D.** `D(x, y) → [0, 1]`. input은 *pair*(condition, output)입니다. 이것이 핵심 차이입니다. D는 `y`가 real처럼 보이는지만이 아니라 `x`와 일관적인지도 판단해야 합니다.

**U-Net generator.** bottleneck을 가로지르는 skip connection이 있는 encoder-decoder입니다. input과 output이 low-level structure(edge, silhouette)를 공유하는 task에서 중요합니다. skip이 없으면 high-frequency detail이 사라집니다.

**PatchGAN discriminator.** 단일 real/fake score를 출력하는 대신, D는 각 cell이 약 70×70 pixel의 receptive field를 판단하는 `N×N` grid를 출력합니다. 그 값을 평균냅니다. 이것은 realism이 local하다는 Markov random field assumption입니다. 학습이 훨씬 빠르고, parameter가 적고, output이 더 선명합니다.

**Loss.**

```text
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 term은 training을 안정화하고 G를 알려진 target 쪽으로 밉니다. L1은 L2보다 edge가 더 선명합니다(mean이 아니라 median). `λ = 100`은 Pix2Pix의 기본값이었습니다.

## CycleGAN — pair가 없을 때

Pix2Pix에는 paired `(x, y)` data가 필요합니다. CycleGAN(Zhu et al., 2017)은 추가 loss, 즉 *cycle consistency* loss를 대가로 이 요구사항을 없앱니다. 두 generator `G: X → Y`와 `F: Y → X`가 있습니다. `F(G(x)) ≈ x`와 `G(F(y)) ≈ y`가 되도록 학습합니다. 이를 통해 paired example 없이도 horse를 zebra로, summer를 winter로 translate할 수 있습니다.

2026년에 unpaired image-to-image는 주로 CycleGAN보다 diffusion(ControlNet, IP-Adapter)으로 처리합니다. 하지만 cycle-consistency 아이디어는 거의 모든 unpaired domain adaptation 논문에 살아 있습니다.

## 직접 만들기

`code/main.py`는 1-D data 위의 tiny conditional GAN을 구현합니다. condition `c`는 class label(0 또는 1)입니다. task는 주어진 class에 대한 conditional distribution에서 sample을 생성하는 것입니다.

### 1단계: G와 D input 모두에 condition 추가

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding이 가장 단순한 방법입니다. 더 큰 모델은 learned embedding, FiLM modulation, cross-attention을 사용합니다.

### 2단계: conditional 학습

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

generator는 marginal이 아니라 *주어진 condition*에 대한 real distribution과 일치해야 합니다.

### 3단계: class별 output 검증

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 함정

- **Condition이 무시됨.** G가 marginalize하도록 학습하고, condition signal이 약해 D가 penalize하지 못합니다. 해결: D에 더 강하게 condition을 넣으세요(late가 아니라 early layer). projection discriminator(Miyato & Koyama 2018)를 사용하세요.
- **L1 weight가 너무 낮음.** G가 faithful한 output이 아니라 임의의 real-looking output으로 drift합니다. Pix2Pix-style task에서는 λ≈100에서 시작하세요.
- **L1 weight가 너무 높음.** L1도 여전히 L_p norm이기 때문에 G가 흐릿한 output을 만듭니다. training이 안정화되면 anneal down하세요.
- **D의 ground-truth leakage.** D input으로 `y`만이 아니라 `(x, y)`를 concatenate하세요. 그렇지 않으면 D는 consistency를 확인할 수 없습니다.
- **class별 mode collapse.** 각 class가 독립적으로 collapse할 수 있습니다. class-conditional diversity check를 실행하세요.

## 사용하기

2026년 image-to-image task의 상태:

| 작업 | 가장 맞는 접근법 |
|------|---------------|
| Sketch → photo, same domain, paired data | Pix2Pix / Pix2PixHD(여전히 빠르고 선명함) |
| Sketch → photo, unpaired | Scribble conditioning model을 쓰는 ControlNet |
| Semantic seg → photo | SPADE / GauGAN2 또는 SD + ControlNet-Seg |
| Style transfer | IP-Adapter 또는 LoRA를 쓰는 diffusion. GAN method는 legacy |
| Depth → photo | Stable Diffusion 위의 ControlNet-Depth |
| Super-resolution | Real-ESRGAN(GAN), ESRGAN-Plus, 또는 SD-Upscale(diffusion) |
| Colorization | ColTran, diffusion-based colorizer, 또는 Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN 또는 ControlNet-based |

Pix2Pix는 (a) 수천 개의 paired example이 있고, (b) task가 narrow하고 repeatable하며, (c) 빠른 inference가 필요할 때 여전히 올바른 도구입니다. generic open-domain task에서는 diffusion이 이깁니다.

## 출시하기

`outputs/skill-img2img-chooser.md`를 저장하세요. Skill은 task description, data availability(paired vs unpaired, N samples), latency/quality budget을 받아 approach(Pix2Pix, CycleGAN, ControlNet variant, SDXL + IP-Adapter), training data requirements, inference cost, eval protocol(LPIPS, FID, task-specific)을 출력합니다.

## 연습문제

1. **쉬움.** `code/main.py`를 수정해 세 번째 class를 추가하세요. G가 여전히 각 class의 noise를 올바른 mode로 mapping하는지 확인하세요.
2. **중간.** 1-D setting에서 L1을 perceptual-style loss로 바꾸세요(예: feature extractor 역할을 하는 작은 frozen D). conditional distribution의 sharpness가 바뀌나요?
3. **어려움.** 1-D setting에서 CycleGAN을 스케치하세요. 두 distribution, 두 generator, cycle loss입니다. paired data 없이 그 둘 사이를 mapping하는 법을 학습함을 보이세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Conditional GAN | "label이 있는 GAN" | G(z, c), D(x, c). 두 network 모두 condition을 봅니다. |
| Pix2Pix | "Image-to-image GAN" | U-Net G와 PatchGAN D + L1 loss를 쓰는 paired cGAN. |
| U-Net | "skip이 있는 encoder-decoder" | 대칭 conv network. skip은 high-freq를 보존합니다. |
| PatchGAN | "Local-realism classifier" | D가 global score 대신 patch별 score를 출력합니다. |
| CycleGAN | "Unpaired image translation" | 두 G + cycle-consistency loss. paired data가 없습니다. |
| SPADE | "GauGAN" | semantic map으로 intermediate activation을 normalize합니다. segmentation-to-image. |
| FiLM | "Feature-wise linear modulation" | condition에서 온 feature별 affine transform. 저렴한 conditioning입니다. |

## 프로덕션 노트: latency-bound baseline으로서의 Pix2Pix

paired data와 narrow task(sketch → render, semantic map → photo, day → night)가 있으면 Pix2Pix의 one-shot inference는 latency에서 diffusion보다 한 자릿수 이상 빠릅니다. production comparison은 보통 다음과 같습니다.

| Path | Steps | 단일 L4의 512² 기준 typical latency |
|------|-------|------------------------------------|
| Pix2Pix(U-Net forward) | 1 | ~30 ms |
| SD-Inpaint 또는 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix는 static batch에서 throughput이 좋습니다(모든 request가 같은 FLOPs). Diffusion은 quality와 generalization에서 이깁니다. 현대적 방식은 narrow task에는 Pix2Pix-style distilled model을 배포하고 tail input에는 diffusion fallback을 두는 경우가 많습니다.

## 더 읽을거리

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 논문.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D.
