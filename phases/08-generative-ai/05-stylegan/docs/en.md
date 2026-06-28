# StyleGAN

> 대부분의 생성기는 `z`를 모든 레이어에 동시에 섞어 넣습니다. StyleGAN은 이것을 분리했습니다. 먼저 `z`를 중간 표현 `w`로 매핑한 다음, AdaIN을 통해 모든 해상도 레벨에 `w`를 *주입*합니다. 이 한 가지 변화가 latent space의 얽힘을 풀었고, 7년 넘게 포토리얼리스틱 얼굴 생성을 사실상 해결된 문제로 만들었습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 08 (Normalization), Phase 3 · 07 (CNNs)
**Time:** ~45 minutes

## 문제

DCGAN은 transposed convolution 스택을 통해 `z`를 이미지로 매핑합니다. 문제는 `z`가 포즈, 조명, 정체성, 배경 등 모든 것을 함께 얽힌 상태로 제어한다는 점입니다. `z`의 한 축을 따라 이동하면 네 가지가 모두 바뀝니다. 표현이 그렇게 분해되어 있지 않기 때문에 모델에 "같은 사람, 다른 포즈"를 요청할 수 없습니다.

Karras et al. (2019, NVIDIA)의 제안은 이렇습니다. `z`를 conv 레이어에 직접 넣지 않습니다. 네트워크 입력으로 상수 `4×4×512` 텐서를 넣습니다. `z ∈ Z → w ∈ W`를 매핑하는 8-layer MLP를 학습합니다. *adaptive instance normalization* (AdaIN)을 통해 모든 해상도에서 `w`를 주입합니다. 각 conv feature map을 정규화한 뒤, `w`의 affine projection으로 스케일하고 시프트합니다. 확률적 디테일(피부 모공, 머리카락 가닥)을 위해 레이어별 noise도 추가합니다.

결과적으로 `W`는 "high-level style"(포즈, 정체성)과 "fine style"(조명, 색상)에 대해 대체로 직교하는 축을 갖습니다. 이미지 A의 `w`를 저해상도 레벨에 쓰고 이미지 B의 `w`를 고해상도 레벨에 쓰면 두 이미지의 style을 서로 바꿀 수 있습니다. 이 변화가 편집, cross-domain stylization, 그리고 "StyleGAN inversion" 계열 연구 전체를 열었습니다.

## 개념

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network.** `f: Z → W`인 8-layer MLP입니다. `Z = N(0, I)^512`입니다. `W`는 Gaussian이 되도록 강제되지 않습니다. 데이터에 맞춘 형태를 학습합니다.

**Synthesis network.** 학습된 상수 `4×4×512`에서 시작합니다. 각 해상도 블록은 `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`입니다. 해상도는 4, 8, 16, 32, 64, 128, 256, 512, 1024로 두 배씩 증가합니다.

**AdaIN.**

```text
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

여기서 `y_scale`과 `y_bias`는 `w`의 affine projection에서 나옵니다. feature map별로 정규화한 뒤 다시 스타일을 입힙니다. 여기서 "style"은 feature map의 1차 및 2차 통계입니다.

**Per-layer noise.** 각 feature map에 single-channel Gaussian noise를 더하고, 학습된 채널별 계수로 스케일합니다. 전역 구조를 건드리지 않고 확률적 디테일을 제어합니다.

**Truncation trick.** 추론 시 `z`를 샘플링하고 `w = mapping(z)`를 계산한 뒤, `w' = ŵ + ψ·(w - ŵ)`를 사용합니다. 여기서 `ŵ`는 많은 샘플에 대한 평균 `w`입니다. `ψ < 1`은 다양성을 품질과 맞바꿉니다. 거의 모든 StyleGAN 데모는 `ψ ≈ 0.7`을 사용합니다.

## StyleGAN 1 → 2 → 3

| 버전 | 연도 | 혁신 |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing. |
| StyleGAN2 | 2020 | Weight demodulation이 AdaIN을 대체합니다(droplet artifact 수정). skip/residual architecture. path-length regularization. |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernel. 텍스처가 픽셀 격자에 달라붙는 문제를 제거합니다. |
| StyleGAN-XL | 2022 | Class-conditional, 1024², ImageNet. |
| R3GAN | 2024 | 더 강한 regularization으로 재정의한 모델. FFHQ-1024에서 diffusion과의 격차를 20배 적은 파라미터로 좁힙니다. |

2026년에도 StyleGAN3는 (a) 높은 FPS가 필요한 narrow-domain 포토리얼리즘, (b) few-shot domain adaptation(새 데이터셋 이미지 100장으로 학습하고 mapping을 freeze), (c) inversion 기반 편집(실제 사진을 재구성하는 `w`를 찾은 뒤 그 `w`를 편집)에서 기본 선택지로 남아 있습니다. open-domain text-to-image에는 적합한 도구가 아닙니다. 그 영역은 diffusion입니다.

## 직접 만들기

`code/main.py`는 1-D 장난감 "style-GAN lite"를 구현합니다. mapping MLP, 학습된 상수 벡터를 받아 `w`에서 나온 scale/bias로 modulation하는 synthesis 함수, 그리고 per-layer noise가 포함됩니다. affine modulation을 통해 `w`를 주입하는 방식이 generator 입력에 `z`를 이어 붙이는 방식과 같거나 더 낫다는 점을 보여줍니다.

### 1단계: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### 2단계: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

feature map별 scale과 bias는 linear projection을 통해 `w`에서 나옵니다.

### 3단계: layer별 noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

채널별 sigma는 학습 가능합니다.

## 함정

- **Droplet artifact.** StyleGAN 1은 AdaIN이 평균을 0으로 만들기 때문에 feature map에 덩어리진 droplet을 만들었습니다. StyleGAN 2의 weight demodulation은 activation 대신 convolution weight를 스케일하여 이 문제를 고칩니다.
- **Texture sticking.** StyleGAN 1과 2의 텍스처는 객체 좌표가 아니라 픽셀 좌표를 따라갔습니다(보간할 때 눈에 띕니다). StyleGAN 3의 alias-free convolution은 windowed sinc filter로 이를 해결합니다.
- **Mode coverage.** Truncation `ψ < 0.7`은 깔끔해 보이지만 좁은 원뿔 영역에서 샘플링합니다. 다양성이 필요하면 `ψ = 1.0`을 사용하세요.
- **Inversion은 손실이 있습니다.** 실제 사진을 `W`로 invert하는 작업은 보통 optimization이나 encoder(e4e, ReStyle, HyperStyle)로 수행합니다. 반복이 많아질수록 결과가 원본에서 벗어납니다.

## 활용하기

| 사용 사례 | 접근법 |
|----------|----------|
| 포토리얼 인간 얼굴(애니메이션, 제품, narrow domain) | StyleGAN3 FFHQ / custom fine-tune |
| 사진 기반 얼굴 편집 | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| 아바타 파이프라인 | 저데이터 fine-tune을 위한 StyleGAN3 w/ ADA |
| 적은 이미지에서 domain adaptation | Mapping network를 freeze하고 synthesis만 fine-tune |
| Multi-modal 또는 text-conditioned generation | 사용하지 마세요. diffusion을 쓰세요 |

답이 "사람 얼굴 사진"인 제품급 데모에서는 StyleGAN이 추론 비용(single forward pass, 4090에서 <10ms)과 같은 품질 기준에서의 선명도 측면에서 diffusion을 이깁니다.

## 출시하기

`outputs/skill-stylegan-inversion.md`를 저장하세요. 이 skill은 실제 사진을 받아 inversion method(e4e / ReStyle / HyperStyle), 예상 latent loss, editing budget(artifact가 생기기 전까지 `W`에서 얼마나 멀리 이동할 수 있는지), 그리고 검증된 editing direction 목록(age, expression, pose)을 출력합니다.

## 연습 문제

1. **쉬움.** `adain_on=True`와 `adain_on=False`로 `code/main.py`를 실행하세요. 고정 latent와 교란된 latent에 대한 출력 분포 폭을 비교하세요.
2. **중간.** Mixing regularization을 구현하세요. 학습 batch에서 `w_a`, `w_b`를 계산하고, synthesis의 앞 절반에는 `w_a`, 뒤 절반에는 `w_b`를 적용하세요. Decoder가 disentangled style을 학습하나요?
3. **어려움.** pretrained StyleGAN3 FFHQ 모델(`ffhq-1024.pkl`)을 사용하세요. label된 샘플로 SVM을 학습해 "smile"을 제어하는 `w` direction을 찾고, 정체성이 벗어나기 전까지 얼마나 밀 수 있는지 보고하세요.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-------------------|-----------|
| Mapping network | "The MLP" | `f: Z → W`, 8 layers, latent geometry를 data statistics에서 분리합니다. |
| W space | "The style space" | Mapping network의 출력. 대체로 disentangled되어 있습니다. |
| AdaIN | "Adaptive instance norm" | Feature map을 정규화한 뒤 `w` projection으로 scale + shift합니다. |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1은 다양성을 품질과 맞바꿉니다. |
| Path-length regularization | "PL reg" | `w`의 단위 변화당 이미지의 큰 변화를 penalize하여 `W`를 더 매끄럽게 만듭니다. |
| Weight demodulation | "The StyleGAN2 fix" | Activation 대신 conv weight를 정규화하여 droplet artifact를 제거합니다. |
| Alias-free | "StyleGAN3's trick" | Windowed sinc filter. 텍스처가 픽셀 격자에 달라붙는 현상을 없앱니다. |
| Inversion | "Find w for a real image" | `G(w) ≈ x`가 되도록 `x → w`를 optimize하거나 encode합니다. |

## 프로덕션 노트: 2026년에도 StyleGAN이 쓰이는 이유

4090에서 StyleGAN3는 1024² FFHQ 얼굴을 10ms 미만에 생성합니다. `num_steps = 1`이고, VAE decode도 없고, cross-attention pass도 없습니다. 제품 관점에서 이것은 이미지 생성기가 가질 수 있는 최저 지연 시간입니다. 같은 해상도에서 50-step SDXL + VAE-decode 파이프라인은 약 3초가 걸립니다. **300× 격차**이며, narrow-domain 제품(아바타 서비스, 신분증 문서 파이프라인, stock face generation)에서는 TCO에서 이깁니다.

운영 관점의 결과는 두 가지입니다.

- **Scheduler도 batcher도 없습니다.** 목표 occupancy에 맞춘 static batch가 최적입니다. Continuous batching(LLM과 diffusion에 필수)은 모든 요청이 같은 FLOPs를 쓰기 때문에 이점이 없습니다.
- **Truncation `ψ`가 safety knob입니다.** `ψ < 0.7`은 mapping network 범위의 좁은 원뿔에서 샘플링합니다. 이것이 serving layer가 sample variance를 제어할 수 있는 유일한 레버입니다. 피크 부하에서는 `ψ`를 낮추고, 프리미엄 사용자에게는 높이세요.

## 더 읽을거리

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe.
