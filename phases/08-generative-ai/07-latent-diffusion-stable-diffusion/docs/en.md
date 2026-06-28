# Latent Diffusion과 Stable Diffusion

> 512×512 이미지에서 pixel-space diffusion을 돌리는 것은 계산적으로 너무 비쌉니다. Rombach et al. (2022)은 이미지를 생성하는 데 786k 차원 전체가 필요하지 않다는 점을 보았습니다. 의미 구조를 담을 만큼만 있으면 되고, 나머지는 별도 decoder가 처리하면 됩니다. VAE의 latent space 안에서 diffusion을 실행합니다. 이 한 가지 아이디어가 Stable Diffusion입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 minutes

## 문제

512²에서 pixel-space diffusion을 하면 U-Net은 `[B, 3, 512, 512]` shape의 tensor에서 실행됩니다. Sampling step 하나가 500M-param U-Net 기준 약 100 GFLOPS입니다. 50 step이면 이미지 한 장당 5 TFLOPS입니다. 10억 장 이미지로 학습하면 compute bill은 터무니없이 커집니다.

그 FLOPs의 대부분은 지각적으로 중요하지 않은 디테일, 즉 손실 VAE가 압축해 버릴 수 있는 고주파 texture를 net 전체로 밀어 넣는 데 쓰입니다. Rombach의 아이디어는 이렇습니다. VAE를 한 번 학습하고(*first stage*) freeze한 뒤, 4-channel 64×64 latent space 안에서만 diffusion을 실행합니다(*second stage*). 같은 U-Net입니다. 픽셀 수는 1/16입니다. 비슷한 품질에서 FLOPs는 약 64배 적습니다.

이것이 Stable Diffusion 레시피입니다. SD 1.x / 2.x는 `64×64×4` latent 위에 860M U-Net을 사용했고, SDXL은 `128×128×4` 위에 2.6B U-Net을 사용했으며, SD3는 U-Net을 flow matching을 쓰는 Diffusion Transformer(DiT)로 바꿨습니다. Flux.1-dev(Black Forest Labs, 2024)는 12B-param DiT-MMDiT를 제공합니다. 모두 같은 two-stage substrate 위에서 동작합니다.

## 개념

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Two stages, separately trained.**

1. **Stage 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`입니다. 목표 compression은 각 spatial axis에서 8× downsample하고 channel을 조정해 전체 latent 크기를 pixel count의 약 1/16로 만드는 것입니다. Loss = reconstruction(L1 + LPIPS perceptual) + KL입니다. KL은 작은 weight를 사용합니다. `z`에서 정확히 sampling할 필요가 없으므로 `z`를 너무 Gaussian으로 강제하지 않기 위해서입니다. Decoded image가 선명하도록 adversarial loss와 함께 학습하는 경우가 많습니다.

2. **Stage 2 — diffusion on `z`.** `z = E(x_real)`를 데이터로 취급합니다. U-Net(또는 DiT)을 학습해 `z_t`를 denoise합니다. 추론 시 diffusion으로 `z_0`를 샘플링한 뒤 `x = D(z_0)`로 decode합니다.

**Text conditioning.** 추가 component가 두 개 있습니다. Frozen text encoder(SD 1.x는 CLIP-L, SD 2/XL은 CLIP-L+OpenCLIP-G, SD3와 Flux는 T5-XXL)와 cross-attention injection입니다. 모든 U-Net block은 `[Q = image features, K = V = text tokens]`를 받아 섞습니다. Text가 이미지에 영향을 주는 유일한 경로는 이 token들입니다.

**Loss function은 Lesson 06과 동일합니다.** Noise에 대한 같은 DDPM / flow matching MSE입니다. 데이터 domain만 바꿉니다.

## 아키텍처 변형

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

추세는 U-Net을 DiT(latent patch 위의 transformer)로 바꾸고, text encoder를 키우며(prompt adherence에서는 T5가 CLIP보다 낫습니다), latent channel을 늘리는 것입니다(4 → 16은 디테일 여지를 더 줍니다).

```figure
noise-schedule
```

## 직접 만들기

`code/main.py`는 Lesson 06의 DDPM 위에 장난감 1-D "VAE"(시연용 identity encoder + decoder이며, 실제 VAE는 conv net입니다)를 쌓고 classifier-free guidance로 class conditioning을 추가합니다. Raw 1-D 값에서 실행하든 encoded 값에서 실행하든 같은 diffusion loss가 작동한다는 핵심 통찰을 보여줍니다.

### 1단계: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

실제 VAE에는 학습된 weight가 있습니다. 교육용으로는 이 linear map만으로도 diffusion이 원래 data space에 신경 쓰지 않고 `z` 위에서 동작한다는 점을 보이기에 충분합니다.

### 2단계: `z`-space의 diffusion

Lesson 06과 같은 DDPM입니다. Net이 보는 데이터는 `z = E(x)`입니다. `z_0`를 샘플링한 뒤 `D(z_0)`로 decode합니다.

### 3단계: classifier-free guidance

학습 중 class label을 10% 확률로 drop합니다(null token으로 대체). 추론 시 `ε_cond`와 `ε_uncond`를 모두 계산한 뒤 다음을 사용합니다.

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0`은 guidance 없음(전체 다양성), `w = 3`은 기본값, `w = 7+`는 saturated / over-sharp입니다.

### 4단계: text conditioning(개념, 코드 아님)

Class label을 frozen text encoder output으로 바꿉니다. Cross-attention을 통해 text embedding을 U-Net에 넣습니다.

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

이것이 class-conditional diffusion model과 Stable Diffusion 사이의 유일한 본질적 차이입니다.

## 함정

- **VAE-scale mismatch.** SD 1.x VAE는 encoding 뒤에 적용되는 scaling constant(`scaling_factor ≈ 0.18215`)가 있습니다. 이것을 잊으면 U-Net이 variance가 크게 잘못된 latent 위에서 학습합니다. 모든 checkpoint가 이 값을 포함합니다.
- **Text encoder가 조용히 틀립니다.** SD3는 >=128 token의 T5-XXL이 필요하며, CLIP-only fallback은 손실이 큽니다. 항상 `use_t5=True`를 확인하세요. 그렇지 않으면 prompt fidelity가 무너집니다.
- **Latent space를 섞는 문제.** SDXL, SD3, Flux는 모두 다른 VAE를 사용합니다. SDXL latent에서 학습한 LoRA는 SD3에서 동작하지 않습니다. Hugging Face diffusers 0.30+는 맞지 않는 checkpoint load를 거부합니다.
- **CFG가 너무 높습니다.** `w > 10`은 saturated하고 기름진 이미지를 만들며, 다양성을 희생해 prompt에 과적합합니다. 적정 범위는 `w = 3-7`입니다.
- **Negative prompt leaking.** 빈 negative prompt는 null token이 되고, 채워진 negative prompt는 `ε_uncond`가 됩니다. 둘은 같지 않습니다. 일부 pipeline은 조용히 null을 기본값으로 씁니다.

## 활용하기

2026년 production stack:

| Target | Recommended backbone |
|--------|----------------------|
| Narrow domain, paired data, 처음부터 model training | SDXL fine-tune (LoRA / full) — 가장 빠른 출시 경로 |
| Open-domain text-to-image, open weights | Flux.1-dev (12B, Apache / non-commercial) or SD3.5-Large |
| 가장 빠른 inference, open weights | Flux.1-schnell (1-4 step, Apache) or SDXL-Lightning |
| 최고의 prompt adherence, hosted | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| Edit workflows | Flux.1-Kontext (Dec 2024) — image + text를 native로 받음 |
| Research, baseline | SD 1.5 — 오래됐지만 잘 연구됨 |

## 출시하기

`outputs/skill-sd-prompter.md`를 저장하세요. 이 skill은 text prompt + target style을 받아 model + checkpoint, CFG scale, sampler, negative prompt, resolution, optional ControlNet/IP-Adapter combo, per-step QA checklist를 출력합니다.

## 연습 문제

1. **쉬움.** Guidance `w ∈ {0, 1, 3, 7, 15}`로 `code/main.py`를 실행하세요. Class별 mean sample을 기록하세요. 어느 `w`에서 class mean이 실제 data mean을 지나쳐 벌어지나요?
2. **중간.** 장난감 linear encoder를 reconstruction loss가 있는 tanh-MLP encoder/decoder pair로 바꾸세요. 새 latent에서 diffusion을 다시 학습하세요. 샘플 품질이 바뀌나요?
3. **어려움.** Diffusers로 실제 Stable Diffusion inference를 설정하세요. `sdxl-base`를 load하고 CFG=7로 30 Euler step을 실행한 뒤 시간을 재세요. 이제 `sdxl-turbo`로 바꿔 4 step, CFG=0으로 실행하세요. 같은 subject, 다른 품질입니다. 무엇이 왜 달라졌는지 설명하세요.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-------------------|-----------|
| First stage | "The VAE" | 학습된 encoder/decoder pair입니다. 512²를 64²로 압축합니다. |
| Second stage | "The U-Net" | Latent space 위의 diffusion model입니다. |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`입니다. Conditioning 강도를 조정합니다. |
| Null token | "Empty prompt embed" | `ε_uncond`에 쓰는 unconditional embed입니다. |
| Cross-attention | "How text gets in" | 각 U-Net block이 text token을 K와 V로 attend합니다. |
| DiT | "Diffusion Transformer" | U-Net을 latent patch 위의 transformer로 대체합니다. 더 잘 scale됩니다. |
| MMDiT | "Multi-modal DiT" | SD3 architecture입니다. Text stream과 image stream이 joint attention을 사용합니다. |
| VAE scaling factor | "Magic number" | Latent를 약 5.4로 나누어 diffusion이 unit-variance space에서 동작하게 합니다. |

## 프로덕션 노트: 8GB 소비자 GPU에서 Flux-12B 실행하기

Reference Flux integration은 "소비자 GPU가 있는데, 이걸 배포할 수 있을까?"에 대한 표준 레시피입니다. 핵심은 production inference literature가 diffusion DiT에 적용한다고 설명하는 같은 세 가지 knob입니다.

1. **Staggered loading.** Flux에는 VRAM에 동시에 있을 필요가 없는 네트워크가 세 개 있습니다. T5-XXL text encoder(fp32에서 ~10 GB), CLIP-L(작음), 12B MMDiT, VAE입니다. 먼저 prompt를 encode하고, encoder를 *delete*한 뒤, DiT를 load해 denoise하고, DiT를 *delete*한 뒤, VAE를 load해 decode합니다. 8GB 소비자 GPU에는 한 번에 한 stage만 들어갑니다.
2. **bitsandbytes를 통한 4-bit quantization.** T5 encoder와 DiT 모두에 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`를 적용합니다. Memory를 8× 줄이며, Aritra의 benchmark(노트북에 링크됨)에 따르면 text-to-image에서 품질 저하는 감지하기 어렵습니다.
3. **CPU offload.** `pipe.enable_model_cpu_offload()`는 각 forward pass가 진행될 때 module을 CPU와 GPU 사이에서 자동으로 swap합니다. 지연 시간이 10-20% 늘지만 pipeline이 실행 가능해집니다.

Memory 계산은 이렇습니다. `10 GB T5 / 8 = 1.25 GB` quantized, `12 B params × 0.5 bytes = ~6 GB` quantized DiT, 여기에 activation이 더해집니다. stas00의 표현을 빌리면 이것은 TP=1 inference의 극단입니다. Model parallelism은 없고, quantization은 최대입니다. Production이라면 H100에서 TP=2나 TP=4를 쓰겠지만, 단일 개발 노트북에서는 이 레시피가 맞습니다.

## 더 읽을거리

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 family.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 위 checkpoint들의 reference implementation.
