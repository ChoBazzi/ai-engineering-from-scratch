# 생성 모델 — 분류 체계와 역사

> 모든 이미지 모델, 텍스트 모델, 비디오 모델, 3D 모델은 다섯 가지 묶음 중 하나에 들어갑니다. 잘못된 묶음을 고르면 몇 주 동안 수학과 싸우게 됩니다. 올바른 묶음을 고르면 지난 12년 동안 이 분야가 쌓아 온 발전이 머릿속에 깔끔하게 정리됩니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 minutes

## 문제

생성 모델은 한 가지 일을 합니다. 어떤 알 수 없는 분포 `p_data(x)`에서 뽑힌 training sample이 주어졌을 때, 같은 분포에서 온 것처럼 보이는 새 sample을 출력하는 일입니다. 얼굴, 문장, MIDI 파일, 단백질 구조도 조금 멀리서 보면 모두 같은 문제입니다.

문제는 `p_data`가 수백만 차원의 공간에 산다는 점입니다. 512x512 RGB 이미지는 약 786k 차원입니다. sample들은 그 공간 안의 얇은 manifold 위에 놓여 있고, 예시는 많아야 10M개 정도뿐입니다. density를 brute-force로 구하는 것은 가망이 없습니다. 모든 생성 모델은 어려운 문제 하나를 조금 덜 어려운 문제로 바꾸는 타협입니다.

지난 12년 동안 다섯 계열이 살아남았습니다. 각 계열이 어떤 타협을 하는지 알면 왜 어떤 작업에서는 이기고 어떤 작업에서는 무너지는지 이해할 수 있습니다.

## 개념

![생성 모델의 다섯 계열 — 무엇을 모델링하는지에 따른 분류](../assets/taxonomy.svg)

**1. Explicit density, tractable.** 실제로 평가할 수 있는 합으로 `log p(x)`를 씁니다. Autoregressive 모델(PixelCNN, WaveNet, GPT)은 `p(x) = ∏ p(x_i | x_<i)`로 factorize합니다. Normalizing flow(RealNVP, Glow)는 단순한 base distribution의 invertible transform으로 `p(x)`를 만듭니다. 장점: exact likelihood와 깔끔한 training loss. 단점: autoregressive inference는 순차적이라 긴 sequence에서 느리고, flow는 invertible architecture가 필요해 구조가 제한됩니다.

**2. Explicit density, approximate.** `log p(x)`를 아래에서 bound(ELBO)하고 그 bound를 optimize합니다. VAE(Kingma 2013)는 variational posterior를 가진 encoder-decoder를 사용합니다. Diffusion 모델(DDPM, Ho 2020)은 weighted ELBO를 암묵적으로 optimize하는 denoiser를 학습합니다. 2026년 기준 diffusion은 이미지, 비디오, 3D의 지배적인 backbone입니다.

**3. Implicit density.** density를 완전히 건너뜁니다. sample을 만드는 generator `G(z)`와 real/fake를 구분하는 discriminator `D(x)`를 학습합니다. GAN(Goodfellow 2014)입니다. inference는 빠르지만(forward pass 한 번) training은 불안정하기로 악명이 높습니다. StyleGAN 1/2/3은 2026년에도 얼굴, 침실처럼 고정 domain의 photorealism에서 state of the art입니다.

**4. Score-based / continuous-time.** log-density의 gradient `∇_x log p(x)`(score)를 직접 학습합니다. Song & Ermon(2019)은 score matching이 diffusion을 SDE로 일반화한다는 것을 보였습니다. Flow matching(Lipman 2023)은 2024-2026년의 핵심 흐름입니다. simulation-free training, 더 곧은 path, DDPM보다 4-10배 빠른 sampling을 제공합니다. Stable Diffusion 3, Flux, AudioCraft 2는 모두 flow matching을 사용합니다.

**5. Discrete code 위의 token-based autoregressive.** VQ-VAE나 residual quantizer로 high-dimensional data를 짧은 discrete token sequence로 압축한 뒤 Transformer로 token sequence를 모델링합니다. Parti, MuseNet, AudioLM, VALL-E, Sora의 patch tokenizer가 모두 이 방식을 씁니다. 학습된 tokenizer를 더한 bucket 1이라고 볼 수 있습니다.

## 짧은 역사

| 연도 | 모델 | 왜 중요했나 |
|------|-------|-------------|
| 2013 | VAE (Kingma) | 사용 가능한 training loss를 가진 첫 deep generative model. |
| 2014 | GAN (Goodfellow) | Implicit density, likelihood 없음. 놀랄 만큼 선명한 sample. |
| 2015 | DRAW, PixelCNN | 순차적 이미지 생성. |
| 2017 | Glow, RealNVP | Invertible flow. 깊이를 갖춘 exact likelihood. |
| 2017 | Progressive GAN | 첫 megapixel 얼굴. |
| 2019 | StyleGAN / StyleGAN2 | 해당 domain에서는 여전히 이기기 어려운 photorealistic 얼굴. |
| 2020 | DDPM (Ho) | Diffusion이 실용화됨. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image가 mainstream이 됨. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = commodity. |
| 2022 | ControlNet, LoRA | pretrained diffusion에 대한 정밀 제어. |
| 2023 | SDXL, Midjourney v5, Flow matching | scale + 더 나은 training dynamics. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion. flow matching의 승리. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | production-grade video. |
| 2026 | Consistency + Rectified Flow | diffusion backbone에서 one-step sampling. |

## 다섯 질문 triage

새 생성 모델 논문이 나오면 method section을 읽기 전에 이 다섯 질문에 답하세요.

1. **무엇을 모델링하는가?** pixel, latent, discrete token, 3D Gaussian, mesh, waveform?
2. **density가 explicit인가 implicit인가?** `log p(x)`를 적어 내는가?
3. **Sampling이 one-shot인가 iterative인가?** iterative는 inference가 느리다는 뜻이고, one-shot은 대개 adversarial 또는 distilled라는 뜻입니다.
4. **Conditioning은 무엇인가? unconditional, class, text, image, pose?** 이것이 loss와 architecture scaffolding을 결정합니다.
5. **Evaluation은 무엇인가? FID, CLIP score, IS, human preference, task accuracy?** 각각 알려진 failure mode가 있습니다(Lesson 14 참고).

이 phase의 모든 레슨에서 이 다섯 질문에 다시 답하게 됩니다. 끝날 때쯤에는 반사적으로 떠올릴 수 있어야 합니다.

## 직접 만들기

이 레슨의 코드는 가벼운 visualization입니다. 1-D mixture-of-Gaussians sample을 세 가지 toy approach(kernel density, discrete histogram, nearest-sample "GAN-ish" generator)로 fit해서, 한 화면에 출력할 수 있는 문제에서 explicit density와 implicit density의 차이를 볼 수 있게 합니다.

`code/main.py`를 실행하세요. 두 mode를 가진 Gaussian mixture에서 2000개 sample을 뽑은 뒤 다음을 출력합니다.

```text
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

주목하세요. 앞의 두 방식은 "이 점은 얼마나 그럴듯한가?"를 물을 수 있습니다. 세 번째는 그럴 수 없습니다. 이것이 앞으로 모든 레슨에서 중요해질 *explicit vs implicit* 구분입니다.

## 사용하기

2026년에 어떤 task에는 어떤 계열을 써야 할까요?

| 작업 | 가장 맞는 계열 | 이유 |
|------|-------------|-----|
| Photoreal faces, narrow domain | StyleGAN 2/3 | 여전히 가장 선명하고 inference가 가장 빠릅니다. |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Fast text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Speech + music | Token-based AR(AudioLM, VALL-E, MusicGen) 또는 flow matching(AudioCraft 2) | Discrete token은 scale 비용이 낮습니다. |
| 3D scenes | Gaussian Splatting fit, diffusion prior | reconstruction은 3D-GS, novel-view는 diffusion. |
| Density estimation(no sampling) | Flows | exact `log p(x)`를 가진 유일한 계열. |
| Simulation / physics | Flow matching, score SDE | Straight-line path와 smooth vector field. |

## 출시하기

`outputs/skill-model-chooser.md`로 저장하세요.

이 skill은 task description을 받아 (1) 사용할 계열, (2) open option 세 개와 hosted option 세 개의 ranked list, (3) 주의해야 할 likely failure mode, (4) compute/time budget을 출력합니다.

## 연습문제

1. **쉬움.** 다음 다섯 제품 각각의 family와 backbone을 식별하세요: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. 근거는 public technical report에서 찾아야 합니다.
2. **중간.** 내일 읽을 논문이 diffusion보다 sampling이 100배 빠르다고 주장합니다. 그 speedup이 conditioning과 high resolution에서도 유지되는지 확인할 질문 세 가지를 적으세요.
3. **어려움.** 관심 있는 domain 하나(예: protein structure, CAD, molecules, trajectories)를 고르세요. 그 domain의 현재 SOTA 모델에 대해 다섯 질문 triage에 답하고, 더 나은 모델이라면 무엇을 바꿀지 스케치하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| Generative model | "새로운 것을 만든다" | `p_data(x)`의 sampler를 학습하고, 선택적으로 `log p(x)`를 노출합니다. |
| Explicit density | "평가할 수 있다" | 모델이 closed-form 또는 tractable `log p(x)`를 제공합니다. |
| Implicit density | "GAN-style" | sampler만 있습니다. 주어진 점의 `p(x)`를 평가할 방법이 없습니다. |
| ELBO | "Evidence lower bound" | `log p(x)`의 tractable lower bound. VAE와 diffusion이 이를 optimize합니다. |
| Score | "Log-density의 gradient" | `∇_x log p(x)`. diffusion과 SDE 모델은 이 field를 학습합니다. |
| Manifold hypothesis | "Data는 surface 위에 산다" | High-dimensional data는 low-dimensional manifold에 집중됩니다. dimensionality reduction이 작동하는 이유입니다. |
| Autoregressive | "다음 조각을 예측한다" | joint를 conditional의 곱으로 factorize합니다. |
| Latent | "압축된 code" | decoder가 input을 reconstruct할 수 있는 low-dimensional representation. |

## 프로덕션 노트: 다섯 계열, 다섯 inference 형태

각 계열은 서로 다른 inference-server cost curve로 이어집니다. production-inference 문헌은 LLM inference를 prefill + decode로 설명합니다. 같은 분해를 여기에도 적용할 수 있습니다.

- **Autoregressive(bucket 1 and 5).** sequential decode가 latency를 지배합니다. KV-cache, continuous batching, speculative decoding이 모두 직접 적용됩니다.
- **VAE / diffusion / flow-matching(buckets 2 and 4).** LLM 의미의 decode가 없습니다. Cost = `num_steps × step_cost`이고, `step_cost`는 full latent resolution에서의 transformer 또는 U-Net forward입니다. production knob은 step count(DDIM / DPM-Solver / distillation), batch size, precision(bf16 / fp8 / int4)입니다.
- **GAN(bucket 3).** forward pass 한 번입니다. schedule도 KV-cache도 없습니다. TTFT ≈ total latency입니다. 이것이 StyleGAN이 narrow-domain UX에서 여전히 이기는 이유입니다.

논문 abstract에서 "diffusion보다 빠르다"를 보면 "더 적은 steps × 같은 step cost" 또는 "같은 steps × 더 싼 step cost"로 번역하세요. 나머지는 marketing입니다.

## 더 읽을거리

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 논문.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 논문.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 논문.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — SDE로 보는 diffusion.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching 논문.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
