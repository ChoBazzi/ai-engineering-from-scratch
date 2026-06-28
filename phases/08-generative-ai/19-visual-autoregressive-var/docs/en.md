# Visual Autoregressive Modeling(VAR): Next-Scale Prediction

> Diffusion 모델은 시간 방향으로 반복 샘플링합니다(denoising steps). VAR은 scale 방향으로 반복 샘플링합니다. 먼저 1x1 token을 예측하고, 그다음 2x2, 4x4를 거쳐 최종 해상도까지 올라가며, 각 scale은 이전 scale을 조건으로 삼습니다. 2024년 논문은 VAR이 이미지 생성에서 GPT-style scaling law를 따르고 같은 compute budget에서 DiT를 앞선다는 것을 보였습니다. 이 lesson은 핵심 메커니즘을 직접 만듭니다.

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 minutes

## 문제

Autoregressive generation은 예측 가능하게 scale되기 때문에 language modeling을 지배했습니다. Compute와 parameter가 늘어나면 perplexity가 낮아지고 출력이 좋아집니다. 2024년 전 이미지 생성에서 주요 AR 시도는 두 가지였습니다. PixelRNN/PixelCNN(pixel-by-pixel)과 DALL-E 1 / Parti / MuseGAN(VQ-VAE code 위 token-by-token)입니다.

둘 다 generation-order problem에 시달렸습니다. Pixel과 token은 2D grid에 배치되어 있지만, AR 모델은 1D raster order로 방문해야 합니다. 초반 corner pixel은 이미지가 결국 무엇이 될지 알 수 없습니다. 생성 품질은 GPT-on-text보다 나쁘게 scale되었고, 같은 compute에서 diffusion-model 품질에 도달하지 못했습니다.

VAR은 생성되는 대상을 바꿔 generation-order problem을 해결합니다. 이미지 token을 공간에서 하나씩 예측하는 대신, VAR은 해상도가 증가하는 전체 이미지를 예측합니다. 1단계: 1x1 token(전체 이미지 "summary")을 예측합니다. 2단계: 2x2 token grid(더 거친 feature)를 예측합니다. 3단계: 4x4 grid를 예측합니다. K단계: 최종 (H/8)x(W/8) grid를 예측합니다.

각 scale은 이전 모든 scale에 attention을 둡니다("scale order"에서 causal). 자기 scale 내부에서는 parallel입니다. Scale k의 전체 이미지가 transformer pass 한 번으로 만들어지므로 order problem이 사라집니다.

## 개념

### VQ-VAE Multi-Scale Tokenizer

VAR에는 **multi-scale discrete tokenizer**가 필요합니다. 이미지 x에 대해 점점 더 높은 해상도의 token grid sequence를 만듭니다.

```text
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

각 z_k는 같은 codebook을 사용합니다(일반적인 크기는 4096-16384). 각 scale의 tokenization은 독립적이지 않습니다. 각 scale의 residual을 합하면 f를 재구성하도록 학습됩니다.

```text
f ≈ upsample(embed(z_1), target_size) + ... + upsample(embed(z_K), target_size)
```

이는 **residual VQ** 변형입니다. Scale k는 scale 1..k-1이 놓친 것을 포착합니다. Decoder는 모든 scale embedding의 합을 받아 이미지를 생성합니다.

Multi-scale VQ tokenizer는 VQGAN처럼 한 번 학습한 뒤 freeze합니다. 생성 작업은 모두 그 위의 autoregressive model이 수행합니다.

### Next-Scale Prediction

생성 모델은 이전 모든 scale의 token을 보고 다음 scale의 token을 예측하는 transformer입니다.

Input sequence structure:
```text
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

Position embedding은 scale index와 scale 내부 spatial position을 모두 인코딩합니다. Attention은 scale order에서 causal입니다. Scale k, position (i, j)의 token은 scale 1..k의 모든 token과 scale k 자체에서 intra-scale order상 앞에 오는 token에 attend할 수 있습니다. VAR은 고정 positional attention을 사용하고 intra-scale causality는 두지 않습니다. 한 scale 안의 모든 position이 parallel하게 예측됩니다.

Training loss: 각 scale k에서 prior-scale token이 주어졌을 때 token z_k를 예측합니다. Discrete VQ code에 대한 cross-entropy loss입니다. "sequence"가 scale-structured로 바뀌었을 뿐 GPT와 같은 구조입니다.

### 생성

추론 때:

```text
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

K = 10 scales라면 생성은 transformer forward pass 10번입니다. 각 pass는 해당 scale 전체를 parallel하게 생성합니다. Scale 내부에서 token별 autoregression은 없습니다. 256x256 이미지 기준으로 DiT의 28-50회 대비 대략 10회 pass입니다.

### Next-Scale이 Next-Token을 이기는 이유

구조적 이점은 세 가지입니다.
1. **Coarse-to-fine은 자연 이미지 통계와 맞습니다.** 사람의 시각 인식과 이미지 dataset 모두 scale-dependent regularity를 보입니다. Low-frequency structure는 안정적이고 예측 가능하며, high-frequency detail은 low-frequency content에 조건화됩니다. Next-scale prediction은 이를 활용합니다.
2. **Scale 내부 parallel generation.** GPT-style token AR과 달리 VAR은 한 scale의 모든 token을 한 단계에서 생성합니다. Effective generation length가 linear가 아니라 log-scale입니다.
3. **Generation order bias가 없습니다.** Scale k의 token은 scale k-1 전체를 봅니다. Early token이 late context가 생기기 전에 결정하도록 강제하는 "left-of" 또는 "above" bias가 없습니다.

### Scaling law

Tian et al.은 VAR이 ImageNet에서 FID에 대해 power-law scaling curve를 따른다는 것을 보였습니다. GPT가 perplexity에 대해 보이는 것과 같습니다. Parameter나 compute를 두 배로 늘리면 error가 안정적으로 절반이 됩니다. 이는 language model만큼 깔끔한 scaling behavior를 보인 최초의 image-generative model이었습니다. 그 결과 VAR-scale prediction은 architecture별 경험적 추측이 아니라 compute로부터 예측 가능해집니다.

### Diffusion과의 관계

VAR과 diffusion은 같은 data-compression 이야기를 공유합니다. 둘 다 생성 문제를 더 쉬운 subproblem의 sequence로 나눕니다.

- Diffusion: 노이즈를 점진적으로 추가하고, 한 단계를 되돌리는 법을 학습합니다.
- VAR: 해상도를 점진적으로 추가하고, 다음 scale을 예측하는 법을 학습합니다.

둘은 문제를 가로지르는 서로 다른 축입니다. 둘 다 다루기 쉬운 conditional distribution을 만듭니다. 경험적으로 VAR은 추론이 더 빠르고(pass가 적고 scale 내부가 모두 parallel), class-conditional ImageNet에서 DiT와 맞먹거나 앞섭니다. Text-conditional VAR(VARclip, HART)은 활발한 연구 방향입니다.

## 직접 만들기

`code/main.py`에서 다음을 수행합니다.
1. Synthetic "image" data(2D Gaussian rings) 위에 작은 **multi-scale VQ tokenizer**를 만듭니다.
2. **VAR-style transformer**를 학습해 token을 next-scale-predict하게 합니다.
3. Transformer를 4번 호출해(4 scales) 샘플링하고 decode합니다.
4. Scale-ordered training이 scale 내부 parallel generation을 실제로 가능하게 하는지 검증합니다.

이는 toy implementation입니다. 핵심은 scale-structured attention mask와 parallel-within-scale generation이 실제로 작동하는 모습을 보는 것입니다.

## 출시하기

이 lesson은 `outputs/skill-var-tokenizer-designer.md`를 만듭니다. 이 skill은 multi-scale tokenizer를 설계합니다. Scale 수, scale ratio, codebook size, residual sharing, decoder architecture를 다룹니다.

## 연습 문제

1. **Scale count ablation.** 4, 6, 8, 10 scales로 VAR을 학습하세요. Autoregressive pass 수 대비 reconstruction quality를 측정하세요. Scale이 많을수록 residual이 더 세밀해져 품질은 좋아지지만 pass 수가 늘어납니다.

2. **Codebook size.** Codebook size 512, 4096, 16384로 tokenizer를 학습하세요. Codebook이 클수록 reconstruction은 좋아지지만 prediction은 더 어려워집니다. Knee를 찾으세요.

3. **Parallel-within-scale check.** 학습된 VAR에서 attention pattern을 명시적으로 측정하세요. Scale k 내부에서 모델이 cross-scale position에는 attend하지만 intra-scale에는 attend하지 않나요? Mask implementation을 검증하세요.

4. **VAR vs DiT scaling.** 같은 ImageNet class-conditional task에서 matched param budgets(예: 33M, 130M, 458M)로 VAR과 DiT를 학습하세요. FID vs compute를 그리세요. 각 size에서 VAR이 DiT를 앞서야 합니다. 논문의 결과를 작은 scale에서 재현하세요.

5. **Text conditioning.** VAR을 확장해 text embedding(CLIP pooled)을 adaLN을 통한 추가 conditioning input으로 받게 하세요. 이것이 HART recipe입니다. Text-aligned sampling에서 FID가 얼마나 개선되나요?

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| VAR | "Visual AutoRegressive" | VQ token grid의 pyramid 위에서 next-scale prediction으로 하는 이미지 생성 |
| Next-scale prediction | "더 거친 것부터 더 세밀하게 예측" | 모델이 이전 모든 scale에 조건화해 증가하는 해상도 scale의 token을 예측합니다. |
| Multi-scale VQ tokenizer | "Residual VQ" | 증가하는 해상도의 K개 token grid를 만들고 decoder가 모든 scale을 합산하는 VQ-VAE입니다. |
| Scale k | "Pyramid level k" | k=1의 1x1부터 k=K의 (H/p)x(W/p)까지 이어지는 K개 해상도 level 중 하나입니다. |
| Parallel-within-scale | "Scale마다 forward 한 번" | Scale k의 모든 token은 autoregressive가 아니라 transformer pass 한 번으로 예측됩니다. |
| Causal-across-scales | "Scale-ordered attention" | Scale k의 token은 scale 1..k 전체에 attend할 수 있지만 scale k+1..K에는 attend할 수 없습니다. |
| Residual VQ | "Additive tokenization" | 각 scale의 token이 lower scale이 남긴 residual을 encode하고 decoder는 모든 scale embedding을 합산합니다. |
| VAR scaling law | "Image GPT scaling" | Language model의 perplexity처럼 FID가 compute에 대해 예측 가능한 power law를 따릅니다. |
| HART | "Hybrid VAR + text" | MaskGIT-style iterative decoding과 VAR의 scale structure를 결합한 text-conditional VAR variant입니다. |
| Scale position embedding | "(scale, row, col) triple" | Positional encoding이 scale index와 scale 내부 spatial coordinate를 모두 담습니다. |

## 더 읽을거리

- [Tian et al., 2024 - "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) - VAR paper, canonical reference.
- [Peebles and Xie, 2022 - "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) - DiT, diffusion comparison baseline.
- [Esser et al., 2021 - "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) - VQGAN, VAR의 multi-scale tokenizer가 확장하는 tokenizer family.
- [van den Oord et al., 2017 - "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) - VQ-VAE, discrete image tokenization의 foundation.
- [Tang et al., 2024 - "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) - text-conditional VAR.
