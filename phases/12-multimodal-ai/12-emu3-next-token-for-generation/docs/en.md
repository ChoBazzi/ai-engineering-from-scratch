# Emu3: 이미지와 비디오 생성을 위한 다음 토큰 예측

> BAAI의 Emu3(Wang et al., 2024년 9월)는 diffusion 대 autoregressive 논쟁을 끝냈어야 할 2024년 결과다. 텍스트 + VQ image token + 3D VQ video token의 unified vocabulary 전반에서 next-token-prediction objective만으로 학습한 단일 Llama식 decoder-only transformer가 image generation에서는 SDXL을, perception에서는 LLaVA-1.6을 이긴다. CLIP loss가 없다. Diffusion schedule도 없다. 품질을 위해 inference에서 classifier-free guidance를 쓰지만, 핵심 학습 objective는 teacher forcing을 쓰는 next-token prediction이다. Nature에 게재되었다. 이 lesson은 더 나은 tokenizer와 scale만 있으면 충분하다는 Emu3의 논지를 읽고 diffusion 접근과 대비한다.

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 minutes

## 학습 목표

- Image quality에는 diffusion이 필요하다는 오래된 가정에도 불구하고 Emu3의 single-loss next-token objective가 왜 작동하는지 설명한다.
- 3D video tokenizer를 설명한다. Spatiotemporal VQ codebook이 어떤 모습인지, patch가 왜 시간축까지 span하는지 설명한다.
- Emu3와 Stable Diffusion XL을 training compute, inference cost, quality ceiling 관점에서 비교한다.
- 같은 Emu3 모델이 수행하는 세 역할인 Emu3-Gen(image gen), Emu3-Chat(perception), Emu3-Stage2(video gen)를 말할 수 있다.

## 문제

2024년까지의 통념은 image generation에는 diffusion이 필요하다는 것이었다. 논리는 이랬다. Discrete image token은 detail을 reconstruct하기에 너무 많은 정보를 잃고, autoregressive sampling은 수천 token에 걸쳐 error를 누적한다. Stable Diffusion, DALL-E 3, Imagen, Midjourney는 모두 어떤 형태로든 diffusion을 쓴다. Chameleon(Lesson 12.11)은 작은 scale에서 이를 부분적으로 반박했지만, 품질에서 SDXL을 따라잡지는 못했다.

Emu3는 이 논리를 정면으로 공격했다. 주장: 더 나은 visual tokenizer + 충분한 scale + next-token loss = perception도 하는 같은 모델 안에서 diffusion을 이기는 image generation.

출간 당시 이 베팅은 논쟁적이었다. 2년이 지난 지금, open-source unified-generation 계열(Emu3, Show-o, Janus-Pro, Transfusion)은 연구의 기본 경로가 되었고 production frontier model도 어떤 변형을 쓰는 것으로 보인다.

## 개념

### Emu3 토크나이저

핵심 재료는 visual tokenizer다. Emu3는 token당 8x8 resolution reduction을 갖는 custom IBQ 계열 tokenizer(Inverse Bottleneck Quantizer, SBER-MoVQGAN 계열)를 학습한다. 512x512 이미지는 codebook size 32768에서 64x64 = 4096 token이 된다.

이는 Chameleon의 512x512당 1024 token(K=8192)보다 크지만 token당 비용은 더 싸다(더 작은 codebook lookup, 더 단순한 codec). 핵심 metric은 reconstruction PSNR 30.5 dB이며, Stable Diffusion의 continuous latent space 32 dB와 경쟁할 만하다.

Video의 경우 3D VQ tokenizer는 spatiotemporal patch(4x4x4 pixel)를 하나의 integer로 encode한다. 8 FPS의 4초 clip은 32 frame이다. 256x256에서 spatial 4x, temporal 4x reduction을 쓰면 token count는 (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 token이다.

Tokenizer 품질이 ceiling이다. Emu3의 기여 중 하나는 "아주 좋은 tokenizer를 학습했다"는 점이다.

### 단일 손실 학습

Emu3는 하나의 objective를 쓴다. 텍스트 토큰, 2D image token, 3D video token 전반의 shared vocabulary에서 next-token prediction을 한다. 학습 중 contribution balance를 위해 modality별 factor로 weight를 곱하지만, loss function 자체는 동일하다.

학습 mix는 다음을 포함한다.
- Image gen: `<text caption> <image> image_tokens </image>`
- Image perception: `<image> image_tokens </image> <question> text_tokens`
- Video gen: `<text caption> <video> video_tokens </video>`
- Video perception: 유사한 형식.
- Text only: 표준 NTP.

모델은 data distribution에서 언제 image token을 내고 언제 text token을 낼지 배운다. Generation은 모델이 `<image>` 태그 뒤에서 image token을 예측하면서 emergent하게 생긴다.

### Classifier-free guidance와 temperature

Autoregressive image generation은 inference에서 classifier-free guidance(CFG)를 쓰면 훨씬 좋아진다. Emu3도 이를 쓴다. 한 번은 full caption으로, 한 번은 empty caption으로 두 번 generate한 뒤 guidance weight(보통 3.0-7.0)로 logit을 mix한다. Diffusion에서 쓰던 같은 CFG 기법을 autoregressive setting으로 가져온 것이다.

Temperature도 중요하다. 너무 높으면 artifact가 생기고, 너무 낮으면 mode collapse가 난다. Emu3의 권장 temperature는 perception에서는 1.0, image generation에서는 0.8이다.

### 세 역할, 하나의 모델

Emu3는 기능적으로 구분된 세 API로 제공되지만 underlying weight set은 하나다.

- Emu3-Gen. Image generation. Input text, output image token.
- Emu3-Chat. VQA와 captioning. Input image(token), output text.
- Emu3-Stage2. Video generation과 video VQA. Input text 또는 video, output text 또는 video.

Task-specific head가 없다. Prompt template만 다르다. 같은 checkpoint다.

### 벤치마크

Emu3 논문(2024년 9월) 기준:

- Image generation: MJHQ-30K FID(5.4 vs 5.6)에서 SDXL을 이기고, GenEval overall(0.54 vs 0.55, 통계적으로 동률) 및 Deep-Eval composite에서 비슷하다.
- Image perception: VQAv2에서 LLaVA-1.6을 이기고(75.1 vs 72.4), MMMU에서는 대략 비슷하다.
- Video generation: 4초 clip 품질에서 Sora 시대의 공개 benchmark model과 경쟁적인 FVD를 보인다.

숫자가 항상 이기는 것은 아니다. Emu3는 어떤 곳에서 1점을 얻고 다른 곳에서 1점을 잃는다. 그래도 "next-token prediction is all you need"라는 주장은 modality 전반에서 방어 가능하다.

### 계산 비용

Emu3는 7B parameter model을 약 300B multimodal token으로 학습했다. GPU-hour는 Llama-2-7B pretraining과 대략 비슷하다(A100급 silicon 기준 2k-4k GPU-year). Stable Diffusion 3 같은 diffusion model도 비슷한 budget으로 학습하지만 별도 text encoder와 더 복잡한 pipeline이 필요하다.

Inference에서 Emu3는 이미지당 SDXL보다 느리다. 30 tok/s로 4096 image token을 생성하면 512x512 이미지 하나에 약 2분이 걸린다. SDXL의 2-5초와 대비된다. Speculative decoding과 KV-cache optimization이 gap을 줄이지만 없애지는 못한다. Autoregressive image gen은 compute-heavy하다. 이것이 지속적인 trade-off다.

### 왜 중요한가

Emu3의 깊은 기여는 개념적이다. Next-token prediction이 image generation에서 diffusion과 맞먹도록 scale된다면 unified-model 경로(하나의 loss, 하나의 backbone, 임의 modality)는 가능하다. 미래 모델은 별도 text encoder, 별도 diffusion scheduler, 별도 VAE가 필요하지 않다. Modality당 하나의 tokenizer와 하나의 transformer, 그리고 scale이면 된다.

Show-o, Janus-Pro, InternVL-U는 모두 이 thesis 위에 쌓거나 이를 도전한다. 중국 연구소(BAAI, DeepSeek)는 2025년까지 미국 연구소보다 이 방향의 발표를 더 공격적으로 해왔다.

## 활용하기

`code/main.py`는 두 개의 장난감 조각을 만든다.

- 2D vs 3D VQ tokenizer count calculator: (resolution, patch, clip_length, FPS)가 주어지면 image와 video의 token count를 계산한다.
- Temperature를 쓰는 classifier-free guidance 기반 autoregressive image-token sampler.

CFG 구현은 Emu3 recipe와 일치한다. Conditional logit과 unconditional logit을 guidance weight로 mix한다.

## 산출물

이 lesson은 `outputs/skill-token-gen-cost-analyzer.md`를 만든다. Generation product spec(image 또는 video, target resolution, quality tier, latency budget)이 주어지면 token count와 inference cost를 계산하고 Emu3 계열과 diffusion 중 하나를 고른다.

## 연습 문제

1. Emu3는 8x8 reduction에서 512x512 이미지당 4096 token을 만든다. 1024x1024와 2048x2048에 해당하는 값을 계산하라. Inference latency에는 무슨 일이 생기는가?

2. Emu3 Section 3.3의 video tokenizer를 읽어라. 3D VQ patch shape를 설명하고 왜 8x8x1이 아니라 4x4x4인지 설명하라.

3. Classifier-free guidance weight 5.0 vs 3.0은 어떤 시각적 효과를 내는가? `code/main.py`의 수식을 추적하라.

4. 300B token의 Emu3-7B training FLOPs를 계산하고 Stable Diffusion 3와 비교하라. 어느 쪽이 학습 비용이 더 컸는가?

5. Emu3는 FID에서 SDXL을 이기지만 specialized VLM 대비 VQAv2에서는 그렇지 않다. Unified-loss 접근이 benchmark별로 specialist와 다른 강점을 보이는 이유를 설명하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | token[0..i]가 주어졌을 때 token[i+1]을 예측하는 표준 autoregressive loss이며, tokenized된 모든 modality에 작동한다 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | Chameleon보다 큰 codebook(32768+)과 더 나은 reconstruction을 갖는 VQ-VAE 계열 |
| 3D VQ | "Spatiotemporal quantizer" | (time, row, col)로 indexing되는 codebook이며, 하나의 token이 4x4x4 pixel cube를 덮는다 |
| Classifier-free guidance | "CFG" | Conditional logit과 unconditional logit을 gamma weight로 mix해 inference에서 이미지 품질을 높이는 기법 |
| Unified vocabulary | "Shared tokens" | 텍스트 + 이미지 + 비디오가 모두 같은 integer space에서 나오며, 모델은 다음에 올 modality를 예측한다 |
| MJHQ-30K | "Image gen benchmark" | 30k prompt를 갖는 Midjourney-quality benchmark이며, Emu3는 여기서 FID를 보고한다 |

## 더 읽을거리

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
