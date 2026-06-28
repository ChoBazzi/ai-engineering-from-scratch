# Chameleon과 초기 융합 토큰 전용 멀티모달 모델

> 지금까지 본 모든 VLM은 이미지와 텍스트를 분리해 둔다. 시각 토큰은 vision encoder에서 나오고, projector로 흘러간 뒤, LLM 내부에서 텍스트를 만난다. 시각 vocabulary와 텍스트 vocabulary는 절대 겹치지 않는다. Chameleon(Meta, 2024년 5월)은 이렇게 물었다. 둘이 겹치면 어떨까? 이미지를 공유 vocabulary의 discrete token 시퀀스로 바꾸는 VQ-VAE를 학습한다. 이제 모든 멀티모달 문서는 하나의 시퀀스다. 텍스트 토큰과 이미지 토큰이 interleave되고, 단일 autoregressive loss를 쓴다. 부수 효과도 있다. 모델이 한 번의 inference 호출에서 텍스트 토큰과 이미지 토큰을 번갈아 내는 mixed-modality output을 생성할 수 있다. 이 lesson은 early-fusion 논지를 읽고 장난감 버전을 end-to-end로 만든다.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 minutes

## 학습 목표

- shared vocabulary와 단일 loss가 모델이 할 수 있는 일을 왜 바꾸는지 설명한다.
- VQ-VAE가 이미지를 transformer의 next-token objective와 호환되는 discrete sequence로 tokenization하는 방식을 설명한다.
- Chameleon의 학습 안정화 기법인 QK-Norm, dropout 배치, LayerNorm 순서를 말할 수 있다.
- Chameleon과 BLIP-2의 Q-Former 접근을 비교하고, 각각이 언제 적절한 선택인지 설명한다.

## 문제

Adapter 기반 VLM(LLaVA, BLIP-2, Qwen-VL)은 텍스트와 이미지를 서로 다른 것으로 취급한다. 텍스트 토큰은 `embed(text_token)`을 지나고, 이미지는 `visual_encoder(image) → projector → ... pseudo_tokens`를 지난다. 모델에는 중간 지점에서 합쳐지는 두 입력 경로가 있다.

세 가지 결과가 생긴다.

1. LLM은 이미지를 소비할 수만 있고, 내보낼 수는 없다. 출력은 텍스트뿐이다.
2. 글처럼 문단과 이미지가 번갈아 나오는 mixed-modality document는 다루기 어색하다. 모델 밖에서 멀티모달 입력을 파싱하거나 generation을 체인으로 이어야 한다.
3. Distributional mismatch가 생긴다. 시각 토큰과 텍스트 토큰은 hidden space의 서로 다른 영역에 있어 미묘한 alignment 문제가 생긴다.

Chameleon은 전제를 거부한다. 이미지는 공유 vocabulary에서 온 discrete token 시퀀스일 뿐이다. 모델을 interleaved document, 하나의 loss, 하나의 autoregressive decoder로 학습하면 mixed-modality generation이 자연스럽게 열린다.

## 개념

### Image tokenizer로서의 VQ-VAE

Tokenizer는 vector-quantized variational autoencoder다. 아키텍처는 다음과 같다.

- Encoder: 이미지를 공간 feature map으로 매핑하는 CNN + ViT. 예를 들어 dim 256의 32x32 feature다.
- Codebook: 학습되는 K개 vector vocabulary(Chameleon은 8192개 사용)이며, 역시 dim 256이다.
- Quantization: 각 spatial feature마다 L2 distance 기준으로 가장 가까운 codebook entry를 찾는다. 연속 feature를 정수 index로 대체한다.
- Decoder: quantized feature를 다시 pixel로 되돌리는 CNN이다.

학습은 VAE reconstruction loss + commitment loss + codebook loss로 한다. Codebook index가 이미지용 discrete alphabet이 된다.

Chameleon에서는 이미지 하나가 8192개 vocabulary에서 뽑은 32*32 = 1024개 token이 된다. 이를 텍스트 토큰(예: LLM의 BPE vocabulary 32000개)과 concatenate한다. 최종 vocabulary는 40192개다. Transformer는 하나의 시퀀스와 하나의 loss만 본다.

### 공유 vocabulary

Chameleon의 vocabulary는 텍스트 토큰, 이미지 토큰, modality separator를 결합한다. 각 토큰에는 단일 ID가 있다. 입력 embedding layer는 모든 ID를 D차원 hidden vector로 매핑한다. 출력 projection은 hidden을 다시 vocab logit으로 매핑한다. Softmax는 modality와 무관하게 다음 토큰을 고른다.

Separator가 중요하다. `<image>`와 `</image>` 태그는 이미지 토큰 시퀀스를 감싼다. Generation 시점에 모델이 `<image>`를 내보내면, downstream software는 이어지는 1024개 토큰이 pixel rendering을 위해 decoder에 보낼 VQ index임을 안다.

### 혼합 모달리티 생성

Inference는 공유 vocabulary 위의 next-token prediction이다. 예시 prompt: "Draw a cat and describe it." Chameleon은 다음처럼 낸다.

```text
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

모델은 순서를 자율적으로 고른다. 이미지 다음 텍스트, 텍스트 다음 이미지, 또는 interleave된 형태가 될 수 있다. 같은 decoder, 같은 loss다.

Generation이 텍스트 전용인 adapter VLM과 비교하라. Chameleon은 모델 출력 modality라는 질문을 다시 연다.

### 학습 안정성: QK-Norm, dropout, LayerNorm 순서

Early-fusion 학습은 scale이 커지면 불안정하다. Chameleon 논문은 세 가지 기법을 기록한다.

- QK-Norm. Attention 안에서 dot product 전에 query와 key projection에 LayerNorm을 적용한다. 깊이가 커질 때 logit magnitude가 폭발하는 것을 막는다. 2024년 이후 여러 대형 모델이 사용한다.
- Dropout 배치. Attention과 MLP 뒤뿐 아니라 모든 residual-add 뒤에 dropout을 둔다. 이미지 토큰의 gradient가 지배할 수 있을 때 더 강한 regularization이 필요하다.
- LayerNorm 순서. Residual branch에는 표준적인 Pre-LN을 쓰고, 마지막 block의 skip connection에 extra LN을 둔다. 최종 layer의 gradient flow를 안정화한다.

이 기법들이 없으면 34B parameter Chameleon 학습은 여러 checkpoint에서 diverge했다. 적용하면 수렴한다. 학습 recipe는 architecture만큼이나 중요한 기여다.

### Tokenizer의 reconstruction ceiling

VQ-VAE는 lossy하다. 8192개 codebook entry와 512x512 이미지당 1024개 token에서는 reconstruction PSNR 상한이 약 26-28 dB에 걸린다. 알아볼 수 있는 image generation에는 충분하지만 continuous-space diffusion(Stable Diffusion 3는 32+ dB)에 비해 눈에 띄게 떨어진다.

Tokenizer가 병목이다. 더 나은 tokenizer(MAGVIT-v2, IBQ, SBER-MoVQGAN)는 ceiling을 높인다. Emu3(Lesson 12.12)는 tokenizer만 더 좋게 만들어 SDXL 수준의 generation을 달성한다.

### Chameleon vs BLIP-2 / LLaVA

Chameleon(early fusion, shared vocab):
- 하나의 loss, 하나의 decoder.
- Mixed-modality output을 생성한다.
- Tokenizer가 품질 ceiling이다.
- 비싸다. Inference path에서 생성 이미지마다 VQ-VAE decoder가 필요하다.

BLIP-2 / LLaVA(late fusion, separate towers):
- Vision in, text out만 가능하다.
- Pretrained LLM을 재사용한다.
- Understanding에는 tokenizer 병목이 없다.
- 싸다. 단일 forward pass면 된다.

Task로 고른다. Image generation이 필요하면 Chameleon 계열이다. Understanding만 필요하면 adapter-VLM이 더 단순하고 pretrained compute를 더 많이 재사용한다.

### Fuyu와 AnyGPT

Fuyu(Adept, 2023)는 관련 접근이다. 별도 vision encoder를 아예 건너뛰고 raw image patch를 토큰처럼 LLM의 input projection에 넣으며 tokenizer를 쓰지 않는다. Chameleon보다 단순하지만 shared-vocab output generation은 잃는다.

AnyGPT(Zhan et al., 2024)는 Chameleon을 텍스트, 이미지, 음성, 음악 네 modality로 확장한다. 각 modality에 같은 VQ-VAE 기법을 쓰고 transformer를 공유한다. Any-to-any generation이다. Lesson 12.16에서 더 다룬다.

## 활용하기

`code/main.py`는 장난감 end-to-end early-fusion 모델을 만든다.

- 8x8 patch를 codebook index(K=16)로 매핑하는 작은 VQ-VAE식 quantizer.
- (text id 0..31) + (image id 32..47) + (separator 48, 49)로 된 shared vocabulary.
- 합성 caption + image-token sequence로 학습되는 장난감 autoregressive decoder(bigram table).
- Prompt가 주어졌을 때 text + image token을 번갈아 내는 sampling loop.

코드는 transformer를 의도적으로 아주 작게(bigram) 유지해 signal flow를 end-to-end로 추적할 수 있게 한다.

## 산출물

이 lesson은 `outputs/skill-tokenizer-vs-adapter-picker.md`를 만든다. Product spec(understand only vs understand + generate, required image quality, cost budget)이 주어지면 Chameleon 계열(early fusion)과 LLaVA 계열(late fusion) 중 하나를 고르고, quantitative rule of thumb으로 정당화한다.

## 연습 문제

1. Chameleon은 K=8192 codebook entry와 512x512 이미지당 1024개 token을 쓴다. 24-bit RGB 이미지 대비 compression ratio를 추정하라. Lossy한가? 얼마나 lossy한가?

2. 같은 VQ-VAE density에서 4K 이미지(3840x2160)는 이미지 토큰을 몇 개 생성하는가? Chameleon식 모델이 한 번의 inference call로 4K 이미지를 생성할 수 있는가? 먼저 깨지는 것은 context, tokenizer quality, KV cache 중 무엇인가?

3. Pure Python으로 QK-Norm을 구현하라. 64차원 query와 key가 주어졌을 때 LayerNorm 전후의 dot product를 보여라. 깊은 모델에서 magnitude control이 왜 중요한가?

4. Chameleon Section 2.3의 training stability 내용을 읽어라. QK-Norm 없이 34B에서 논문이 관찰한 정확한 failure mode를 설명하라. "norm explosion" signature는 무엇이었는가?

5. Text-only prompt가 주어졌을 때 mixed-modality response를 내도록 장난감 decoder를 확장하라. Training-data distribution이 60% text-first / 40% image-first일 때 모델이 image-first와 text-first를 얼마나 자주 고르는지 측정하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | 이미지가 첫 단계부터 transformer vocabulary를 공유하는 discrete token으로 변환되는 방식 |
| VQ-VAE | "Image tokenizer" | 이미지를 transformer가 예측할 수 있는 integer index로 매핑하는 CNN + ViT + codebook |
| Shared vocabulary | "One dictionary" | 텍스트 + 이미지 + modality separator를 모두 포괄하는 단일 token ID 공간 |
| QK-Norm | "Attention stabilizer" | Query와 key의 dot product 전에 적용하는 LayerNorm이며, norm blowup을 막는다 |
| Mixed-modality generation | "Text + image output" | 한 pass에서 interleaved text token과 image token을 자율적으로 생성하는 inference |
| Codebook size | "K entries" | VQ-VAE가 quantize할 수 있는 discrete vector 수이며, compression과 fidelity를 trade off한다 |
| Tokenizer ceiling | "Reconstruction limit" | VQ token을 decoding해 달성할 수 있는 최고 PSNR이며, 모델의 이미지 품질을 제한한다 |

## 더 읽을거리

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
