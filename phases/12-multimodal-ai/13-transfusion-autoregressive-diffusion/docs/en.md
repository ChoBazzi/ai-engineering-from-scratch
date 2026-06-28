# Transfusion: 하나의 Transformer 안의 자기회귀 텍스트와 확산 이미지

> Chameleon과 Emu3는 discrete token에 모든 것을 건다. 작동은 하지만 quantization bottleneck이 눈에 보인다. 이미지 품질은 continuous-space diffusion model보다 낮은 곳에서 plateau에 도달한다. Transfusion(Meta, Zhou et al., 2024년 8월)은 반대 베팅을 한다. 이미지는 continuous로 유지하고, VQ-VAE를 완전히 제거하며, 하나의 transformer를 두 loss로 학습한다. 텍스트 토큰에는 next-token-prediction을 적용한다. 이미지 patch에는 flow-matching / diffusion loss를 적용한다. 두 objective는 같은 weight를 optimize한다. Stable Diffusion 3의 기반 architecture(MMDiT)는 가까운 친척이다. 이 lesson은 Transfusion의 논지를 읽고, 장난감 two-loss trainer를 만들며, 하나의 transformer가 두 작업을 모두 하게 만드는 attention mask를 추적한다.

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 minutes

## 학습 목표

- 하나의 backbone 위에서 두 loss(text token의 NTP, image patch의 diffusion MSE)를 실행하는 transformer를 연결한다.
- Image patch 사이의 bidirectional attention과 text token 위의 causal attention이 왜 올바른 mask 선택인지 설명한다.
- Transfusion식(continuous image, diffusion loss)과 Chameleon식(discrete image, NTP)을 compute, quality, code complexity 관점에서 비교한다.
- MMDiT의 기여를 말할 수 있다. 각 block의 modality-specific weight와 residual stream의 joint attention이다.

## 문제

Discrete image token 대 continuous image token 논쟁은 LLM보다 오래되었다. Continuous representation(raw pixel, VAE latent)은 detail을 보존한다. Discrete token(VQ index)은 transformer의 native vocabulary에 맞지만 quantization 단계에서 detail을 잃는다.

Chameleon / Emu3는 discrete로 갔다. 하나의 loss와 하나의 architecture를 얻었지만 image fidelity는 tokenizer 품질에 의해 제한된다.

Diffusion model은 continuous로 갔다. 뛰어난 이미지 품질을 얻었지만 LLM과 별도 모델이고, noise-schedule engineering이 복잡하며, text generation과 깔끔하게 통합되지 않는다.

Transfusion은 묻는다. 둘 다 가질 수 있을까? 이미지는 continuous로 유지하고, 여전히 하나의 모델을 학습하며, 두 loss를 하나의 gradient step으로 이어 붙일 수 있을까?

## 개념

### 두 손실 아키텍처

단일 decoder-only transformer가 다음을 포함하는 시퀀스를 처리한다.

- Text token(BPE vocab에서 온 discrete token).
- Image patch(continuous, 16x16 pixel block을 linear embedding으로 hidden dim에 project한다. ViT encoder input과 같다).
- Continuous patch가 어디에 있는지 표시하는 `<image>`와 `</image>` 태그.

Forward pass는 한 번 실행된다. Loss는 token별로 두 head 중 하나를 고른다.

- Text token: vocab-logits head 위의 표준 cross-entropy.
- Image patch: continuous patch 위의 diffusion loss. 각 patch에 더해진 noise를 예측한다.

Gradient는 공유 transformer body를 통과한다. 두 loss가 공유 weight를 동시에 개선한다.

### 어텐션 마스크: 인과적 텍스트와 양방향 이미지

Text token은 causal이어야 한다. Text token이 future text에 attend하면 teacher forcing이 깨진다. 반면 image patch는 하나의 snapshot을 나타낸다. 같은 image block 안에서는 서로 bidirectionally attend해야 한다.

Mask는 다음과 같다.

```text
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

학습과 inference에서 block-triangular mask로 구현한다.

### Transformer 내부의 diffusion loss

Diffusion loss는 표준적이다. Image patch에 noise를 더하고 모델이 그 noise(또는 동등하게 clean patch)를 예측하게 한다. Transfusion 버전은 flow matching을 쓴다. Noisy 상태에서 clean 상태로 가는 velocity field를 예측한다.

학습 중에는 다음을 수행한다.
1. 각 image patch x0마다 random timestep t를 sample한다.
2. Noise ε를 sample하고 xt = (1-t) * x0 + t * ε를 계산한다(flow matching을 위한 linear interpolation).
3. Transformer가 v_theta(xt, t)를 예측한다. Loss = MSE(v_theta(xt, t), ε - x0).
4. 같은 sequence에서 나온 text NTP loss와 함께 backprop한다.

Inference에서 generation은 다음과 같다.
- Text token: 표준 autoregressive sampling.
- Image patch: prior text token에 conditioned된 diffusion sampling loop(보통 10-30 step).

### MMDiT: Stable Diffusion 3의 variant

Stable Diffusion 3(Esser et al., 2024년 3월)는 Transfusion과 거의 같은 시기에 MMDiT(Multimodal Diffusion Transformer)를 선보였다. 두 architecture는 형제에 가깝다.

MMDiT의 핵심 차이는 다음과 같다.

- Block별 modality-specific weight. 각 transformer block에는 text token과 image patch용으로 별도의 Q, K, V, MLP weight가 있다. Attention은 joint(cross-modality)이고, 그 밖의 모든 것은 modality-specific이다.
- Rectified flow training. DDPM보다 sampling이 알려져 있고 수학이 단순한 특정 flow-matching variant다.
- Scale. MMDiT는 SD3(2B 및 8B parameter variant)의 backbone이다. Transfusion 논문은 7B까지 scale한다.

둘 다 같은 핵심 아이디어로 수렴한다. 하나의 transformer가 text에는 NTP를, continuous image representation에는 diffusion을 실행한다.

### 왜 Chameleon식보다 나은가

Image generation에서 continuous-diffusion과 discrete-NTP 사이의 quality gap은 측정 가능하다. Transfusion 논문은 다음을 보고한다.

- 7B parameter에서 같은 크기의 Chameleon식 model보다 FID가 3-5점 좋다.
- Tokenizer 학습이 필요 없다. Image encoder는 더 단순하다(hidden으로 가는 linear projection, ViT의 input layer와 동일).
- Autoregressive image token과 달리 inference에서 image patch denoising을 parallelize할 수 있다.

단점도 있다. Transfusion은 dual-loss model이라 training dynamic이 더 까다롭다. Loss weight tuning이 필요하다. NTP와 diffusion 사이의 schedule mismatch 때문에 한 head가 지배할 수 있다.

### Downstream에 있는 것들

Janus-Pro(Lesson 12.15)는 understanding과 generation을 위한 vision encoder를 분리해 Transfusion의 아이디어를 정제한다. 하나에는 SigLIP, 다른 하나에는 VQ를 쓰며 transformer body는 공유한다. Show-o(Lesson 12.14)는 diffusion을 discrete-diffusion(masked prediction)으로 바꾼다. Unified-generation 계열은 Transfusion 이후 빠르게 갈라진다.

이미지를 내보내는 2026년 production VLM, 예를 들어 Gemini 3 Pro, GPT-5, Claude Opus 4.7의 image generation path는 거의 확실히 이 계열의 descendant를 일부 사용한다. 세부 사항은 proprietary다.

## 활용하기

`code/main.py`는 작은 MNIST식 문제 위의 장난감 Transfusion을 만든다.

- Text caption은 digit(0-9)을 설명하는 짧은 integer sequence다.
- 이미지는 byte로 된 4x4 grid다.
- Shared-weight linear projection 쌍이 transformer stand-in 역할을 한다. Text에는 NTP loss, noisy patch에는 MSE loss를 쓴다.
- Training loop는 두 loss를 번갈아 적용하고, attention mask는 명시적이다.
- Generation은 한 forward pass 안에서 text caption과 4x4 image를 만든다.

Transformer는 장난감이다. Two-loss plumbing, attention mask construction, inference loop가 실제 artifact다.

## 산출물

이 lesson은 `outputs/skill-two-loss-trainer-designer.md`를 만든다. 새 multimodal training task(text + image, text + audio, text + video)가 주어지면 two-loss schedule(loss weight, mask shape, shared vs modality-specific block)을 설계하고 implementation risk를 표시한다.

## 연습 문제

1. Transfusion식 모델이 70% text token과 30% image patch로 학습한다. Image diffusion loss magnitude가 text NTP loss의 약 10배다. 어떤 loss weight가 둘을 균형 잡는가?

2. `[T, T, <image>, P, P, P, P, </image>, T]` 시퀀스에 대한 block-triangular mask를 구현하라. 각 entry를 0 또는 1로 표시하라.

3. MMDiT에는 modality-specific QKV weight가 있다. Transfusion의 fully-shared transformer 대비 parameter count overhead가 얼마나 추가되는가? 7B parameter에서는 그만한 가치가 있는가?

4. Generation: text prompt가 주어졌을 때 모델은 50 token 동안 NTP를 실행하고, `<image>`에 도달한 뒤 256 patch에 대해 20 denoise step으로 diffusion을 실행한다. 총 forward pass 수는 얼마인가?

5. SD3 paper Section 3을 읽어라. Rectified flow를 설명하고, 왜 DDPM보다 적은 inference step으로 수렴하는지 설명하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | 단일 transformer가 같은 gradient step에서 text token의 cross-entropy와 continuous image patch의 MSE를 함께 optimize하는 방식 |
| Flow matching | "Rectified flow" | Noise에서 clean data로 가는 velocity field를 예측하는 diffusion variant이며, DDPM보다 수학이 단순하다 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3의 architecture. Joint attention, modality-specific MLP와 norm을 쓴다 |
| Block-triangular mask | "Causal text + bidirectional image" | Text token 사이에서는 causal이고 image region 내부에서는 bidirectional인 attention mask |
| Continuous image representation | "No VQ" | Integer codebook index가 아니라 real-valued vector인 image patch |
| Velocity prediction | "v-parameterization" | Network output이 noise 자체가 아니라 noise와 data 사이의 velocity field인 방식 |

## 더 읽을거리

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
