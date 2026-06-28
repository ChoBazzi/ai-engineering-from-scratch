# CLIP과 대조적 비전-언어 사전학습

> OpenAI의 CLIP(2021)은 다음 5년을 움직일 만큼 큰 하나의 아이디어를 증명했습니다. Noisy web image-caption pair와 contrastive loss만으로 image encoder와 text encoder를 같은 vector space에 정렬하는 것입니다. Supervised label은 0개. Pair는 400M개. 그 결과 나온 embedding space는 zero-shot classification, image-text retrieval을 수행하고, 모든 2026년 VLM에 vision tower로 연결됩니다. SigLIP 2(2025)는 softmax를 sigmoid로 바꾸어 더 낮은 비용으로 CLIP을 넘어섰습니다. 이 lesson은 InfoNCE부터 sigmoid pairwise loss까지 수학을 따라가고 stdlib Python으로 training step을 구현합니다.

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 minutes

## 학습 목표

- Mutual information에서 InfoNCE loss를 유도하고 numerically-stable vectorized version을 구현합니다.
- Sigmoid pairwise loss(SigLIP)가 softmax가 요구하는 all-gather overhead 없이 batch 32768+까지 scale되는 이유를 설명합니다.
- Text template(`a photo of a {class}`)을 만들고 cosine similarity의 argmax를 취해 zero-shot ImageNet classification을 실행합니다.
- CLIP / SigLIP pretraining이 주는 네 가지 lever를 말합니다: batch size, temperature, prompt template, data quality.

## 문제

CLIP 이전의 vision은 supervised였습니다. Labeled dataset(ImageNet: 1.2M image, 1000 class)을 모아 CNN을 train하고 배포했습니다. Label은 비싸고, labeler가 합의할 수 있는 대상에 bias가 있으며, finetuning 없이는 새 task로 잘 transfer되지 않습니다.

Image-caption web에는 공짜로 얻을 수 있는 loosely-labeled pair가 10억 개 이상 있습니다. Golden retriever 사진과 alt text "my dog Max in the park"는 supervision signal을 담고 있습니다. Text가 image를 설명합니다. 질문은 이것을 유용한 training으로 바꿀 수 있느냐입니다.

CLIP의 답은 image-caption pair를 matching task로 다루는 것이었습니다. N개의 image와 N개의 caption batch가 주어지면, 각 image를 자기 caption과 N-1개의 distractor 사이에서 맞추도록 학습합니다. Supervision은 "이 둘은 함께 속하고, 나머지 N-1개는 그렇지 않다"입니다. Class label도 human annotation도 없습니다. Contrastive loss만 있습니다.

결과 embedding space는 CLIP이 train된 것보다 더 많은 일을 합니다. "a photo of a cat"이 명시적으로 cat이라고 label된 적 없는 고양이 사진 근처에 embed되기 때문에 ImageNet zero-shot이 작동합니다. 이것이 2026년의 모든 VLM을 낳은 bet입니다.

## 개념

### 듀얼 인코더

CLIP에는 두 tower가 있습니다.

- Image encoder `f`: ViT 또는 ResNet이며 image마다 D-dim vector를 출력합니다.
- Text encoder `g`: 작은 transformer이며 caption마다 D-dim vector를 출력합니다.

두 tower는 output을 unit length로 normalize합니다. 둘 다 unit-norm이므로 similarity는 `cos(f(x), g(y)) = f(x)^T g(y)`입니다.

N개의 (image, caption) pair batch에 대해 shape `(N, N)`의 similarity matrix `S`를 만듭니다.

```text
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

여기서 `tau`는 learned temperature입니다(CLIP은 0.07로 초기화하며 log-space에서 학습).

### InfoNCE 손실

CLIP은 row와 column에 대해 symmetric cross-entropy를 사용합니다.

```text
loss_i2t = CE(S, labels=identity)     # 각 image의 positive는 자기 caption
loss_t2i = CE(S^T, labels=identity)   # 각 caption의 positive는 자기 image
loss = (loss_i2t + loss_t2i) / 2
```

이것이 InfoNCE입니다. CE의 softmax는 각 image가 batch의 다른 모든 caption보다 자기 caption과 더 잘 match되도록 강제합니다. "Negative"는 다른 모든 batch item입니다. 더 큰 batch = 더 많은 negative = 더 강한 signal입니다. CLIP은 batch 32k에서 train되었습니다. Scale이 중요합니다.

### Temperature

`tau`는 softmax의 sharpness를 제어합니다. 낮은 tau -> sharp distribution, hard negative mining 효과. 높은 tau -> soft, 모든 sample이 기여합니다. CLIP은 `log(1/tau)`를 학습하고 collapse를 막기 위해 clip합니다. SigLIP 2는 초기 tau를 고정하고 learned bias를 대신 사용합니다.

### Sigmoid가 더 잘 scale되는 이유(SigLIP)

Softmax는 전체 similarity matrix가 동기화되어 있어야 합니다. Distributed training에서는 모든 embedding을 모든 replica에 all-gather한 뒤 softmax를 수행해야 합니다. 이는 communication이 world size에 대해 quadratic입니다.

SigLIP은 softmax를 element-wise sigmoid로 바꿉니다. 각 pair `(i, j)`에 대해 loss는 "이것이 matching pair인가?"라는 binary classification입니다. Diagonal은 positive class label이고, 나머지는 모두 negative입니다. Loss는 다음과 같습니다.

```text
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` if `i == j`, else 0. 각 pair의 loss는 독립적입니다. All-gather가 필요 없습니다. 각 GPU는 local block을 계산하고 합산합니다. SigLIP 2는 CLIP이 비례적으로 더 많은 communication을 요구할 batch 32k-512k까지 저렴하게 scale됩니다.

### 제로샷 분류

N개의 class name이 주어지면 각 class에 대해 text template을 만듭니다.

```text
"a photo of a {class}"
```

각 template을 text encoder로 embed합니다. Image는 image encoder로 embed합니다. Cosine similarity의 argmax가 predicted class입니다. Target class에 대한 training은 없습니다.

Prompt template은 중요합니다. CLIP original paper는 class마다 80개 template(plain, artistic, photo, painting 등)을 사용하고 embedding을 평균냈습니다. ImageNet에서 +3 point였습니다. 최신 사용에서는 보통 template 한두 개를 고릅니다.

### Linear probe와 finetuning

Zero-shot은 baseline입니다. Linear probe(target class에 대해 frozen CLIP feature 위에 linear layer 하나를 train)는 in-domain task에서 zero-shot을 이깁니다. Full finetuning은 in-domain에서 linear probe를 이기지만 zero-shot transfer를 해칠 수 있습니다. 세 가지 regime과 세 가지 trade-off가 있습니다.

### SigLIP 2: NaFlex와 dense feature

SigLIP 2(2025)는 다음을 추가합니다.
- NaFlex: 단일 model이 variable aspect ratio와 resolution을 처리합니다.
- Segmentation과 depth estimation을 위한 더 좋은 dense feature. VLM의 frozen backbone 사용을 목표로 합니다.
- Multilingual: CLIP이 English-only였던 것과 달리 100+ language에서 train되었습니다.
- CLIP이 400M에서 멈춘 반면 1B param scale까지 확장합니다.

2026년 open VLM에서 SigLIP 2 SO400m/14는 기본 vision tower입니다. CLIP은 특정 LAION-2B training distribution이 query pattern과 맞는 pure image-text retrieval에서 여전히 기본값입니다.

### ALIGN, BASIC, OpenCLIP, EVA-CLIP

ALIGN(Google, 2021): CLIP과 같은 아이디어, 1.8B pair scale, 90% noisy. Noisy data가 scale된다는 점을 증명했습니다. OpenCLIP(LAION): LAION-400M / 2B에서 CLIP을 open reproduction한 것으로, 여러 scale을 제공하는 go-to open checkpoint입니다. EVA-CLIP: masked image modeling에서 initialize하며 VLM을 위한 강한 backbone입니다. BASIC: Google의 CLIP+ALIGN hybrid입니다. 모두 같은 family이며 data와 tuning이 다릅니다.

### 제로샷 상한

CLIP-class model은 ImageNet zero-shot에서 약 76%(CLIP-G, OpenCLIP-G) 근처가 한계입니다. 그 이상은 훨씬 큰 data(SigLIP 2는 80%+) 또는 architecture change(supervised head, 더 많은 parameter)가 필요합니다. Benchmark는 포화되고 있습니다. 진짜 가치는 downstream VLM이 소비하는 embedding space입니다.

```figure
multimodal-fusion
```

## 사용하기

`code/main.py`는 다음을 구현합니다.

1. Numpy 없이 InfoNCE shape를 볼 수 있는 toy dual encoder(hash-based image feature, text char feature).
2. Pure Python의 InfoNCE loss(log-sum-exp를 통한 numerical stability).
3. 비교용 sigmoid pairwise loss.
4. Zero-shot classification routine: text prompt set과 cosine similarity를 계산하고 prediction의 argmax를 구합니다.

실행하고 loss curve를 관찰하세요. 절대값은 toy이지만, shape는 실제 CLIP trainer가 내보내는 것과 맞습니다.

## 결과물

이 lesson은 `outputs/skill-clip-zero-shot.md`를 생성합니다. Image set(path)과 target class list가 주어지면 CLIP template으로 text prompt를 만들고, 명시된 checkpoint(예: `openai/clip-vit-large-patch14`)로 양쪽을 embed한 뒤, similarity score와 함께 top-1 / top-5 prediction을 반환합니다. 이 skill은 prompt list에 없는 class에 대해 claim하지 않습니다.

## 연습문제

1. Pair 4개 batch에 대해 InfoNCE를 손으로 구현하세요. 4x4 similarity matrix를 만들고, softmax를 실행하고, diagonal을 골라 cross-entropy를 계산하세요. Python 구현이 이 hand calculation과 일치하는지 확인하세요.

2. SigLIP은 temperature 외에도 bias parameter `b`를 사용합니다: `S'[i,j] = S[i,j]/tau + b`. Batch에 큰 class imbalance(row마다 positive보다 negative가 훨씬 많음)가 있을 때 `b`는 어떤 역할을 하나요? SigLIP Section 3(arXiv:2303.15343)을 읽으세요.

3. Cat vs dog zero-shot classifier를 만드세요. 두 prompt template `a photo of a {class}`와 `a picture of a {class}`를 시도하세요. 100개 test image에서 accuracy를 측정하세요. Template ensemble이 single보다 나은가요?

4. 512-GPU run, batch 32k에서 softmax InfoNCE와 sigmoid pairwise의 communication cost를 계산하세요. 어느 쪽이 O(N)이고 어느 쪽이 O(N^2)로 scale되나요? SigLIP Section 4를 인용하세요.

5. OpenCLIP scaling-laws paper(arXiv:2212.07143, Cherti et al.)를 읽으세요. Figure에서 data scaling에 대한 결론을 재현하세요. Fixed model size에서 ImageNet zero-shot accuracy와 training data size 사이의 log-linear 관계는 무엇인가요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | Batch의 similarity matrix에 대한 cross-entropy입니다. 각 item의 positive는 paired item이고 negative는 나머지 전부입니다 |
| Sigmoid loss | "SigLIP loss" | Pair별 binary cross-entropy입니다. Softmax도 all-gather도 없어서 distributed training에서 저렴하게 scale됩니다 |
| Temperature | "tau" | Softmax/sigmoid 전에 logit을 scale하는 scalar이며 distribution sharpness를 제어합니다 |
| Zero-shot | "no-finetune classification" | Text prompt로 class embedding을 만들고 cosine similarity로 classify합니다. Target class에 대한 training은 없습니다 |
| Prompt template | "a photo of a ..." | Class name을 둘러싼 text scaffold이며 zero-shot accuracy에 1-5 point 영향을 줍니다 |
| Dual encoder | "Two-tower" | 하나의 image encoder + 하나의 text encoder이며 shared D-dim space에 output을 냅니다 |
| Hard negative | "Tough distractor" | Positive와 충분히 비슷해 model이 둘을 분리하려고 애써야 하는 negative입니다 |
| Linear probe | "Frozen + one layer" | Frozen feature 위에 linear classifier만 train합니다. Feature quality를 측정합니다 |
| NaFlex | "Native flexible resolution" | Resize 없이 어떤 aspect ratio와 resolution의 image도 ingest하는 SigLIP 2 capability |
| Temperature scaling | "log-parametrized tau" | CLIP은 gradient가 잘 동작하도록 `log(1/tau)`를 parameterize하고 near-zero tau로 collapse되는 것을 막기 위해 clip합니다 |

## 더 읽을거리

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP paper.
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP.
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — multilingual + NaFlex.
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — noisy web data로 scale.
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws.
