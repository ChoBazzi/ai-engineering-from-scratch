# Flamingo와 퓨샷 VLM을 위한 게이트형 교차 어텐션

> DeepMind의 Flamingo(2022)는 누구보다 먼저 두 가지를 보였습니다. 하나의 model이 image, video, text가 임의로 interleaved된 sequence를 처리할 수 있다는 점. 그리고 VLM이 in-context로 학습할 수 있다는 점입니다. 세 개의 example (image, caption) pair가 들어 있는 few-shot prompt를 주면, model은 gradient step 없이 새 image에 caption을 붙입니다. Mechanism은 gated cross-attention layer입니다. Frozen LLM의 기존 layer 사이에 삽입되며, 0에서 시작하는 learned tanh gate를 사용해 initialization에서 LLM의 text capability를 보존합니다. 이 lesson은 Flamingo의 Perceiver resampler와 gated cross-attention architecture를 따라갑니다. 이는 Gemini의 interleaved input과 Idefics2의 visual token의 조상입니다.

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 minutes

## 학습 목표

- Gated cross-attention이 tanh(gate) = 0을 통해 initialization에서 frozen LLM의 text capability를 보존하는 방식을 설명합니다.
- Perceiver resampler를 따라갑니다: N image patch -> cross-attention을 통한 K fixed "latent" query.
- Flamingo가 image placement를 존중하는 causal masking으로 interleaved image-text sequence를 처리하는 방식을 설명합니다.
- Few-shot multimodal prompt 구조(3개 image-caption example 이후 query image)를 재현합니다.

## 문제

BLIP-2는 32개 visual token을 frozen LLM의 input layer에 넣습니다. Prompt당 image 하나에는 작동합니다. 하지만 "여기 image A가 있으니 caption을 붙여라. 여기 image B가 있으니 caption을 붙여라. 이제 image C가 있으니 caption을 붙여라"처럼 text와 interleaved된 많은 image를 넣으려면 어떻게 해야 할까요? LLM의 self-attention은 image token과 text token을 하나의 stream에서 처리해야 하고, 어떤 position이 어떤 image에 attend할 수 있는지의 문제가 복잡해집니다.

Flamingo의 답은 LLM input stream을 전혀 바꾸지 않는 것입니다. 기존 LLM block 사이에 extra cross-attention layer를 삽입합니다. Text token은 늘 그랬듯 LLM의 causal self-attention을 통과합니다. 몇 개 LLM block마다 text token은 새 gated layer를 통해 image feature에도 cross-attend합니다. Gate(initialized to zero) 덕분에 step zero에서 새 layer는 no-op입니다. Model은 pretrained LLM과 정확히 똑같이 동작합니다. Training이 진행되면서 gate가 열리고 visual information이 흐르기 시작합니다.

Flamingo가 답한 두 번째 질문은 prompt마다 variable number of images(0, 1, 또는 여러 장)를 어떻게 처리하느냐입니다. Perceiver resampler는 patch 수가 몇 개든 받아 fixed number of visual latent token을 만드는 작은 cross-attention module입니다. LLM cross-attention layer는 prompt의 image 수와 관계없이 같은 shape를 봅니다.

## 개념

### 고정된 LLM

Flamingo는 frozen Chinchilla 70B LLM에서 시작합니다. 70B weight는 모두 건드리지 않습니다. 기존 text self-attention과 FFN은 정상적으로 동작합니다.

### Perceiver 리샘플러

Prompt의 각 image에 대해 ViT는 N개의 patch token을 만듭니다. Perceiver resampler에는 K개의 fixed learnable latent가 있습니다(Flamingo는 K=64 사용). 각 resampler block은 두 sub-step입니다.

1. Cross-attention: K latent가 N patch token에 attend합니다(latent에서 Q, patch에서 K/V).
2. Latent 내부의 self-attention + FFN.

6개 resampler block 이후 output은 ViT가 몇 개 patch를 만들었는지와 관계없이 dim 1024의 K=64 visual token입니다. 224x224 image(196 patch)와 480x480 image(900 patch)가 모두 64 resampler token으로 나옵니다.

Video의 경우 resampler가 temporally 적용됩니다. 각 frame의 patch가 64 latent를 만들고, temporal positional encoding이 t=0과 t=N을 구분하게 합니다. 전체 video는 T * 64 visual token이 됩니다.

### 게이트형 교차 어텐션

Frozen LLM의 M개 layer마다(Flamingo는 M=4 사용) 새 gated cross-attention block을 삽입합니다.

```text
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha`는 0으로 initialized되는 learnable scalar입니다.
- `tanh(0) = 0`이므로 init에서 gated branch는 0을 기여합니다.
- `alpha`가 0에서 멀어질수록 cross-attention contribution이 부드럽게 커집니다.
- Residual connection 덕분에 완전히 열린 gate도 LLM의 text representation을 덮어쓰지 않습니다. 그 위에 visual information을 더할 뿐입니다.

이것이 Flamingo의 가장 중요한 design choice입니다. Visual conditioning은 additive이고 gated이며 initialization에서 0입니다. Step 0의 Flamingo는 text-only input에서 완벽한 Chinchilla 70B입니다.

### Interleaved input을 위한 masked cross-attention

"<image A> caption A <image B> caption B <image C> ?" 같은 prompt에서 각 text token은 sequence에서 자기 앞에 나온 image만 보아야 합니다. Cross-attention mask는 이를 강제합니다. Position `t`의 text token은 image index `i < i_t`인 image resampler token에만 attend합니다. 여기서 `i_t`는 position `t` 앞의 가장 최근 image입니다. "가장 최근 preceding image만 본다"와 "모든 preceding image를 본다"는 모두 유효한 선택입니다. Flamingo는 전자를 선택했습니다.

### 인컨텍스트 퓨샷 학습

Flamingo prompt는 다음과 같습니다.

```text
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

Model은 completion pattern을 보고 "bird"(또는 image3가 보여 주는 것)를 출력합니다. Gradient step은 없습니다. Frozen LLM의 in-context learning capability가 gated cross-attention을 통해 유지됩니다. 이것이 paper의 punchline이고 중요한 이유입니다.

### 학습 데이터

Flamingo는 세 dataset에서 train되었습니다.

1. MultiModal MassiveWeb(M3W): reading order를 reconstruct한 interleaved image와 text를 가진 43M web page.
2. Image-Text Pairs(ALIGN + LTIP): 4.4B pair.
3. Video-Text Pairs(VTP): 27M short video clip.

OBELICS(2023)는 interleaved web corpus의 open reproduction이며, Idefics, Idefics2, 대부분의 open "Flamingo-like" model이 이것으로 train됩니다.

### OpenFlamingo와 Otter

OpenFlamingo(2023)는 open reproduction입니다. Architecture는 동일합니다(frozen LLaMA 또는 MPT 위의 Perceiver resampler + gated cross-attention). Checkpoint는 3B, 4B, 9B입니다. Base LLM이 더 작고 data가 적어서 quality는 Flamingo보다 낮습니다.

Otter(2023)는 multimodal instruction dataset인 MIMIC-IT로 instruction tuning하여 OpenFlamingo를 확장했고, gated cross-attention이 instruction following에도 작동함을 보였습니다.

### 후손들

- Idefics / Idefics2 / Idefics3: Hugging Face의 gated cross-attention lineage입니다. 점점 단순해졌습니다(Idefics2는 adaptive pooling을 적용한 direct patch token을 선호하며 resampler를 제거했습니다).
- Flamingo-to-Chameleon transition: 2024년에는 많은 team이 early-fusion(Lesson 12.11)으로 이동했습니다. Flamingo-style gated cross-attention은 backbone freezing이 필요한 production에서 남아 있습니다.
- Gemini의 interleaved input: 정확한 mechanism은 proprietary이지만, conceptually Flamingo의 interleaved-format flexibility를 물려받았습니다.

### BLIP-2와의 비교

| | BLIP-2 | Flamingo |
|---|---|---|
| Visual bridge | Input에서 한 번 쓰는 Q-Former | M개 layer마다 gated cross-attention |
| Visual tokens | Image당 32 | Cross-attn layer마다 image당 64 |
| Frozen LLM | Yes | Yes |
| Few-shot in-context | 약함 | 강함 — paper의 중심 |
| Interleaved inputs | Native support 없음 | Yes, design target |
| Training data | 130M pairs | 1.3B pairs + 43M interleaved pages |
| Parameter count | 188M trained | 약 10B trained(cross-attn layers) |
| Compute | 8 A100에서 며칠 | 수천 TPUv4에서 몇 주 |

예산 내 single-image VQA에는 BLIP-2를 고르세요. Interleaved, few-shot, multi-image reasoning에는 Flamingo/Idefics2를 고르세요.

## 사용하기

`code/main.py`는 다음을 보여 줍니다.

1. 36개 fake patch token과 8개 learnable latent에서 동작하는 Perceiver resampler(pure Python cross-attention).
2. `alpha = 0`이면 output이 input과 같고(LLM unchanged), `alpha = 2.0`이면 visual contribution이 섞이는 gated cross-attention step.
3. "(image 1) (text 1) (image 2) (text 2)" sequence에 대한 2D attention mask를 만드는 interleaved-mask builder.

## 결과물

이 lesson은 `outputs/skill-gated-bridge-diagnostic.md`를 생성합니다. Open VLM의 config(resampler Y/N, cross-attn frequency, gate scheme)가 주어지면 Flamingo lineage element를 식별하고 freezing strategy를 설명합니다. Fine-tune이 text performance를 저하시킨 이유를 debugging할 때 유용합니다. 답은 gate가 너무 빨리, 너무 크게 열렸다는 것일 수 있습니다.

## 연습문제

1. Flamingo-9B의 visual parameter count를 계산하세요: 9B LLM + 1.4B gated cross-attention layers + 64M resampler. Total params 중 train되는 fraction은 얼마인가요?

2. PyTorch로 gated residual `y = tanh(alpha) * cross + x`를 구현하세요. `alpha=0`이면 init에서 `y==x`가 정확히 성립함을 실험으로 보이세요.

3. 각 prompt의 image count가 다른 batch에서 multiple image를 처리하는 방식에 대해 OpenFlamingo Section 3.2(arXiv:2308.01390)를 읽으세요. Padding strategy를 설명하세요.

4. Flamingo의 cross-attention mask는 왜 text token이 모든 preceding image가 아니라 *가장 최근* preceding image에만 attend하도록 하나요? Flamingo paper Section 2.4를 읽고 tradeoff를 설명하세요.

5. In-context few-shot: 새로운 Flamingo variant를 위해 "image -> main object의 color" 예시 4개가 있는 prompt를 구성하세요. Example 수를 0에서 8까지 바꿀 때 예상되는 accuracy pattern을 설명하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | Variable number of input patch에서 K개의 fixed token을 생성하는 module |
| Gated cross-attention | "Tanh-gated bridge" | Residual layer `y = tanh(alpha)*cross + x`, learnable alpha, init 0 |
| Interleaved input | "Mixed sequence" | Image와 text가 reading order 안에서 자유롭게 섞인 prompt format |
| Frozen LLM | "No LLM gradients" | Text LLM의 weight는 update되지 않고 resampler + cross-attn layer만 train됩니다 |
| Few-shot | "In-context examples" | Prompt에 몇 개의 (image, answer) pair를 넣으면 model이 finetuning 없이 generalize합니다 |
| OBELICS | "Interleaved web corpus" | Image와 text가 reading order로 배치된 141M web page open dataset |
| Chinchilla | "70B frozen base" | DeepMind의 Chinchilla paper에서 나온 Flamingo의 frozen text LLM |
| Gate schedule | "How alpha moves" | Training 중 cross-attention gate가 열리는 속도 |
| Cross-attn frequency | "Every M layers" | Gated cross-attention block이 삽입되는 빈도입니다. Flamingo는 M=4를 사용합니다 |
| OpenFlamingo | "Open reproduction" | 3-9B MosaicML/LAION open checkpoint이며 Flamingo와 architecture가 동일합니다 |

## 더 읽을거리

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — original paper.
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — open reproduction.
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — interleaved web corpus.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — general Perceiver architecture.
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — instruction-tuned Flamingo descendant.
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo approach의 modern simplification.
