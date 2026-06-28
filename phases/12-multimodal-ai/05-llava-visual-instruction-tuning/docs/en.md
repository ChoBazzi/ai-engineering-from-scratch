# LLaVA와 시각 지시 튜닝

> LLaVA(2023년 4월)는 지구상에서 가장 많이 복제된 multimodal architecture입니다. BLIP-2의 Q-Former를 2-layer MLP로 바꾸고, Flamingo의 gated cross-attention을 naive token concatenation으로 바꾸었으며, text-only caption에서 GPT-4가 생성한 158k visual-instruction turn으로 train했습니다. 2023년부터 2026년 사이 VLM을 만든 실무자는 거의 모두 LLaVA의 어떤 variant를 만들었습니다. LLaVA-1.5는 AnyRes를 추가했습니다. LLaVA-NeXT는 resolution을 올렸습니다. LLaVA-OneVision은 image, multi-image, video를 하나의 recipe로 통합했습니다. 이 lesson은 recipe를 읽고, projector를 구현하며, 왜 "더 단순한 것이 이겼는지" 설명합니다.

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 minutes

## 학습 목표

- ViT patch embedding(dim 1024)을 LLM embedding dim(dim 4096)으로 mapping하는 2-layer MLP projector를 만듭니다.
- LLaVA two-stage recipe를 따라갑니다: (1) 558k caption pair에서 projector alignment, (2) GPT-4가 생성한 158k turn에서 visual instruction tuning.
- Image token placeholder, system prompt, user/assistant turn을 포함한 LLaVA-format prompt를 구성합니다.
- Q-Former가 token-budget에서 이겼는데도 community가 Q-Former에서 MLP로 이동한 이유를 설명합니다.

## 문제

BLIP-2의 Q-Former(Lesson 12.03)는 image를 32 token으로 압축합니다. 깔끔하고 효율적이며 benchmark에 좋습니다. 하지만 두 가지 문제가 있습니다.

첫째, Q-Former는 trainable이지만 그 loss가 final task가 아닙니다. Stage 1은 ITC+ITM+ITG를 train합니다. Stage 2는 LM loss를 train합니다. Query는 LLM이 나중에 decode해야 하는 어떤 intermediate representation을 학습합니다. Bottleneck에서 정보가 손실됩니다.

둘째, Q-Former는 188M params가 필요하고, LLaVA가 등장한 2023년 scale에서는 target LLM과 함께 co-design해야 했습니다. LLM을 바꾸면 Q-Former를 retrain해야 합니다. Vision encoder를 바꿔도 retrain해야 합니다. 모든 조합이 별도 R&D project였습니다.

LLaVA의 답은 민망할 정도로 단순했습니다. ViT의 576개 patch token을 가져와 각각을 2-layer MLP(`1024 -> 4096 -> 4096`)에 통과시키고, 576개 모두를 LLM input sequence에 쏟아 넣는 것입니다. Bottleneck이 없습니다. 이상한 objective에 대한 stage 1 pretraining도 없습니다. Direct LM loss로 MLP를 train할 뿐입니다.

Data는 어디서 올까요? LLaVA의 두 번째 insight는 GPT-4(text-only)를 사용해 instruction data를 생성하는 것이었습니다. Image에 대한 COCO caption과 bounding-box data를 GPT-4에 넣고, conversation, description, complex reasoning question을 만들라고 요청합니다. 158k instruction-response turn이 공짜로 생깁니다. Human annotation은 없습니다.

결과는 8 A100에서 하루 동안 실행되고, MMMU에서 Flamingo를 이기며, community가 확장할 수 있는 open checkpoint를 배포한 VLM이었습니다. 2023년 말에는 50개 이상의 fork를 낳았습니다.

## 개념

### 아키텍처

LLaVA-1.5 at 13B:
- Vision encoder: CLIP ViT-L/14 @ 336(stage 1에서는 frozen, stage 2에서는 optional unfrozen).
- Projector: GELU activation을 가진 2-layer MLP, `1024 -> 4096 -> 4096`.
- LLM: Vicuna-13B(이후 Llama-3.1-8B).

Image + text prompt의 forward pass:

```text
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Image는 LLM context의 576 token을 차지합니다. 2048 context에서는 text에 1472 token이 남습니다. 32k context에서는 반올림 오차 수준입니다.

### 1단계: 프로젝터 정렬

ViT를 freeze합니다. LLM을 freeze합니다. 2-layer MLP만 train합니다. Dataset은 558k image-caption pair(LAION-CC-SBU)입니다. Loss는 projected image token에 conditioned된 caption의 language modeling입니다.

Batch 128에서 single epoch를 돌리면 몇 시간 안에 끝납니다. Projector는 ViT-space를 LLM-space로 mapping하는 법을 배웁니다. Task-specific supervision은 없습니다.

### 2단계: 시각 지시 튜닝

Projector를 unfreeze합니다(계속 trainable). LLM도 unfreeze합니다(보통 full, 때로는 LoRA). 158k visual-instruction turn에서 train합니다.

Instruction data가 핵심입니다. Liu et al.은 다음 방식으로 생성했습니다.
1. COCO image를 가져옵니다.
2. Text description(5개 human caption + bounding-box list)을 추출합니다.
3. GPT-4에 세 prompt template으로 보냅니다.
   - Conversation: "Generate a back-and-forth dialogue between a user and assistant about this image."
   - Detailed description: "Give a rich, detailed description of the image."
   - Complex reasoning: "Ask a question that requires reasoning about the image, then answer it."
4. GPT-4 output을 (instruction, response) pair로 parse합니다.

이 과정은 image를 직접 건드리지 않습니다. Text description만 사용합니다. GPT-4는 그럴듯한 image content를 hallucinate합니다. Noise가 있지만 작동했습니다. 158k turn만으로 dialogue가 열렸습니다.

### Community가 이를 복제한 이유

- Tune해야 하는 stage-1-specific loss가 없습니다. 처음부터 끝까지 LM loss입니다.
- Projector가 며칠이 아니라 몇 시간에 train됩니다.
- Projector만 retrain하면 LLM을 교체할 수 있습니다(LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3).
- Visual-instruction data pipeline은 GPT-4를 사용하며 새 domain에 대해 저렴하게 재생성할 수 있습니다.

### LLaVA-1.5 and LLaVA-NeXT

LLaVA-1.5(2023년 10월)는 다음을 추가했습니다.
- Instruction tuning에 섞은 academic-task data(VQA, OKVQA, RefCOCO).
- 더 나은 system prompt.
- 2048 -> 32k context.

LLaVA-NeXT(2024년 1월)는 다음을 추가했습니다.
- AnyRes: high-res image를 336x336 crop의 2x2 또는 1x3 grid와 하나의 global low-res thumbnail로 나눕니다. 각 crop은 576 token이 되며 총 visual token은 image당 약 2880개입니다. OCR과 chart task가 크게 뛰었습니다.
- ShareGPT4V(high-quality GPT-4V caption)를 포함한 더 나은 instruction data mixture.
- 더 강한 base LLM(Mistral-7B, Yi-34B).

### LLaVA-OneVision

Lesson 12.08은 OneVision을 깊게 다룹니다. 짧게 말하면 같은 projector를 쓰되 single-image, multi-image, video를 하나의 model에서 shared visual-token budget으로 다루는 curriculum으로 train합니다.

### Q-Former와의 비교

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576(base) 또는 2880(AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | LM only |
| LLM drop-in | Retrain 필요 | 최소 retrain으로 교체 |
| Multi-image | 어색함 | 자연스러움(concat) |
| Video | 어색함 | 자연스러움(per-frame concat) |
| Token budget | 작음 | 큼 |

MLP는 simplicity와 token flexibility에서 이깁니다. Q-Former는 token budget에서 이깁니다. 2023년 말에는 token budget이 더 이상 binding constraint가 아니게 되었고(LLM context가 32k-128k+로 커짐) simplicity가 지배했습니다.

### 프롬프트 형식

```text
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>`는 placeholder token입니다. Tokenization 전에 576개 visual token(AnyRes에서는 2880개)으로 대체됩니다. Tokenizer는 train 때보다 약간 긴 sequence를 보지만, stage 1이 이를 가르쳤기 때문에 LLM은 새로운 input을 처리합니다.

### 파라미터 경제성

LLaVA-1.5-7B breakdown:
- CLIP ViT-L/14 @ 336: 303M(stage 1에서는 frozen, stage 2에서는 종종 unfrozen).
- Projector(2x linear): 약 22M trainable.
- Llama-7B: 7B.
- Total: 7.3B params. Stage 2에서 trainable한 것은 full 7B + 22M projector.

Stage 2 training cost: 8xA100에서 약 20시간. 이것이 핵심 숫자입니다. 하루, 하나의 node, reproducible. 그래서 LLaVA가 퍼졌습니다.

## 사용하기

`code/main.py`는 다음을 구현합니다.

1. Pure Python의 2-layer MLP projector(toy scale에서 dim 16 -> 32 -> 32).
2. Prompt-building pipeline: system prompt + `<image>`를 N projected token으로 교체 + user turn + assistant generation placeholder.
3. 576-token visual block이 LLM context에서 어떻게 보이는지 보여 주는 visualizer(2k / 32k / 128k context 소비 비율).

## 결과물

이 lesson은 `outputs/skill-llava-vibes-eval.md`를 생성합니다. LLaVA-family checkpoint가 주어지면 10-prompt vibes-eval suite(3 captioning, 3 VQA, 2 reasoning, 2 refusal)를 실행하고 사람이 읽기 쉬운 scorecard를 보고합니다. Benchmark가 아니라 projector와 LLM이 잘 연결되는지 확인하는 smoke test입니다.

## 연습문제

1. `1024 -> 4096 -> 4096`의 2-layer MLP projector에 대한 trainable-parameter count를 계산하세요. GELU와 bias를 포함하면 LLaVA-13B의 몇 fraction인가요?

2. "Refusal" case의 LLaVA prompt를 구성하세요. Image에는 private individual이 포함되어 있습니다. 예상 assistant response를 작성하세요. LLaVA가 왜 zero-shot으로 이를 거절해야 하며, refusal을 강화하려면 어떤 training data가 필요한가요?

3. LLaVA-NeXT blog의 AnyRes section을 읽으세요. 1344x672 image를 AnyRes로 처리할 때 visual token count를 계산하세요. 336x336의 base 576 token과 비교하세요.

4. LLaVA stage-1 projector는 caption의 LM loss로 train됩니다. Stage 1을 건너뛰고 바로 stage 2(visual instruction tuning)로 가면 어떻게 되나요? 답을 위해 Prismatic VLMs ablation(arXiv:2402.07865)을 인용하세요.

5. LLaVA-Instruct-150k는 COCO caption과 GPT-4를 사용해 instruction을 생성합니다. 새 domain(medical X-ray, satellite imagery)을 위해 domain instruction을 생성하는 네 단계 data pipeline을 설명하세요. 각 단계에서 무엇이 잘못될 수 있나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Projector | "MLP bridge" | ViT dim을 LLM dim으로 mapping하는 GELU 포함 2-layer MLP |
| Image token | "<image> placeholder" | Inference 전에 N projected visual token으로 대체되는 prompt marker |
| Visual instruction tuning | "LLaVA stage 2" | GPT-4가 생성한 (image, instruction, response) triplet에서의 training |
| Stage 1 alignment | "Projector pretraining" | ViT와 LLM을 freeze하고 caption의 LM loss로 projector를 train합니다 |
| AnyRes | "Multi-crop tiling" | High-res image를 tile grid로 나누고 각 tile의 visual token을 concatenate합니다 |
| LLaVA-Instruct | "GPT-4-generated" | COCO caption + GPT-4에서 synthesized된 158k instruction-response pair |
| Vision encoder freeze | "Backbone locked" | CLIP weight가 stage 1에서 update되지 않으며 stage 2에서도 때로는 update되지 않습니다 |
| ShareGPT4V | "Better captions" | GPT-4V가 생성한 1M dense caption이며 더 높은 quality alignment에 사용됩니다 |
| VQA | "Visual question answering" | Image에 대한 free-form question에 답하는 task |
| Prismatic VLMs | "Design-space paper" | Projector와 data choice를 체계적으로 test한 Karamcheti 2024 ablation |

## 더 읽을거리

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA paper.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions dataset.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — unified single-image, multi-image, video.
