# CLIP에서 BLIP-2까지 - 모달리티 브리지로서의 Q-Former

> CLIP은 image와 text를 정렬하지만 caption을 생성하거나, 질문에 답하거나, 대화를 이어갈 수는 없습니다. BLIP-2(Salesforce, 2023)는 작은 trainable bridge로 이 문제를 해결했습니다. 32개의 learnable query vector가 frozen ViT의 feature에 cross-attention하고, 그 결과가 frozen LLM의 input stream에 바로 들어갑니다. 188M parameter의 bridge가 11B LLM과 ViT-g/14를 연결했습니다. 2026년까지의 모든 adapter-based VLM(MiniGPT-4, InstructBLIP, LLaVA의 사촌들)은 이 후손입니다. 이 lesson은 Q-Former architecture를 읽고, two-stage training을 설명하며, visual token을 frozen text decoder에 넣는 toy version을 만듭니다.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 minutes

## 학습 목표

- Frozen vision encoder와 frozen LLM 사이의 trainable bottleneck이 cost와 stability에서 end-to-end finetuning보다 나은 이유를 설명합니다.
- 고정된 learnable query set이 external image feature에 attend하는 cross-attention block을 구현합니다.
- BLIP-2의 two-stage pretraining을 따라갑니다: representation(ITC + ITM + ITG), 이후 generative(frozen decoder를 사용하는 LM loss).
- Q-Former와 LLaVA가 사용하는 더 단순한 MLP projector를 비교하고, 언제 각 선택이 이기는지 논증합니다.

## 문제

Frozen ViT가 image마다 dim 1408의 patch token 256개를 만든다고 합시다. Frozen 7B LLM은 dim 4096의 token embedding을 기대합니다. 명백한 bridge인 1408에서 4096으로 가는 linear layer도 작동하지만, 256개 patch token을 모두 LLM context에 넣으면 image마다 256개의 추가 token을 소비합니다. 32개 image batch에서는 visual modality만으로 8192 token을 씁니다.

BLIP-2의 질문은 이것입니다. LLM이 captioning, question answering, image reasoning을 할 만큼 충분한 정보를 보존하면서 256-token image representation을 훨씬 적은 token(예: 32개)으로 압축할 수 있을까? 그리고 frozen backbone을 건드리지 않고 bridge parameter만으로 이 bridge를 train할 수 있을까?

답은 Q-Former입니다. 32개의 learnable "query" vector가 ViT patch token에 cross-attend하고, LLM이 소비하는 32-token visual summary를 만듭니다. 총 188M parameter입니다. LLM을 연결하기 전 contrastive, matching, generative objective로 train됩니다.

## 개념

### 학습 가능한 쿼리

Q-Former의 핵심 trick은 LLM의 text token이 image patch에 attend하도록 하지 않고, 32개의 learnable query vector `Q`를 새로 도입해 그 query들이 image patch에 attend하도록 하는 것입니다. Query는 model의 parameter입니다. Training 중 학습되며 모든 image에 같은 32개 query가 사용됩니다.

Cross-attention 이후 각 query는 image의 compressed summary를 담습니다. 예를 들면 "main object를 설명하라", "background를 설명하라", "object 수를 세라"와 같은 역할입니다. Query가 semantic label에 문자 그대로 specialize되는 것은 아닙니다. Downstream loss를 낮추는 어떤 encoding이든 학습합니다.

### 아키텍처

Q-Former는 두 경로를 가진 작은 transformer(12 layer, 약 100M params)입니다.

1. Query path: 32개의 query vector가 self-attention(서로 간), frozen ViT patch token에 대한 cross-attention, FFN을 통과합니다.
2. Text path: BERT-like text encoder가 query path와 self-attention 및 FFN weight를 공유합니다. Text path에서는 cross-attention이 비활성화됩니다.

Training time에는 두 path가 모두 실행됩니다. Query와 text는 shared self-attention을 통해 상호작용합니다. 그래서 query는 필요한 task(ITM, ITG)에서 text에 condition할 수 있습니다. VLM handoff를 위한 inference time에는 query만 흐르며 32개 visual token을 만듭니다.

### 2단계 학습

BLIP-2는 두 stage로 pretrain합니다.

Stage 1: representation learning(LLM 없음). 세 가지 loss:
- ITC(image-text contrastive): pooled query token과 text CLS token 사이의 CLIP-style contrastive.
- ITM(image-text matching): binary classifier입니다. 이 image-text pair가 match인가? Hard-negative-mined.
- ITG(image-grounded text generation): query에 conditioned된 text의 causal LM head입니다. Query가 text-generatable content를 encode하도록 강제합니다.

Q-Former만 train됩니다. ViT는 frozen입니다. LLM은 관여하지 않습니다.

Stage 2: generative learning. Frozen LLM(OPT-2.7B 또는 Flan-T5-XL 등)을 붙입니다. 32개 query output을 작은 linear layer로 LLM embedding dim에 project합니다. 이를 text prompt 앞에 붙입니다. Concatenated prompt + image + caption sequence의 LM loss로 linear projection과 Q-Former만 train합니다.

Stage 2 이후 Q-Former + projection이 완전한 visual adapter입니다. Inference에서는 image -> ViT -> Q-Former -> linear proj -> text 앞에 prepend -> frozen LLM이 output을 냅니다.

### 파라미터 경제성

ViT-g/14(1.1B, frozen) + OPT-6.7B(6.7B, frozen) + Q-Former(188M, trained)를 사용하는 BLIP-2는 총 8B이며 train되는 것은 188M입니다. Q-Former만 보면 전체 stack parameter의 약 2.4%입니다. Training cost도 이를 반영합니다. End-to-end에 몇 주가 필요한 것과 달리 A100 몇 장에서 며칠이면 됩니다.

Quality: BLIP-2는 50x 작으면서도 zero-shot VQA에서 Flamingo-80B와 같거나 더 낫습니다. Bridge가 작동합니다.

### InstructBLIP과 지시 인식 Q-Former

InstructBLIP(2023)은 Q-Former에 instruction text 자체라는 extra input을 추가합니다. Cross-attention time에 query는 image patch와 instruction 모두에 접근합니다. Query는 하나의 고정 summary를 학습하는 대신 instruction별로 specialize될 수 있습니다("자동차를 세라", "분위기를 묘사하라"). Held-out task benchmark가 개선됩니다.

### MiniGPT-4와 projector-only 접근

MiniGPT-4는 Q-Former를 유지했지만 나머지를 모두 freeze하고 output linear projection만 train했습니다. 저렴하지만 quality 비용이 있습니다. Query는 BLIP-2의 것이지 당신의 것이 아닙니다. 빠른 iteration에는 좋지만 최고의 architecture는 아닙니다.

### LLaVA가 더 단순한 길을 택한 이유

LLaVA(2023, Lesson 12.05)는 Q-Former를 plain 2-layer MLP로 교체했습니다. 이 MLP는 모든 ViT patch token을 LLM space로 project합니다. 24x24 grid의 image 하나당 576 token이 모두 LLM에 들어갑니다. 압축은 더 나쁘지만 LLM이 raw patch에 attend할 수 있습니다. 당시에는 논쟁적이었지만 2023년 말에는 dominant가 되었습니다. Visual instruction data(LLaVA-Instruct-150k)가 MLP를 train해 충분한 signal을 보존할 수 있음을 증명했기 때문입니다. Tradeoff는 LLaVA의 context가 더 빨리 찬다는 점이지만, multi-image와 video로 자연스럽게 scale됩니다.

2026년에는 field가 나뉘었습니다. Token budget이 중요한 곳(long video, many image)에서는 Q-Former가 살아남고, token당 raw quality가 priority인 곳에서는 MLP projector가 지배합니다.

### 게이트형 교차 어텐션: 원조 Flamingo

Flamingo(Lesson 12.04)는 BLIP-2보다 먼저 같은 cross-attention 아이디어를 사용했지만, 단일 bridge가 아니라 frozen LLM의 모든 layer에서 사용했습니다. BLIP-2는 input layer에만 압축해도 작동한다는 점을 보였습니다. Gemini와 Idefics는 둘을 조합합니다. Interleaved input token과 in-context few-shot을 위한 optional gated cross-attention입니다.

### 2026년의 후손들

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, 그리고 token budget 때문에 대부분의 video-language model.
- Perceiver resampler: Flamingo의 variant(Lesson 12.04); Idefics family, Eagle, OmniMAE.
- MLP projector: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Attention pool: VILA, PaliGemma.

네 가지 모두 유효합니다. 결정 질문은 token budget 제약인지, quality-per-token 제약인지입니다.

## 사용하기

`code/main.py`는 stdlib Q-Former-style cross-attention을 만듭니다.

1. 256개 image patch token(dim 128)을 simulate합니다.
2. 32개 learnable query(dim 128)를 instantiate합니다.
3. Scaled-dot-product cross-attention을 실행합니다(query에서 Q, patch에서 K/V).
4. Linear layer로 LLM-dim(512)에 project합니다.
5. LLM-ready visual token 32개를 출력합니다.

모든 수학은 pure Python(vector에 대한 nested loop)입니다. Toy이지만 shape는 정확합니다. Attention-weight matrix가 출력되므로 각 query가 어떤 patch에서 가져왔는지 볼 수 있습니다.

## 결과물

이 lesson은 `outputs/skill-modality-bridge-picker.md`를 생성합니다. Target VLM configuration(vision encoder token count, LLM context budget, deployment constraint, quality target)이 주어지면 Q-Former vs MLP vs Perceiver resampler를 짧은 근거와 함께 추천하고 각 bridge의 parameter-count estimate를 제공합니다.

## 연습문제

1. PyTorch로 cross-attention block을 구현하세요. 32개 query와 256개 key/value에서 attention-weight matrix가 32 x 256이며 softmax 이후 각 row의 합이 1인지 확인하세요.

2. BLIP-2 stage 1에서 Q-Former는 ITC, ITM, ITG 세 loss를 동시에 실행합니다. 각각의 forward signature를 pseudo-code로 작성하세요. 어느 loss가 text encoder path를 활성화해야 하나요?

3. Parameter count를 비교하세요. Q-Former(12 layer, 768 hidden) vs 2-layer MLP projector(1408 -> 4096, two layers). 어떤 LLM scale에서 188M Q-Former 비용이 training efficiency로 회수되나요?

4. Q-Former 초기화 방식에 대한 BLIP-2 paper(arXiv:2301.12597) Section 3.2를 읽으세요. Random이 아니라 BERT-base에서 initialize하면 convergence가 빨라지는 이유를 설명하세요.

5. 10분 video를 1 FPS로 sample해 60 frame으로 만들 때, (Q-Former -> 32 tokens/frame)과 (MLP projector -> 576 tokens/frame)의 frame당 token cost를 계산하세요. 어느 쪽이 128k-token LLM context window에 들어가나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 표현 | 실제 의미 |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | Frozen ViT feature에 cross-attend하는 32개 learnable query vector를 가진 작은 transformer |
| Learnable queries | "Soft prompt for vision" | Cross-attention의 query side 역할을 하는 고정 parameter set입니다. Model별로 학습되고 모든 input에서 공유됩니다 |
| Cross-attention | "Q from here, K/V from there" | Query, key, value가 서로 다른 source에서 오는 attention입니다. Query가 ViT patch에서 정보를 끌어오는 방식입니다 |
| ITC | "Image-text contrastive" | Q-Former pooled query와 text CLS에 적용되는 CLIP-style loss |
| ITM | "Image-text matching" | Hard-negative-mined pair에 대한 binary classifier입니다. Query가 fine-grained mismatch를 구분하도록 강제합니다 |
| ITG | "Image-grounded text generation" | Query에 conditioned된 text를 생성하는 causal LM loss입니다. Query가 text-decodable content를 encode하도록 강제합니다 |
| Two-stage pretraining | "Representation then generative" | Stage 1은 Q-Former만 train합니다(ITC/ITM/ITG). Stage 2는 frozen LLM을 붙이고 projection + Q-Former만 train합니다 |
| Frozen backbone | "Do not finetune" | Vision encoder와 LLM weight를 고정하고 bridge만 train합니다 |
| Projection head | "Linear to LLM dim" | Q-Former output을 LLM embedding dimension으로 mapping하는 final linear layer |
| Perceiver resampler | "Flamingo's version" | 비슷한 learnable-query cross-attention이며, BLIP-2처럼 single bridge가 아니라 Flamingo에서 모든 layer에 사용됩니다 |

## 더 읽을거리

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — core paper.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — ITC/ITM/ITG trio를 가진 predecessor.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — stage 1 training의 conceptual ancestor.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — instruction-aware Q-Former.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — projector-only approach.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — learnable-query cross-attention을 위한 general architecture.
