# Janus-Pro: 통합 멀티모달 모델을 위한 분리형 인코더

> Unified multimodal model에는 피할 수 없는 긴장이 있다. Understanding은 semantic feature를 원한다. SigLIP이나 DINOv2는 concept-level information이 풍부한 vector를 낸다. Generation은 reconstruction-friendly code를 원한다. VQ token은 다시 선명한 pixel로 조합된다. 두 목표는 단일 encoder 안에서 양립하지 않는다. Janus(DeepSeek, 2024년 10월)와 Janus-Pro(DeepSeek, 2025년 1월)는 해결책은 시도를 멈추는 것이라고 주장한다. 두 encoder를 decouple하라. Task 사이에서 transformer body는 공유하되, understanding은 SigLIP으로 route하고 generation은 VQ tokenizer로 route한다. 7B에서 Janus-Pro는 GenEval에서 DALL-E 3를 이기고, MMMU에서는 LLaVA와 맞먹는다. 이 lesson은 하나의 encoder가 실패하는 곳에서 두 encoder가 왜 작동하는지 읽는다.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 minutes

## 학습 목표

- 단일 shared encoder가 understanding 또는 generation 품질 중 하나를 왜 compromise하는지 설명한다.
- Janus-Pro의 routing을 설명한다. Understanding input side에는 SigLIP feature를, generation의 input/output에는 VQ token을 쓴다.
- Janus-Pro가 Janus가 실패한 곳에서 성공하게 만든 data-mix scaling을 추적한다.
- Decoupled(Janus-Pro), coupled-continuous(Transfusion), coupled-discrete(Show-o) architecture를 비교한다.

## 문제

Unified model은 understanding과 generation 전반에서 transformer body를 공유한다. 이전 시도(Chameleon, Show-o, Transfusion)는 모두 양방향에 하나의 visual tokenizer를 쓴다. Tokenizer는 compromise다.

- Reconstruction(generation)에 optimized: VQ-VAE는 fine-grained pixel detail을 잡지만 semantic coherence가 약한 token을 만든다.
- Semantics(understanding)에 optimized: SigLIP embedding은 "cat" image를 "cat" token 근처에 모으지만 좋은 reconstruction은 허용하지 않는다.

Show-o와 Transfusion은 이 때문에 한 방향에서 눈에 보이는 quality tax를 치른다. Janus-Pro는 묻는다. Task가 다른 필요를 갖는데 왜 하나의 tokenizer를 요구하는가?

## 개념

### 분리형 시각 인코딩

Janus-Pro의 architecture는 두 encoder를 분리한다.

- Understanding path. Input image → SigLIP-SO400m → 2-layer MLP → transformer body.
- Generation path. Input image(기존 image에 condition하는 경우) → VQ tokenizer → token ID → transformer body.
- 출력 생성. Transformer가 예측한 image token → VQ decoder → pixel.

Transformer body는 공유된다. Body의 upstream과 downstream은 모두 task-specific이다.

Input은 prompt format으로 disambiguate한다. `<understand>` tag는 SigLIP으로 route하고, `<generate>`는 VQ로 route한다. 또는 task에서 routing이 implicit하게 결정된다.

### 왜 작동하는가

Understanding loss는 SigLIP feature를 받는다. CLIP-style pretraining이 semantic similarity에 맞게 tuning한 feature다. Input feature가 task에 더 적합하기 때문에 model의 perception benchmark는 Show-o / Transfusion보다 좋아진다.

Generation loss는 VQ token을 받는다. Tokenizer가 reconstruction에 맞게 tuning한 token이다. VQ code가 pixel로 깨끗하게 compose되기 때문에 image quality는 Show-o보다 좋아진다.

Shared transformer body는 두 input distribution(SigLIP과 VQ)을 보고 둘 모두와 일하는 법을 배운다. 주장은 이렇다. 충분한 data와 충분한 parameter가 있으면 body가 switching을 흡수한다.

### 데이터 스케일링: Janus와 Janus-Pro

Janus(original, arXiv 2410.13848)는 decoupling을 소개했지만 작은 scale(1.3B parameter, 제한된 data)이었다. Janus-Pro(arXiv 2501.17811)는 scale을 키웠다.

- 7B parameter(1.3B 대비).
- Stage 1(alignment)용 image-text pair 90M, 기존 72M에서 증가.
- Stage 2(unified)용 72M, 기존 26M에서 증가.
- Stage 3용 image-gen instruction sample 200k 추가.

결과적으로 Janus-Pro-7B는 MMMU에서 LLaVA와 맞먹고(60.3 vs ~58), GenEval에서는 DALL-E 3를 이긴다(0.80 vs 0.67). 하나의 open model이 unified spectrum 양쪽에서 경쟁력 있게 작동한다.

### JanusFlow: rectified flow 변형

JanusFlow(arXiv 2411.07975)는 VQ generation path를 rectified-flow generation path(continuous)로 바꾼다. Split은 SigLIP-for-understanding + rectified-flow-for-generation이 된다. Quality ceiling은 더 올라간다. Architecture는 decoupled-encoders-shared-body로 남는다.

### Shared body의 일

Transformer body는 unified sequence를 처리하지만 두 input distribution을 받는다. Body의 일은 다음과 같다.

- Understanding: SigLIP feature + text token을 소비하고 text를 autoregressively emit한다.
- Generation: text token + optional image VQ token을 소비하고 image VQ token을 autoregressively emit한다.

Body에는 block별 modality-specific weight가 없다. Qwen이나 Llama 안에서 기대할 만한 text-style transformer에 두 input adapter를 더한 형태다.

흥미롭게도 이는 Janus-Pro의 body를 pretrained LLM에서 initialize할 수 있음을 뜻한다. Janus-Pro는 실제로 DeepSeek-MoE-7B에서 initialize한다. 이 선택은 중요하다. LLM이 reasoning ability를 제공하기 때문에 pure-from-scratch unified model이 도달하기 어려운 수준에 닿는다.

### InternVL-U와 비교

InternVL-U(Lesson 12.10)는 2026년 follow-up이다. 다음을 결합한다.

- Native multimodal pretraining(InternVL3 backbone).
- Decoupled-encoder routing(SigLIP in, VQ + diffusion heads out).
- Unified understanding + generation + editing.

InternVL-U는 Janus-Pro의 architectural choice를 더 큰 framework 안에 흡수한다. Decoupled-encoder idea는 이제 scale 있는 unified model의 기본값이다.

### 한계

Decoupled encoder는 architecture complexity를 더한다. 학습해야 할 tokenizer가 둘이고, 유지해야 할 input path가 둘이며, fail mode도 두 set이다. Generation이 필요 없는 product에는 Janus-Pro가 과설계다. LLaVA 계열 understanding model을 고르라.

Understanding이 필요 없는 product에도 Janus-Pro는 과하다. Stable Diffusion 3 / Flux model을 고르라.

둘 다 필요한 product에서는 Janus-Pro가 현재 reference open architecture다.

## 활용하기

`code/main.py`는 Janus-Pro routing을 simulation한다.

- 두 mock encoder: SigLIP-like(256-dim semantic vector 생성)와 VQ-like(integer code 생성).
- Task tag를 기준으로 encoder를 고르는 prompt router.
- 어떤 encoder가 만들었든 token sequence를 처리하는 shared body(stand-in).
- Stage 1(alignment)에서 stage 3(instruction tune)로 전환되는 weighted-sample schedule.

Image QA, T2I, image editing 세 예시의 routed path를 출력한다.

## 산출물

이 lesson은 `outputs/skill-decoupled-encoder-picker.md`를 만든다. Frontier-ish quality의 unified generation + understanding을 원하는 product가 주어지면 Janus-Pro, JanusFlow, InternVL-U 중 하나를 concrete data-scale recommendation과 함께 고른다.

## 연습 문제

1. Janus-Pro-7B는 GenEval에서 DALL-E 3를 이긴다. 7B open model이 generation에서는 frontier proprietary model과 맞먹을 수 있지만 understanding에서는 그렇지 않은 이유를 설명하라.

2. Router function을 구현하라. Prompt text가 주어지면 `understand` 또는 `generate`로 classify한다. "describe and then sketch"처럼 ambiguous한 prompt는 어떻게 처리할 것인가?

3. JanusFlow는 VQ path를 rectified flow로 바꾼다. Transformer body는 이제 무엇을 output하며, loss에서는 무엇이 바뀌는가?

4. Janus-Pro architecture가 decoupled encoder를 하나 더 추가해 처리할 수 있는 네 번째 task를 제안하라. 예: image segmentation(DINO-style), depth(MiDaS-style).

5. Janus-Pro Section 4.2의 data scaling을 읽어라. Janus 대비 T2I quality gain에 가장 크게 기여한 data stage는 무엇인가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | 방향별로 별도 tokenizer 또는 encoder를 두는 방식. Understanding에는 semantic, generation에는 reconstruction용을 쓴다 |
| Shared body | "One transformer" | 단일 transformer가 어느 encoder의 output이든 처리한다. Modality-specific weight는 없다 |
| SigLIP for understanding | "Semantic features" | 풍부한 conceptual feature를 제공하지만 reconstruction은 약한 CLIP 계열 vision tower |
| VQ for generation | "Reconstruction codes" | Pixel로 깨끗하게 decode되는 vector-quantized token |
| JanusFlow | "Rectified-flow variant" | VQ 대신 continuous flow-matching generation head를 쓰는 Janus-Pro |
| Routing tag | "Task tag" | Input encoder를 고르는 prompt marker(`<understand>` / `<generate>`) |

## 더 읽을거리

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
