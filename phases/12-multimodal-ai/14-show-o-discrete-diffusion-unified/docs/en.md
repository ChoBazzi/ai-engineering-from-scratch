# Show-o와 이산 확산 통합 모델

> Transfusion은 continuous representation과 discrete representation을 섞는다. Show-o(Xie et al., 2024년 8월)는 반대로 간다. Text token에는 causal next-token prediction을 쓰고, image token에는 MaskGIT 정신을 따른 masked discrete diffusion을 쓴다. 둘은 하나의 transformer 안에 hybrid attention mask와 함께 놓인다. 그 결과 하나의 backbone, modality별 하나의 tokenizer, 하나의 loss formulation(next-token을 masked prediction으로 확장)으로 VQA, text-to-image, inpainting, mixed-modality generation을 통합한다. 이 lesson은 Show-o design을 살펴본다. Masked discrete diffusion이 왜 parallel few-step image generator인지 설명하고, Transfusion 및 Emu3와 대비한다.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 minutes

## 학습 목표

- Masked discrete diffusion을 설명한다. Token을 uniform하게 mask한 뒤 transformer가 복구하도록 하는 schedule이다.
- Parallel image decoding(Show-o, MaskGIT)과 autoregressive image decoding(Chameleon, Emu3)을 speed와 quality 관점에서 비교한다.
- Show-o가 하나의 checkpoint로 처리하는 세 task인 T2I, VQA, image inpainting을 말할 수 있다.
- Masking schedule(cosine, linear, truncated)을 고르고 sample quality에 미치는 효과를 추론한다.

## 문제

Transfusion의 two-loss training은 작동하지만 dynamic이 더 까다롭다. Continuous diffusion loss는 discrete NTP loss와 다른 numerical scale에 있다. Loss weight balancing은 hyperparameter search다. Architecture는 효과적이지만 복잡하다.

Show-o의 답은 이렇다. Chameleon처럼 두 modality를 모두 discrete로 유지하되, 이미지는 sequential하게 생성하지 않고 masked discrete diffusion으로 parallel하게 생성한다. Training objective는 자연스럽게 next-token-prediction을 일반화하는 단일 masked-token-prediction이 된다.

## 개념

### 마스크 이산 확산(MaskGIT)

원래 Chang et al.(2022)의 MaskGIT 기법은 우아하다. 완전히 masked된 이미지(모든 token이 special `<MASK>` id)에서 시작한다. 각 step에서 모든 masked token을 parallel하게 예측한 뒤, confidence가 가장 높은 top-K prediction은 유지하고 나머지는 다시 mask한다. 약 8-16 iteration 뒤 모든 token이 채워진다. Step마다 몇 token을 unmask할지 정하는 schedule을 tuning한다. Cosine schedule이 잘 작동한다.

Training은 단순하다. Masking ratio를 [0, 1]에서 uniform하게 sample하고 image의 VQ token에 적용한 뒤, transformer가 masked token을 복구하도록 학습한다. Text에서 BERT가 한 일을 image generation 규모로 확장한 것이다.

### Show-o: 하나의 transformer, hybrid mask

Show-o는 causal-language-model transformer 안에 MaskGIT를 넣는다. Attention mask는 다음과 같다.

- Text token: causal(표준 LLM).
- Image token: image block 내부에서는 full bidirectional(masked token이 prediction 중 다른 모든 image token을 볼 수 있게).
- Text-to-image: text는 이전 image에 attend하고, image는 이전 text에 attend한다.

Training은 다음을 번갈아 수행한다.
1. Text sequence의 표준 NTP.
2. T2I sample: text → masked image token, masked-token-prediction loss.
3. VQA sample: image → text with masked text token(사실상 NTP).

Unified loss는 `<MASK>` token 위의 cross-entropy이며, text NTP(마지막 token만 "masked")와 image masked-diffusion(random subset이 masked)을 모두 포괄한다.

### 병렬 샘플링

Show-o는 이미지를 약 16 step에 생성한다. Token별 autoregressive의 약 1000 step이나 diffusion의 약 20 step과 다르다. 각 step마다 모든 masked token을 parallel하게 예측하고, confident top-K를 commit한 뒤 반복한다.

비교:
- Chameleon / Emu3(token 위의 autoregressive): N_tokens forward pass, 보통 이미지당 1024-4096.
- Transfusion(continuous diffusion): 약 20 step, 각 step은 full transformer pass.
- Show-o(masked discrete diffusion): 약 16 step, 각 step은 full transformer pass.

Show-o는 같은 규모 모델에서 Chameleon보다 빠르고, step count는 Transfusion과 대략 비슷하지만 per-step cost는 낮다(continuous MSE loss가 아니라 discrete vocab logit).

### 하나의 checkpoint 안의 task

Show-o는 inference에서 prompt format으로 선택되는 네 task를 지원한다.

- Text generation: 표준 autoregressive text output.
- VQA: image in, text out.
- T2I: text in, masked discrete diffusion을 통한 image out.
- Inpainting: 일부 token이 masked된 image를 채우기.

Inpainting capability는 masked-prediction training에서 공짜로 나온다. VQ-token grid의 한 region을 mask하고, 나머지와 text prompt를 넣은 뒤 masked token을 예측한다.

### 마스킹 스케줄

Step마다 몇 token을 unmask할지 정하는 schedule은 품질을 좌우한다. Show-o는 cosine을 권장한다.

```text
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Step 0에서는 모든 token이 masked(ratio 1.0)다. Step T에서는 mask가 없다. Cosine은 prediction이 가장 informative한 중간 ratio에 질량을 집중한다. Linear schedule도 작동하지만 더 빨리 plateau에 도달한다.

### Show-o2

Show-o2(2025 follow-up, arXiv 2506.15564)는 Show-o를 scale한다. 더 큰 LLM base, 더 나은 tokenizer, 개선된 mask schedule을 쓴다. Architecture pattern은 같다.

### Show-o의 위치

2026 taxonomy에서:

- Discrete token + NTP: Chameleon, Emu3. 단순하지만 inference가 느리다.
- Discrete token + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Parallel sampling이 가능하지만 tokenizer 때문에 여전히 lossy하다.
- Continuous + diffusion: Transfusion, MMDiT, DiT. 품질이 가장 높지만 training이 더 복잡하다.
- VLM 안의 continuous + flow matching: JanusFlow, InternVL-U. 가장 최신이다.

Task로 고른다. Open model 하나로 T2I + inpainting + VQA를 합리적인 속도로 원하면 Show-o다. 품질이 최우선이고 two-loss plumbing 비용을 감당할 수 있으면 Transfusion이다.

## 활용하기

`code/main.py`는 Show-o sampling을 simulation한다.

- 16개 VQ token으로 된 장난감 grid.
- Prompt와 현재 unmasked token을 바탕으로 logit을 예측하는 mock "transformer".
- Cosine schedule로 8 step 동안 parallel masked sampling.
- Intermediate state(mask pattern evolution)와 final token을 출력한다.

실행해서 mask가 step by step으로 사라지는 것을 보라.

## 산출물

이 lesson은 `outputs/skill-unified-gen-model-picker.md`를 만든다. Understanding(VQA, captioning)과 generation(T2I, inpainting)을 모두 필요로 하며 open-weights constraint가 있는 product가 주어지면 Show-o 계열, Transfusion/MMDiT 계열, Emu3 / Chameleon 계열 중 하나를 concrete trade-off와 함께 고른다.

## 연습 문제

1. Masked discrete diffusion은 약 16 step에 sample한다. 왜 1 step이 아닌가? Step 0에서 모든 것을 unmask하면 무엇이 깨지는가?

2. Inpainting은 masked diffusion에서 공짜다. Show-o의 inpainting이 specialist model보다 나은 product use case(실제 또는 가상)를 제안하라.

3. Cosine schedule vs linear schedule: T=8일 때 step별 unmasked token 수를 추적하라. 어느 쪽이 더 balanced한가?

4. 512x512 Show-o image는 1024 token이다. Vocab K=16384에서 모델은 1024 * log2(16384) = 14,336 bit(~1.75 KiB)의 data를 낸다. Stable Diffusion은 512*512*24 bit = 6,291,456 bit(~768 KiB)의 raw pixel을 낸다. Compression ratio는 얼마이며, 이로 인해 어떤 품질 대가를 치르는가?

5. LlamaGen(arXiv:2406.06525)을 읽어라. LlamaGen의 class-conditional autoregressive image model은 Show-o의 masked approach와 어떻게 다른가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Masked token을 예측하도록 학습하고, inference에서 가장 confident한 prediction부터 반복적으로 unmask하는 방식 |
| Cosine schedule | "Unmask schedule" | Inference step에 따른 mask ratio decay이며, confidence growth를 중간 구간에 집중한다 |
| Parallel decoding | "All tokens at once" | 각 step이 masked token 전체 sequence를 한 forward pass에서 예측한 뒤 top-K를 commit하는 방식 |
| Hybrid attention | "Causal + bidirectional" | Text token에는 causal이고 image block 내부에는 bidirectional인 mask |
| Inpainting | "Fill-in generation" | 일부 token이 masked된 image에 condition해 missing token을 예측하는 generation이며, training objective에서 공짜로 나온다 |
| Commitment rate | "Top-K per step" | Iteration마다 "done"으로 선언되는 token 수이며, inference와 quality trade-off를 제어한다 |

## 더 읽을거리

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
