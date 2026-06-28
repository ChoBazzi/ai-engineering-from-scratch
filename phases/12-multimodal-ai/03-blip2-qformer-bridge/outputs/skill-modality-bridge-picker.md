---
name: modality-bridge-picker
description: token budget, quality target, training compute가 주어진 VLM configuration에 대해 Q-Former, MLP projector, Perceiver resampler 중 하나를 추천합니다.
version: 1.0.0
phase: 12
lesson: 03
tags: [blip2, qformer, vlm, modality-bridge, architecture]
---

vision encoder의 이미지당 token count, LLM의 context budget, prompt당 대상 이미지 수, training compute budget이 주어지면 사용할 modality bridge를 추천하고 parameter count와 token economics로 정당화하세요.

생성할 내용:

1. Token budget audit. Vision encoder의 이미지당 raw token, 각 bridge option 이후의 이미지당 token, 선언된 image-per-prompt 수에서 소비되는 LLM context 비율을 보고하세요.
2. Bridge comparison. Q-Former(32 token, 약 188M params), MLP projector(모든 patch, 약 20M params), Perceiver resampler(N-layer cross-attention을 통한 K learnable query, 가변) 각각에 대해 parameter, quality proxy, training cost ballpark를 제시하세요.
3. Recommendation. 주어진 constraint에 맞는 단일 최선 선택을 한 줄 근거와 함께 제시하세요. constraint가 모순될 때(high quality + tight token budget + low training compute) 표시하세요.
4. Two-stage training trace. Q-Former가 선택되면 stage 1의 ITC + ITM + ITG loss와 stage 2의 LM loss를 개요로 쓰세요. 각각의 대표 dataset(COCO, LAION, Visual Genome)을 명시하세요.
5. Ablation checklist. Bridge를 확정하기 전에 호출자가 실행해야 할 다섯 가지 실험(query count, two-stage vs single-stage, projector depth, freeze schedule, finetune subset)을 제시하세요.

강한 거절 기준:
- Token budget을 무시하는 모든 추천. 이미지당 576 token의 "Use MLP"는 4k context에서 10장 이미지일 때 실패합니다.
- Q-Former가 MLP를 엄격히 지배한다고 주장하는 경우. Context가 무제한인 single-image high-quality 작업에서는 MLP가 이깁니다.
- Perceiver resampler를 Q-Former와 동등하게 취급하는 경우. Flamingo는 이를 모든 LLM layer에 적용하고, BLIP-2는 한 번만 적용합니다.

거절 규칙:
- 호출자가 frame 수와 frame rate를 지정하지 않고 video를 처리할 bridge를 요청하면 거절하세요. Video bridge는 단순 scale뿐 아니라 specification에서도 single-image bridge와 다릅니다.
- 범위의 LLM이 vision tower와 함께 처음부터 학습되는 경우(early-fusion, Chameleon-style) 거절하세요. 그 경우는 Lesson 12.11에서 별도로 다룹니다.
- Training compute가 명시되지 않았으면 거절하고 호출자가 BLIP-2 stage 2(약 수백 A100-hour)를 감당할 수 있는지, 아니면 projector-only training만 가능한지 물어보세요.

출력: token math, parameter count, recommended architecture, training outline, ablation checklist를 포함한 한 페이지 bridge recommendation. 마지막에는 cross-attention-everywhere는 Lesson 12.04(Flamingo), MLP-only는 Lesson 12.05(LLaVA), data-vs-architecture tradeoff는 Lesson 12.07(ablations)을 가리키는 "what to read next" 문단으로 끝내세요.
