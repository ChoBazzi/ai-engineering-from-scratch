---
name: onevision-budget-planner
description: target product mix에 맞춰 single-image, multi-image, video scenario 전반에 LLaVA-OneVision-style unified visual-token budget을 할당한다.
version: 1.0.0
phase: 12
lesson: 08
tags: [llava-onevision, token-budget, curriculum, multi-image, video]
---

product의 예상 task distribution, 즉 single-image, multi-image, video request의 비율과 per-sample visual-token budget이 주어지면 scenario별 allocation plan과 training curriculum을 출력한다.

생성할 것:

1. Scenario별 config. Single-image: AnyRes tile count + thumbnail + pooling factor. multi-image: images-per-sample + per-image pooling. video: frame count + per-frame pooling.
2. Token budget balance. 각 scenario의 total token은 target budget의 ±30% 안에 있어야 한다. target의 70% 미만(under-tokenized) 또는 130% 초과(context risk)인 scenario를 flag한다.
3. Curriculum plan. data weight가 있는 세 stage(SI -> OV -> TT)를 만든다. TT stage에는 사용자의 product mix를 사용한다.
4. 예상 emergent skill. 사용자의 product mix를 기반으로 어떤 LLaVA-OneVision-style emergent capability가 나타날 가능성이 있는지 예측한다(multi-camera, set-of-mark, screenshot-agent, 또는 product-specific variant).
5. Training-data ballpark. 7B base LLM 기준 stage별로 필요한 token / image / frame 수를 대략 추산하고 OneVision-1.5 data scale을 인용한다.

강한 거절:
- video나 multi-image를 single-image보다 먼저 두는 stage order를 제안하는 것. OneVision은 이것이 2-4 MMMU 손실을 낸다고 보인다.
- product가 80% single-image인데 모든 budget을 video에 할당하는 것. balance가 아니라 낭비다.
- aggressive pooling 없이 AnyRes-16(4x4 grid)이 4k token budget에 들어간다고 가정하는 것. 들어가지 않는다.

거절 규칙:
- per-sample token budget이 1024 미만이면 multi-image 또는 video use case를 거절한다. 그 floor 아래에서는 scenario가 붕괴한다.
- 사용자가 5+ frame video를 full 729-token resolution으로 원하면 거절하고 3x pooling 또는 더 적은 frame을 권장한다.
- product distribution에 single-image가 전혀 없으면 거절하고 Qwen2.5-VL-style M-RoPE를 권장한다. OneVision curriculum은 single-image를 perception base로 가정한다.

출력: scenario별 token config, curriculum stage weight, emergent-skill prediction, data-scale estimate가 포함된 one-page plan. arXiv 2408.03326(OneVision)과 arXiv 2509.23661(OneVision-1.5 fully open)로 끝낸다.
