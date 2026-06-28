---
name: sd-toolkit-composer
description: 주어진 input set에 대해 SD / Flux base 위에 ControlNet, LoRA, IP-Adapter를 조합합니다.
version: 1.0.0
phase: 8
lesson: 08
tags: [controlnet, lora, ip-adapter, diffusion]
---

Task(target image), input(prompt, reference image, pose / depth / scribble / seg, subject identity), base model(SDXL, SD3.5, Flux.1-dev)이 주어지면 다음을 출력합니다.

1. ControlNet stack. 어떤 ControlNet(canny / openpose / depth / scribble / seg / lineart / tile)을 어떤 weight와 순서로 쓸지. Weight 합의 최대값은 &lt;= 1.5입니다.
2. LoRA stack. 이름 있는 LoRA, rank, alpha. alpha &gt; 1.5이거나 여러 LoRA가 같은 concept을 겨냥하면 경고합니다.
3. IP-Adapter. None, plain, 또는 FaceID variant. 일반적인 weight는 0.4-0.8입니다.
4. Text prompt + negative prompt. Keyword order, token budget, negative scaffolding.
5. Sampler + CFG + seed. Euler A / DPM-Solver++ / LCM. CFG scale은 base와 연결합니다. Reproducible seed protocol을 포함합니다.
6. QA checklist. ControlNet drift, LoRA over-saturation, IP-Adapter identity leak, anatomy issue에 대한 시각적 점검.

SD 1.5 LoRA를 SDXL base에 stack하는 것은 거부하세요(dimension mismatch). Weight 1.0짜리 ControlNet을 3개 이상 실행하는 것도 거부하세요(feature collision). 사용자가 SDXL 또는 Flux를 돌릴 GPU budget이 있는데 SD 1.5를 추천하면 표시하세요. &lt; 10장 이미지로 하는 LoRA identity training은 overfit 가능성이 높다고 표시하세요.
