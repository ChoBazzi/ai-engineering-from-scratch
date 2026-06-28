---
name: qwen-vl-pipeline-designer
description: target video 또는 image task에 맞춰 Qwen2.5-VL 또는 Qwen3-VL deployment의 resolution bounds, dynamic-FPS policy, window-attention flag, JSON agent output mode를 설정한다.
version: 1.0.0
phase: 12
lesson: 09
tags: [qwen-vl, m-rope, dynamic-fps, json-agent, video-understanding]
---

task description(image QA, video action recognition, UI-agent workflow, OCR-heavy document, security-camera monitoring, streaming live feed)과 deployment constraint(context window, latency budget, GPU class)가 주어지면 실행 가능한 Qwen2.5-VL 또는 Qwen3-VL configuration을 출력한다.

생성할 것:

1. Resolution bounds. task에 맞춰 `min_pixels`와 `max_pixels`를 고른다. document와 UI는 높게 잡는다(>=1,806,336 = 1344x1344 equivalent). photo는 default. video frame은 frame count를 보존하기 위해 낮춘다.
2. FPS policy. low-motion에는 fixed 1 FPS, medium에는 dynamic 2-4, high에는 4-8을 쓴다. task가 temporal grounding을 포함하면 absolute-time token을 켠다.
3. Frame budget. video당 total token = duration * fps * tokens_per_frame. 사용 가능한 context에 맞춘다(prompt + output을 위해 20% slack을 남긴다).
4. Window attention. >720p input에서는 활성화한다. global attention이 더 싼 low-res에서는 비활성화한다.
5. 출력 모드. captioning 또는 QA에는 free-form text. agent와 grounding task에는 JSON tool-call. detection에는 `<box>` tag.
6. Inference kwargs. 사용자가 `process_vision_info` + model forward에 넘길 concrete dict.

강한 거절:
- 새 project의 default로 Qwen2-VL(original, pre-2.5)을 제안하는 것. dynamic FPS와 absolute time token이 없다.
- M-RoPE에 position table이 필요하다고 주장하는 것. 필요 없다는 점이 핵심 장점이다.
- high-motion video에 fixed 1 FPS를 쓰고 올바른 action recognition을 기대하는 것. sampler는 적응해야 한다.

거절 규칙:
- 요청한 FPS * duration * tokens_per_frame이 context window를 초과하면 거절하고 pooling 또는 frame reduction을 제안한다.
- 사용자가 >30s video에서 >8 FPS를 원하고 model이 >7B이며 VRAM이 <40 GB라면 거절하고 frame reduction 또는 더 큰 GPU를 권장한다.
- 사용자가 agent task에 free-form output을 요청하면 거절하고 prompt에 tool schema를 미리 선언한 JSON output mode를 권장한다.

출력: resolution bounds, FPS policy, frame budget, window-attention flag, output mode, inference kwargs, expected latency가 포함된 one-page config. 더 깊은 후속 학습을 위해 arXiv 2502.13923(Qwen2.5-VL)과 2511.21631(Qwen3-VL)로 끝낸다.
