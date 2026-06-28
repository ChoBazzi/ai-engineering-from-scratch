---
name: clip-zero-shot
description: CLIP / SigLIP checkpoint로 zero-shot image classification을 실행하고 similarity score가 포함된 ranked prediction을 생성합니다.
version: 1.0.0
phase: 12
lesson: 02
tags: [clip, siglip, zero-shot, vision-language]
---

이미지 목록(file path 또는 URL)과 candidate class name 목록이 주어지면, 선언된 CLIP 또는 SigLIP checkpoint를 사용해 ranked zero-shot classification을 생성하세요. 이 skill은 pure-prediction이며 train이나 finetune을 수행하지 않습니다.

생성할 내용:

1. Prompt construction. 각 class에 대해 N개의 text template을 만드세요(기본값: `a photo of a {class}`, `a picture of a {class}`, `an image of a {class}`). 각 prompt를 text encoder로 embed하고 평균을 내 class prototype을 만드세요.
2. Image embedding. 각 input image를 지정된 vision encoder로 embed하세요. 양쪽을 모두 unit length로 normalize하세요.
3. Ranked predictions. 각 image embedding과 각 class prototype 사이의 cosine similarity를 계산하세요. score와 함께 top-1 및 top-5를 반환하세요.
4. Checkpoint metadata. 사용한 정확한 Hugging Face checkpoint 이름(예: `openai/clip-vit-large-patch14` 또는 `google/siglip2-so400m-patch14-384`)과 그 checkpoint가 기대하는 resolution을 명시하세요.
5. Honesty notice. Pretraining distribution 밖의 class에 대한 zero-shot은 신뢰하기 어렵다고 밝히세요. top-1 score를 confidence proxy로 제시하고 0.2 미만이면 경고하세요.

강한 거절 기준:
- 호출자가 제공한 목록에 없는 class에 대해 출력을 definitive label로 표현하는 모든 사용.
- 서로 다른 checkpoint의 score가 비교 가능하다는 주장. SigLIP과 CLIP은 서로 다른 scale로 score를 냅니다.
- downstream consent policy 없이 사람이 포함된 것으로 알려진 이미지에서 실행하는 경우.

거절 규칙:
- 호출자가 medical, legal, safety-critical category(진단, identity, protected attribute)로 분류하라고 요청하면 거절하고 audit trail이 있는 supervised model로 안내하세요.
- 호출자가 하나의 class name만 제공하면(대안이 없는 one-way classification) 거절하세요. zero-shot이 의미 있으려면 최소 두 candidate가 필요합니다.
- checkpoint가 지정되지 않았으면 거절하고 (CLIP, OpenCLIP, SigLIP, SigLIP 2) 중 무엇인지와 scale을 물어보세요.

출력: 이미지별 top-5 prediction ranked list, cosine similarity score, checkpoint name, 사용한 prompt template, confidence flag. 마지막에는 variable aspect ratio를 다루는 NaFlex는 Lesson 12.06, 더 깊은 내용은 SigLIP 2 paper를 가리키는 "what to read next" 문단으로 끝내세요.
