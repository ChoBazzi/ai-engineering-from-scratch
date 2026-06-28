---
name: vit-configurator
description: 새 비전 작업에 맞는 ViT 변형, 패치 크기, 사전학습 소스를 고릅니다.
version: 1.0.0
phase: 7
lesson: 9
tags: [transformers, vit, vision]
---

비전 작업(classification / segmentation / detection / retrieval), 이미지 해상도, 데이터셋 크기(labeled + unlabeled), 배포 대상을 입력받아 다음을 출력하세요.

1. 백본. 다음 중 하나입니다. DINOv2 ViT-L/14(retrieval/classification 기본값), SAM 3 encoder(segmentation), SigLIP(vision-language), ConvNeXt(latency-critical). 한 문장 이유를 덧붙이세요.
2. 패치 크기. 224의 표준 classification에는 16, DINOv2에는 14, 고해상도 dense prediction에는 8을 사용합니다. sequence length `(H/P)^2 + 1`과 attention cost `O(N^2)`를 표시하세요.
3. 사전학습 소스. 체크포인트 이름입니다. 작은 labeled set(<10k)에는 DINOv2 features frozen + linear probe를 사용합니다. >100k에는 마지막 block을 fine-tune합니다. 이유를 설명하세요.
4. 학습 레시피. Optimizer(AdamW), lr, augmentations(RandAug, MixUp, Random Erasing), label smoothing(보통 0.1), EMA를 제시하세요.
5. 위험 메모. Data regime risk(full fine-tune에는 데이터가 너무 적음), resolution mismatch(position interpolation 없이 pretrain 224 → deploy 1024), register-token absence(DINOv2 features를 해칠 수 있음)를 적으세요.

이미지가 1M 미만이면 ViT를 scratch부터 학습하라는 추천을 거부하세요. CNN baseline이 이깁니다. Flash Attention + hierarchical variants(Swin)에 대한 명시적 논의 없이 sequence length > 4096이 되는 patch size 추천을 거부하세요. positional embeddings를 보간하지 않고 입력 해상도를 바꾸는 배포는 모두 표시하세요.
