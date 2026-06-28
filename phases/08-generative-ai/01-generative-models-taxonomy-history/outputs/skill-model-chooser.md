---
name: generative-model-chooser
description: 주어진 작업과 예산에 맞는 생성 모델 계열, backbone, hosted 대안을 고릅니다.
version: 1.0.0
phase: 8
lesson: 01
tags: [generative, taxonomy]
---

작업 설명(modality, domain, latency budget, compute budget, conditioning signal)이 주어지면 다음을 출력하세요.

1. 계열. Explicit-tractable, explicit-approximate(VAE / diffusion), implicit(GAN), score / flow matching, token-AR 중 하나. modality와 latency에 연결한 한 문장 이유를 함께 제시합니다.
2. Backbone + open reference. 사용자가 오늘 fine-tune할 수 있는 open-weights pretrained 모델 하나를 제시합니다(예: Stable Diffusion 3, Flux.1-dev, AudioCraft 2, StyleGAN3, 3D Gaussian Splatting).
3. Hosted 대안. 품질 / 비용 / latency trade-off 기준으로 production API 세 개를 순위화합니다(fal.ai, Replicate, Stability, Runway, Veo, Kling, ElevenLabs 등).
4. 실패 모드. 선택한 계열의 알려진 병리(mode collapse, exposure bias, sampler drift, tokenizer artifacts, CLIP-score gaming)를 설명합니다.
5. 예산. 단일 A100 기준 대략적인 training 시간, sample당 inference 비용, VRAM 하한을 제시합니다.

작업에 likelihood scoring이 필요하면 GAN 추천을 거부하세요. 고해상도 real-time 용도에는 autoregressive-over-pixels 추천을 거부하세요. 나열된 open backbone이 이미 domain을 포괄한다면 "train from scratch" 추천에는 경고를 표시하세요.
