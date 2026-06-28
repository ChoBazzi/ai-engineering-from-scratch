---
name: skill-lora-training-setup
description: Captions, rank, batch size, learning rate를 포함해 custom dataset용 전체 LoRA training config를 작성한다
version: 1.0.0
phase: 4
lesson: 11
tags: [computer-vision, stable-diffusion, lora, fine-tuning]
---

# LoRA Training Setup

Fine-tune 의도에 대한 설명을 `diffusers` 또는 `kohya_ss`에 전달할 준비가 된 구체적인 training config로 바꾼다.

## 사용 시점

- Subject(인물, object, character), style(artist, brand), concept(pose, lighting)에 대한 LoRA를 학습할 때.
- 기존 LoRA를 더 많은 data로 확장할 때.
- 출력이 training image에 underfit 또는 overfit되는 LoRA run을 debugging할 때.

## 입력

- `purpose`: subject | style | concept
- `num_images`: 사용할 수 있는 training image 수
- `base_model`: SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`: 8 | 12 | 16 | 24 | 48+
- `caption_source`: manual | BLIP2-generated | dataset-native

## Rank picker

| Purpose | Rank | Alpha |
|---------|------|-------|
| Subject | 8-16 | rank |
| Style | 16-32 | rank * 2 |
| Concept | 32-64 | rank |

Rank가 높을수록 capacity가 크지만 작은 dataset에서는 overfitting 위험도 커진다. Alpha는 LoRA 효과 강도를 scale한다. `alpha == rank`가 안전한 기본값이다. Style은 문서화된 예외다. `alpha == rank * 2`는 subject fidelity가 목표가 아닐 때만 사용하며, style을 너무 강하게 bake할 위험을 감수하는 대신 더 강한 style push를 준다.

## Training step target

- 이미지 5-20장의 `subject`: 500-1500 steps.
- 이미지 30-100장의 `style`: 1500-4000 steps.
- 이미지 100+장의 `concept`: 4000-10000 steps.

지나치게 많이 학습하지 말라. Training image를 memorise한 LoRA는 generalise할 수 없다.

## Learning rate

- Text encoder LoRA: SD 1.5는 `1e-4`, SDXL은 `5e-5`.
- U-Net LoRA: SD 1.5는 `1e-4`, SDXL은 `1e-4`.
- FLUX / SD3: transformer는 `5e-5`, text encoder는 보통 고정한다.
- `num_images < 15`(subject)이거나 3000 step보다 오래 학습할 때는 LR을 절반으로 낮춘다. 아주 작은 dataset과 긴 run은 둘 다 더 부드러운 update가 유리하다.

## Scheduler

- `cosine_with_warmup`(기본값): 처음 5-10% step 동안 warmup하고 이후 cosine decay한다. `steps >= 1000`일 때 사용한다. Decay tail은 더 선명한 최종 sample을 만든다.
- `constant`: 매우 짧은 run(`steps < 500`)이거나, 현재 학습된 feature를 re-annealing 없이 보존하고 싶은 이전 LoRA를 resume할 때만 사용한다.

## Caption format

- Subject: 모든 caption 앞에 고유 trigger token("myperson")을 붙인다. 기존 concept을 overwrite하지 않도록 trigger token을 rare하게 유지한다. 실제 단어와 흔한 이름은 피한다.
- Style: 모든 caption 끝에 고유 style tag("...in mystyle style")를 붙인다. Tag 자체를 rare trigger token으로 취급한다. 이미 실제 concept에 매핑되는 `impressionism`이 아니라 `mystyle`을 사용한다.
- Concept: 모든 caption에서 concept을 설명한다. Trigger token은 없다. Concept 자체(예: "low-angle shot")가 anchor다.

## Output config

```yaml
model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # and/or unet.to_q, to_k, to_v, to_out

training:
  steps:          <int>
  batch_size:     <int, tuned to gpu_vram_gb>
  grad_accum:     <int, usually 1 on >=16 GB, 4 on <=12 GB>
  learning_rate:  <float>
  optimizer:      AdamW8bit | AdamW
  scheduler:      cosine_with_warmup | constant
  warmup_steps:   <int>
  save_every:     <int>

data:
  images_dir:     <path>
  caption_source: <manual | BLIP2 | native>
  trigger_token:   <string if purpose==subject>
  resolution:      <512 for SD 1.5, 1024 for SDXL>
  aspect_ratio_bucketing: true
  augmentation:
    flip:          true
    color_jitter:  false

validation:
  prompts:
    - "<trigger> ...test prompt..."
    - "<trigger> in a different scene"
  every_steps: 250
```

## Report

```text
[lora setup]
  purpose:   <subject|style|concept>
  base:      <model>
  rank:      <int>
  steps:     <int>
  batch:     <int>   grad_accum: <int>
  lr:        <float>
  vram est.: <float> GB
```

## 규칙

- `rank > 64`를 절대 추천하지 않는다. 그 이상이면 LoRA가 mini fine-tune이 되어 "adapter" 성격을 잃는다.
- `num_images < 5`이면 강하게 경고한다. 1-3장으로 학습한 identity LoRA는 매번 overfit된다.
- `gpu_vram_gb < 12`이면 AdamW8bit와 gradient checkpointing을 요구한다.
- `base_model == FLUX`이고 `gpu_vram_gb < 24`이면 `schnell` variant로 route하고 학습이 더 느리다고 알린다.
- Validation prompt를 절대 생략하지 않는다. Sample grid 없는 LoRA는 평가할 수 없다.
