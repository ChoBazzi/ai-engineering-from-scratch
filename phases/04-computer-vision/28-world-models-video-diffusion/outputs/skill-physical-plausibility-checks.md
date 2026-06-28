---
name: skill-physical-plausibility-checks
description: 출시 전 생성된 모든 video에 대해 object permanence, gravity, continuity를 자동으로 점검한다
version: 1.0.0
phase: 4
lesson: 28
tags: [video-generation, quality, physics, evaluation]
---

# Physical Plausibility Checks

생성 video의 production deployment에는 automated guardrail이 필요하다. Human review는 확장되지 않는다. Physics check는 고전적인 failure mode를 잡아낸다.

## 언제 사용하는가

- Text prompt 또는 image prompt에서 video를 생성하는 모든 product.
- Video generation API endpoint의 QA를 자동화할 때.
- Fine-tuning 또는 base-model update 이후 video model의 quality drift를 monitoring할 때.

## 입력

- `video`: tensor `(T, H, W, 3)` 또는 mp4 경로.
- 선택 reference info: 예상 object 수, initial scene description.

## 점검

### 1. Object permanence

SAM 3.1 Object Multiplex로 모든 detection을 frame에 걸쳐 추적한다. Stable track이 <=3 frame 동안 사라졌다가 다시 나타나면 flag한다. 이는 model이 object를 일시적으로 잃었다는 뜻이다. Object가 frame center 근처에서 사라지면 hard fail(가장자리라면 아님), edge에서는 soft fail한다.

### 2. Motion smoothness

연속 frame 사이의 optical flow는 대부분 continuous해야 한다. 갑작스러운 per-pixel flow spike는 teleportation을 뜻한다. RAFT로 flow를 계산한다. 99th-percentile flow magnitude가 median보다 10배를 초과하는 frame을 flag한다.

### 3. Gravity / support

Solid로 검출된 object(food, balls, tools)는 lifting action이 없을 때 vertical position이 non-increasing이어야 한다. Object 근처에서 "grasping hand"가 검출되지 않았는데 upward drift가 있으면 flag한다.

### 4. Identity consistency

사람이나 character에는 frame에 걸쳐 face-recognition embedding을 사용한다. Persistent identity라면 5-frame window에서 cosine similarity가 0.8보다 커야 한다. Threshold 아래는 character가 morphed했다는 뜻이다.

### 5. Hands and limbs

Pose estimator(Lesson 21)를 실행한다. 손에 보이는 finger가 > 5 또는 < 4인 frame, arm length가 frame 사이에서 두 배가 되는 frame, limb가 surface를 통해 body와 교차하는 frame을 flag한다.

### 6. Text rendering (prompt가 text를 요청한 경우)

User prompt에 따옴표 안의 string이 포함되어 있다면 generated frame에 OCR을 실행하고 requested string과 CER을 계산한다. CER > 20%를 flag한다.

## 보고서

```text
[plausibility]
  video frames:           <T>
  permanence violations:  <N>
  smoothness violations:  <N>
  gravity violations:     <N>
  identity drift:         <N of 5-frame windows>
  limb anomalies:         <N>
  OCR CER vs requested:   <float>

[verdict]
  ship | hold | reject

[samples for review]
  frame ranges where each failure occurred
```

## 규칙

- 단일 check 하나로 hard-block하지 않는다. Score를 aggregate하고 total anomaly가 threshold를 넘으면 video를 review 대상으로 hold한다.
- Identity drift와 permanence violation에 가장 높은 weight를 둔다. 사용자가 그것들을 먼저 알아차린다.
- 시간에 따른 check별 failure rate를 log한다. 상승 추세는 보통 base model update 또는 prompt distribution shift를 뜻한다.
- Flag된 video를 절대 삭제하지 않는다. Model debugging과 post-mortem을 위해 보관한다.
- Sensitive content(people, children, public figures)는 score와 관계없이 모든 video에 human review를 요구한다.
