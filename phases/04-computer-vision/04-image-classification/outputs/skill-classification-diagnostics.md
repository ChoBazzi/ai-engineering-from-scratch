---
name: skill-classification-diagnostics
description: confusion matrix와 class names가 주어졌을 때 per-class failures를 드러내고 가장 영향력 있는 fix 하나를 제안한다
version: 1.0.0
phase: 4
lesson: 4
tags: [computer-vision, classification, evaluation, debugging]
---

# 분류 진단

confusion matrix를 읽기 위한 lens입니다. Aggregate accuracy는 classifier가 작동한다고 말해줍니다. Confusion matrix는 classifier가 *아직 무엇을 모르는지* 말해줍니다.

## 사용 시점

- 학습된 classifier의 validation performance를 처음 살펴볼 때.
- training run 사이에 다음에 무엇을 바꿀지 결정할 때.
- model을 shipping하기 전: critical class가 조용히 실패하지 않는지 검증할 때.
- overall accuracy가 1 point 떨어진 production regression을 debugging하면서 이유를 알아야 할 때.

## 입력

- `cm`: CxC confusion matrix(rows = true, cols = predicted).
- `labels`: 같은 순서의 C class names 목록.
- 선택적 `class_priors`: per-class training frequency(기본값은 `cm`의 row sums).

## 단계

1. **Per-class metrics를 계산합니다.** 0으로 나누는 경우는 해당 class에서 metric이 undefined인 것으로 처리하고 `n/a`로 보고하세요. 조용히 0으로 대체하지 마세요.
   - precision_i = cm[i,i] / sum(cm[:, i])   (class가 한 번도 predicted되지 않았을 때 undefined)
   - recall_i    = cm[i,i] / sum(cm[i, :])   (class에 ground-truth samples가 없을 때 undefined)
   - f1_i        = 2 * p * r / (p + r)        (둘 중 하나가 undefined이면 undefined)

2. F1 기준으로 **최대 세 개의 worst classes를 rank**합니다. confusion matrix에 class가 세 개보다 적으면 존재하는 만큼만 rank하세요. 모든 metric이 undefined인 class는 제외합니다.

3. **각 row의 top off-diagonal cell**을 찾습니다. 이 class에서 가장 흔히 빼앗아 가는 하나의 class입니다. `true -> predicted`로 보고하세요.

4. 각 worst class의 **failure mode를 분류**합니다. label이 재현 가능하도록 다음 quantitative thresholds를 사용하세요.
   - `ambiguity` — 다른 class와 bidirectional confusion: `cm[i,j] / sum(cm[i, :]) >= 0.15`이고 `cm[j,i] / sum(cm[j, :]) >= 0.15`.
   - `imbalance` — 해당 class의 training count가 top confuser의 `< 0.5x`.
   - `label_noise` — `|precision_i - recall_i| >= 0.2`이고 class가 imbalance / ambiguity paths에 있지 않음.
   - `systematic` — 이 class errors에서 0.2 share를 넘는 single confuser가 없음. 오류가 세 개 이상의 다른 class로 퍼져 있음.

5. **가장 영향력 있는 다음 action 하나를 추천**합니다.
   - `ambiguity` -> discriminative examples를 collect 또는 synthesise하고, distinguishing feature를 보존하는 targeted augmentation을 추가합니다.
   - `imbalance` -> minority class를 oversample하거나 class-weighted loss를 적용합니다.
   - `label_noise` -> 해당 class의 stratified sample을 audit합니다. 다른 변경 전에 mislabels를 고칩니다.
   - `systematic` -> 해당 class의 data를 늘리거나 이 class loss에 더 높은 weight를 두고 fine-tune합니다.

## 리포트

```text
[diagnostics]
  aggregate accuracy: X.XX
  macro F1:           X.XX

[top-3 worst classes]
  1. class <name>  F1 = X.XX  prec = X.XX  rec = X.XX
     top confusion: <name> -> <other>  (N cases)
     failure mode:  ambiguity | imbalance | label_noise | systematic
     action:        <one sentence>

  2. ...
  3. ...

[recommendation]
  single biggest lever: <one sentence naming the class and the fix>
```

## 규칙

- 최대 세 개의 class만 반환하세요. 더 많으면 signal이 가려집니다.
- 각 worst class의 dominant confuser를 말하세요. 절대 "confuses with many"라고 요약하지 마세요.
- 모든 recommendation을 confusion matrix evidence에 근거시키세요. 어떤 class인지 명시하지 않는 generic "add more data"는 금지합니다.
- precision과 recall이 0.2보다 많이 다르면 label noise를 항상 candidate로 flag하세요. 실제 class는 보통 학습 후 P와 R이 정렬됩니다.
