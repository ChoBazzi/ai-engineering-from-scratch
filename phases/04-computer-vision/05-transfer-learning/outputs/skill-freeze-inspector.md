---
name: skill-freeze-inspector
description: 어떤 parameter가 trainable인지, 어떤 BatchNorm layer가 eval mode인지, optimizer가 실제로 trainable parameter를 소비하는지 보고한다
version: 1.0.0
phase: 4
lesson: 5
tags: [computer-vision, transfer-learning, debugging, pytorch]
---

# Freeze Inspector

전이 학습 버그는 세 곳에 숨어 있습니다. 고정되어야 하지만 고정되지 않은 parameter, 학습 가능해야 하지만 그렇지 않은 parameter, 그리고 freeze state가 바뀌기 전에 만들어진 optimizer입니다. 이 skill은 셋을 한 번에 드러냅니다.

## 사용할 때

- parameter 일부에 `requires_grad`를 설정한 직후.
- fine-tune 실행의 첫 training step 전에.
- `freeze_bn_stats` 또는 BN mode를 바꾸는 helper를 호출한 뒤.
- val accuracy가 random에 머물러 있고 실제로 아무것도 학습되지 않는다고 의심될 때.

## 입력

- `model`: PyTorch `nn.Module`.
- `optimizer`: training에 곧 사용할 optimizer.
- Optional `expected_frozen_prefixes`: 고정되어야 하는 parameter-name prefix 목록(예: `["conv1", "bn1", "layer1"]`).

## 단계

1. **parameter를 순회합니다.** 각 `(name, param)`에 대해:
   - `requires_grad`를 기록합니다
   - `shape`와 `numel`을 기록합니다

2. **module을 순회합니다.** 각 module에 대해:
   - BatchNorm이면 eval mode인지, affine parameter가 trainable인지 기록합니다.

3. **optimizer를 검사합니다.** 각 parameter group에 대해:
   - `params`를 `id(p)` set으로 flatten합니다.
   - `requires_grad == True`인 모든 params의 `id(p)` set과 비교합니다.

4. **네 가지 failure mode를 감지합니다.**
   - `leaked_train`: parameter가 `requires_grad=True`지만 optimizer에 나타나지 않습니다(gradient는 계산되지만 적용되지 않음).
   - `ghost_train`: parameter가 optimizer에 있지만 `requires_grad=False`입니다(optimizer state가 낭비됨. 나중에 requires_grad를 다시 켜면 버그도 유발할 수 있음).
   - `bn_mismatch`: (a) BN layer가 train mode(running stats 누적)이면서 affine parameter(`weight`, `bias`)가 frozen이거나, (b) BN layer가 eval mode(frozen stats)이면서 affine parameter가 trainable입니다. 두 상태 모두 일관되지 않으며 거의 항상 버그입니다.
   - `expected_vs_actual`: `expected_frozen_prefixes`에 나열된 prefix 중 trainable parameter가 아직 남아 있습니다.

## 보고서

```text
[freeze-inspector]
  model trainable params: <N>
  model frozen params:    <N>
  batchnorm layers in eval mode: <count>
  batchnorm layers in train mode: <count>

[optimizer coverage]
  trainable params fed to optimizer: <M> of <N>
  leaked_train: <list of names> (trainable but not in optimizer)
  ghost_train:  <list of names> (in optimizer but frozen)

[bn audit]
  mismatched layers: <list of names>

[expectations]
  expected_frozen_prefixes: <...>
  violating params:         <list>

[verdict]
  ok | <one-line summary of the most severe issue>
```

## 규칙

- parameter name만 보고하고 weights 자체는 절대 출력하지 마세요.
- 모든 list를 parameter name 기준 알파벳순으로 정렬하세요.
- optimizer coverage가 100%이고 mismatch가 없으면 `ok`를 반환하고 멈추세요.
- `leaked_train`에 대해서는 freeze state가 바뀐 뒤 optimizer를 다시 만들라고 항상 권장하세요.
- `ghost_train`에 대해서는 parameter group을 제거하거나, 의도가 학습이었다면 `requires_grad=True`로 설정하라고 권장하세요.
