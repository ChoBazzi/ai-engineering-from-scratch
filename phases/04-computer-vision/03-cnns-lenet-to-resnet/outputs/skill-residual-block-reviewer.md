---
name: skill-residual-block-reviewer
description: skip-connection correctness, BN placement, activation order, shape alignment 관점에서 PyTorch residual block을 리뷰한다
version: 1.0.0
phase: 4
lesson: 3
tags: [computer-vision, resnet, code-review, pytorch]
---

# 잔차 블록 리뷰어

residual block을 구현한다고 주장하는 모든 PyTorch `nn.Module`을 위한 집중 리뷰어입니다. 거의 모든 깨진 ResNet 재작성에서 나타나는 네 가지 실수를 잡아냅니다.

## 사용 시점

- 누군가 custom BasicBlock 또는 Bottleneck을 작성했는데 loss가 NaN이거나 accuracy가 멈춰 있을 때.
- 한 framework에서 다른 framework로 block을 porting하면서 동등성을 검증하고 싶을 때.
- ResNet 내부(pre-activation, squeeze-excite, anti-alias)를 바꾸는 PR을 리뷰할 때.
- model이 CIFAR 크기 입력에서는 잘 배포되지만 shortcut이 잘못되어 ImageNet resolution에서 crash할 때.

## 입력

- PyTorch class definition. source text 또는 importable path 모두 가능.
- 선택적 `variant`: `basic` | `bottleneck` | `preact` | `seblock`.

## 네 가지 검사

### 1. Shortcut shape 정렬

`stride != 1`이거나 `in_channels != out_channels`인 모든 block에서 shortcut path는 **반드시** shape-matching module이어야 합니다. 보통 1x1 conv plus BN입니다. 이 경우 bare `nn.Identity()`는 forward time의 shape-mismatch error를 보장합니다.

진단:
```text
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy
```

### 2. Addition 대비 BN 배치

Addition `out + shortcut(x)`는 final ReLU보다 **전에** 일어나야 합니다(post-activation, original ResNet). 또는 final ReLU가 완전히 없어야 합니다(pre-activation ResNet v2). main branch에서 ReLU를 적용한 뒤 raw shortcut을 더하는 block은 비대칭 activation range를 만들어 학습을 해칩니다.

진단:
```text
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect
```

### 3. Conv layers의 bias

BatchNorm이 바로 뒤따르는 conv는 `bias=False`여야 합니다. BN의 beta가 이미 bias를 parameterise하므로 추가 conv bias는 파라미터를 낭비하고 수렴을 늦출 수 있습니다.

진단:
```text
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers
```

### 4. In-place ReLU와 autograd

shortcut에 더해질 tensor에 대한 `nn.ReLU(inplace=True)`는 residual add에 여전히 필요할 수 있는 값을 덮어씁니다. add 전에 새 tensor를 만드는 layer가 뒤따르지 않는 모든 `inplace=True`를 flag하세요.

진단:
```text
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add
```

## 리포트

```text
[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence
```

## 규칙

- block을 다시 작성하지 마세요. 보고만 하세요.
- block이 올바르면 모든 곳에 `ok`라고 말하고 멈추세요. 제안하지 마세요.
- 여러 문제가 있으면 위 순서대로 나열하세요(shortcut이 crash의 가장 흔한 원인이므로 먼저).
- 사용자가 deliberate pre-activation 또는 squeeze-excite variant라고 명시했다면 절대 wrong으로 flag하지 마세요.
