---
name: skill-conv-shape-calculator
description: CNN spec을 layer별로 따라가며 각 block의 output shape, receptive field, parameter count를 보고합니다
version: 1.0.0
phase: 4
lesson: 2
tags: [computer-vision, cnn, architecture, debugging]
---

# Conv Shape 계산기

CNN을 계획하거나 디버깅하기 위한 결정적 helper입니다. input shape와 layer spec 목록이 주어지면 모델을 실행하지 않고 shape, receptive field, parameter count를 추적합니다.

## 언제 사용할까

- 새 CNN을 설계하면서 모든 downsample이 깔끔한 size로 떨어지는지 검증하고 싶을 때.
- 논문을 읽고 architecture table을 code로 옮길 때.
- 사전 학습 backbone이 classifier head에서 shape mismatch로 실패했고, 어떤 layer가 spatial size를 바꿨는지 알아야 할 때.
- 두 backbone을 학습하기 전에 parameter efficiency를 비교할 때.

## 입력

- `input_shape`: `(C, H, W)`.
- `layers`: 순서가 있는 layer dict 목록입니다. 각 항목은 다음을 지원합니다.
  - `{type: "conv", c_out, k, s, p, groups=1, bias=true}`
  - `{type: "pool", mode: "max"|"avg", k, s, p=0}`
  - `{type: "adaptive_pool", out_h, out_w}`
  - `{type: "flatten"}`
  - `{type: "linear", out_features, bias=true}`

## 절차

1. `(C, H, W)`, receptive field `1`, effective stride `1`, cumulative params `0`으로 **trace를 초기화**합니다.

2. **각 layer마다** 다음 순서로 갱신합니다.
   - `C_out`(conv/linear)을 계산하거나, pool에서는 `C_in`을 그대로 전달합니다.
   - conv와 pool은 `(H + 2P - K) / S + 1`, adaptive pool은 `out_h/out_w`, linear 전 flatten output shape는 `(C * H * W, 1, 1)`에 대한 `(1, 1)`, linear는 scalar `1x1`로 spatial output을 계산합니다.
   - receptive field와 effective stride를 갱신합니다.
     - Conv/pool: `RF_new = RF_old + (K - 1) * effective_stride`, `effective_stride *= S`.
     - Adaptive pool: effective `S = H_in / out_h`(내림)를 갖는 pool로 취급합니다. `RF_new = RF_old + (H_in - 1) * effective_stride_old`; `effective_stride *= S`. adaptive pool의 RF는 이전 spatial extent 전체와 같다는 점에 주의하세요.
     - Flatten / linear: RF와 effective stride는 더 이상 의미가 없습니다. flatten 전 값으로 고정하고 이후 row에서는 생략합니다.
   - params를 계산합니다.
     - Conv: `C_out * (C_in / groups) * K * K + (C_out if bias else 0)`.
     - Linear: `out_features * in_features + (out_features if bias else 0)`.
     - Pool과 flatten: 0.

3. **문제를 감지**하고 표시합니다.
   - 정수가 아닌 output size(misaligned stride/padding).
   - stack 끝에 도달하기 전 `H_out <= 0`.
   - input size를 넘는 receptive field(그 이후 compute 낭비 가능).
   - 잘못된 channel plan을 시사하는 layer별 params의 갑작스러운 10x 증가.

4. **보고서**를 단일 table로 작성합니다.

```text
idx  layer                C_in  C_out  K  S  P  H_out  W_out  RF    params     cum_params
1    conv 3x3 s=1 p=1     3     32     3  1  1  224    224    3     896        896
2    conv 3x3 s=2 p=1     32    64     3  2  1  112    112    7     18,496     19,392
3    pool max 2x2         64    64     2  2  0  56     56     11    0          19,392
...
```

5. **summary line**: final `(C, H, W)`, final receptive field, total params, warnings.

## 규칙

- spatial size는 항상 정수로 반환합니다. 공식이 정수가 아닌 값을 만들면 error로 표시하고 조용히 floor하지 마세요.
- `groups > 1`이면 `C_in % groups == 0`과 `C_out % groups == 0`을 검증합니다. 아니면 error입니다.
- depthwise conv(`groups == C_in`)는 reader가 params가 낮은 이유를 볼 수 있도록 `layer` column에 표시합니다.
- 사용자가 BatchNorm이나 activation layer를 제공하면 shape 관점에서는 무시하되 params는 계속 누적합니다(BatchNorm마다 `2 * C`).
- 누락된 field의 기본값을 절대 추측하지 마세요. 모든 conv와 pool에 `k`, `s`, `p`를 요구합니다.
