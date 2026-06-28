---
name: skill-vit-patch-and-pos-embed-inspector
description: ViT의 patch embedding과 positional embedding shape가 모델의 예상 sequence length와 일치하는지 검증
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT 패치 및 위치 임베딩 검사기

가장 흔한 ViT 포팅 버그는 224x224에서 사전학습된 checkpoint를 384x384로 설정된 모델에 로드하는 것(또는 그 반대)입니다. positional embedding의 sequence length가 틀리고, 모델은 조용히 쓰레기 같은 출력을 냅니다.

## 사용 시점

- 기본이 아닌 해상도에서 사전학습된 ViT를 파인튜닝할 때.
- ViT-B/16과 ViT-B/32 사이의 weight port가 실패하는 이유를 점검할 때. 검사기는 patch-size mismatch를 표시해, 호출자가 포트를 강제하기보다 아키텍처를 바꿔야 함을 알게 합니다.
- 오류 없이 로드되지만 학습이 나쁜 ViT를 디버깅할 때.

## 입력

- `model`: 인스턴스화된 ViT `nn.Module`.
- `expected_image_size`: 프로덕션에서 모델이 보게 될 H x W.
- `patch_size`: 예상 patch size.

## 단계

1. 모델 내부의 patch embedding conv를 찾습니다. `kernel_size`, `stride`, `in_channels`, `out_channels`를 보고합니다.
2. 예상 패치 수를 계산합니다. 정사각형 이미지: `(image_size / patch_size)^2`. 직사각형: `(H / patch_size) * (W / patch_size)`. `H % patch_size == 0`와 `W % patch_size == 0`를 요구합니다. 그렇지 않으면 표시하고 거부합니다.
3. 학습된 positional embedding을 찾습니다. shape `(1, N, dim)`을 보고합니다.
4. `N`을 `num_patches + 1`(CLS 있음) 또는 `num_patches`(CLS 없음)와 비교합니다. 불일치는 checkpoint가 다른 해상도나 patch size에서 사전학습되었음을 의미합니다.
5. patch conv의 `out_channels`가 positional embedding의 `dim`과 같은지 확인합니다.
6. 모델이 새 해상도에 대해 positional embedding을 interpolate해야 한다면, interpolation utility가 존재하는지 확인합니다(대부분의 `timm` ViT는 `resize_pos_embed`를 통해 자동으로 처리합니다).

## 보고서

```text
[vit-inspector]
  image_size:         HxW
  patch_size:         <int>
  num_patches (computed): <int>
  patch_conv:         k=<int>  s=<int>  in=<int>  out=<int>
  pos_embed shape:    (1, N, dim)
  has CLS token:      yes | no
  pos_embed N:        <int>    expected: <int>
  verdict:            ok | mismatch

[if mismatch]
  action:  reinitialise pos_embed for new sequence length
  tool:    timm.models.vision_transformer.resize_pos_embed
```

## 규칙

- 경고 없이 조용히 interpolate하지 않습니다. pretrained positional structure가 바뀌었을 수 있음을 사용자가 알 수 있도록 동작을 드러냅니다.
- patch_size가 일치하지 않으면 interpolation 추천을 거부합니다. 올바른 아키텍처로 바꿔야 합니다.
- 모델을 제자리에서 고치려 하지 않습니다. 보고하고 제안만 합니다.
