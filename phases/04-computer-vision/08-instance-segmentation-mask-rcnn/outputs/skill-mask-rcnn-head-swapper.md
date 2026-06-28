---
name: skill-mask-rcnn-head-swapper
description: custom num_classes에 맞춰 torchvision Mask R-CNN의 box head와 mask head를 교체하는 정확한 코드를 생성합니다
version: 1.0.0
phase: 4
lesson: 8
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# Mask R-CNN Head Swapper

Mask R-CNN 전용 head-swap boilerplate를 생성합니다. 아래 template은 `model.roi_heads.box_predictor`와 `model.roi_heads.mask_predictor`를 가정하며, 이들은 `maskrcnn_resnet50_fpn`과 `maskrcnn_resnet50_fpn_v2`에만 존재합니다. Faster R-CNN에는 box predictor는 있지만 mask predictor가 없습니다. RetinaNet은 `RetinaNetHead`를 사용하고 `roi_heads`가 전혀 없습니다. 둘 다 다른 skill이 필요합니다.

## 사용할 때

- custom class set에서 `maskrcnn_resnet50_fpn` 또는 `maskrcnn_resnet50_fpn_v2`를 fine-tuning할 때.
- COCO에서 훈련된 Mask R-CNN checkpoint를 non-COCO class count로 옮길 때.
- `cls_score.out_features` 또는 `mask_predictor` mismatch로 crash하는 Mask R-CNN training run을 디버깅할 때.

## 범위 밖

- `fasterrcnn_*` — `mask_predictor`가 없습니다. `box_predictor`만 교체하세요. 별도의 Faster R-CNN head-swap recipe를 사용하세요.
- `retinanet_*` — `roi_heads`가 없습니다. classifier + regression head는 `model.head.classification_head`와 `model.head.regression_head` 아래에 있습니다. RetinaNet 전용 skill을 사용하세요.
- `keypointrcnn_*` — `mask_predictor` 대신 `keypoint_predictor`를 사용합니다.

## 입력

- `model_name`: torchvision detection model constructor, 예: `maskrcnn_resnet50_fpn_v2`.
- `num_classes`: background 포함. 4-object-class dataset은 `num_classes=5`를 뜻합니다.
- `freeze`: `backbone`, `backbone_fpn`, `none` 중 하나.

## 단계

1. model constructor와 두 predictor class(`FastRCNNPredictor`, `MaskRCNNPredictor`)를 import합니다.
2. default-weights pretrained model을 load합니다.
3. `model.roi_heads.box_predictor`를 새 `FastRCNNPredictor(in_features, num_classes)`로 교체합니다.
4. `model.roi_heads.mask_predictor`를 새 `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)`로 교체합니다.
5. 요청된 freeze policy를 적용합니다.
6. module별 trainable params를 나열하는 confirmation block을 출력합니다.

## 출력 코드 template

```python
from torchvision.models.detection import {MODEL_NAME}, {MODEL_WEIGHTS}
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes={NUM_CLASSES}):
    model = {MODEL_NAME}(weights={MODEL_WEIGHTS}.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    {FREEZE_BLOCK}

    return model
```

여기서 `{FREEZE_BLOCK}`는 다음과 같습니다.

- `none` -> 비어 있음
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  ```

## 보고서

```text
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## 규칙

- background를 포함하지 않은 `num_classes`를 절대 권하지 마세요. 항상 사용자에게 상기시키세요.
- 사용 가능한 경우 torchvision detection model의 `_v2` variant를 항상 사용하세요. legacy variant보다 더 나은 pretrained weights를 갖고 있습니다.
- 이 skill 안에서 model을 instantiate하지 마세요. code block을 만들고 사용자가 실행하게 하세요.
- 사용자가 10,000 images보다 큰 dataset에서 `freeze backbone`을 요청하면 backbone도 fine-tuning하는 것을 고려하라고 제안하세요.
