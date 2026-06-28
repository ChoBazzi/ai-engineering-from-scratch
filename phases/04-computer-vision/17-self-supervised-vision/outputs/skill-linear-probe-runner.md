---
name: skill-linear-probe-runner
description: 임의의 고정된 인코더와 라벨 데이터셋을 위한 완전한 linear-probe 평가 작성
version: 1.0.0
phase: 4
lesson: 17
tags: [self-supervised, evaluation, linear-probe, pytorch]
---

# Linear Probe Runner

고정된 인코더 위에 단일 선형 분류기를 학습해 인코더의 특징을 평가합니다. 모든 자기지도 논문의 표준 평가입니다.

## 사용 시점

- 자기지도 checkpoint를 비교할 때.
- 사전학습 epoch에 따른 특징 품질을 추적할 때.
- 파인튜닝 없이 pretrained encoder가 다운스트림 작업에 충분히 좋은지 판단할 때.

## 입력

- `encoder`: 이미지마다 고정 차원 특징을 반환하는 frozen `nn.Module`.
- `feature_dim`: 인코더 출력의 차원 수.
- `train_dataset`: 라벨 데이터셋(image, class_id).
- `val_dataset`: held-out set.
- `num_classes`: 작업 클래스 수.
- `epochs`: 보통 ImageNet-scale에서는 100, 더 작은 데이터셋에서는 50.

## 단계

1. 인코더를 eval mode로 설정하고 모든 parameter에 `requires_grad=False`를 둡니다.
2. train set과 val set에서 특징을 한 번씩 추출합니다. numpy array 또는 memory-mapped file로 저장합니다.
3. 캐시된 특징 위에서 SGD + cosine schedule로 `nn.Linear(feature_dim, num_classes)`를 학습합니다.
4. 표준 hyperparameter: `lr=0.1`, `momentum=0.9`, `weight_decay=0`, `batch_size=1024`. Linear probe는 의외로 `lr`에 민감합니다. 정확도가 낮으면 sweep하세요.
5. 학습이 끝나면 val의 top-1 정확도를 보고합니다.

## 출력 템플릿

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def extract(encoder, loader, device="cpu"):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x.to(device)).cpu()
            feats.append(f)
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(encoder, feature_dim, train_loader, val_loader,
                 num_classes, epochs=50, lr=0.1, device="cpu"):
    for p in encoder.parameters():
        p.requires_grad = False

    f_train, y_train = extract(encoder, train_loader, device)
    f_val, y_val = extract(encoder, val_loader, device)

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt = SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    ds = torch.utils.data.TensorDataset(f_train, y_train)
    train_iter = DataLoader(ds, batch_size=1024, shuffle=True)

    best_val = 0.0
    for ep in range(epochs):
        head.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            acc = (head(f_val.to(device)).argmax(-1).cpu() == y_val).float().mean().item()
        best_val = max(best_val, acc)
    return best_val
```

## 보고

```text
[linear probe]
  encoder:     <name + pretrain checkpoint>
  feature_dim: <int>
  epochs:      <int>
  best_val_top1: <float>
```

## 규칙

- Linear probe 중에는 인코더 가중치를 절대 업데이트하지 마세요. 그것은 probe가 아니라 fine-tune입니다.
- 특징은 한 번만 미리 계산하세요. 매 epoch마다 인코더를 다시 실행하면 compute를 100배 낭비합니다.
- Cosine schedule과 weight decay 없는 SGD를 사용하세요. Adam은 여기서 종종 성능이 낮습니다.
- 인코더 계열마다 learning rate를 최소 한 번은 sweep하세요. 최적값은 SSL 방법마다 달라집니다.
