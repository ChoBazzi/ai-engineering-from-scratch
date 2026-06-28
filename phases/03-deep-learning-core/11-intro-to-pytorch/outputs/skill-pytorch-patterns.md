---
name: skill-pytorch-patterns
description: PyTorch 학습, 평가, 배포를 위한 reference pattern
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, training, deep-learning, gpu, patterns]
---

## 표준 학습 루프

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
```

## Mixed Precision 학습

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

사용할 때: float16을 지원하는 하드웨어(V100, A100, H100, RTX 3090+)에서 GPU 학습을 할 때. 약 1.5-2배 속도 향상과 약 50% 메모리 감소를 기대할 수 있습니다.

## 그래디언트 누적

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

사용할 때: effective batch size가 GPU 메모리가 허용하는 것보다 커야 할 때. loss를 accumulation_steps로 나누면 그래디언트 scale이 일관되게 유지됩니다.

## 저장과 로드

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

학습을 재개하려면 항상 optimizer state를 저장하세요. 추론 전용이라면 `model.state_dict()`만 저장합니다.

## 사용자 정의 Dataset

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = self._load_samples(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load_samples(self, data_dir):
        ...
```

## DataLoader 설정

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
```

| 파라미터 | 하는 일 | 사용할 때 |
|-----------|-------------|-------------|
| num_workers=4 | 병렬 데이터 로딩 | 멀티코어 머신에서는 항상 |
| pin_memory=True | Page-locked CPU 메모리 | GPU에서 학습할 때 |
| drop_last=True | 불완전한 마지막 배치 버리기 | BatchNorm을 사용할 때 |
| persistent_workers=True | epoch 사이에도 worker 유지 | num_workers > 0일 때 |

## 학습률 스케줄

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,
)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        ...
        optimizer.step()
        scheduler.step()
```

OneCycleLR: 대부분의 작업에서 가장 좋은 기본값입니다. max_lr까지 warm up한 다음 cosine decay를 적용합니다. `scheduler.step()`은 매 epoch가 아니라 매 batch 뒤에 호출하세요.

## 가중치 초기화

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)
```

## 추론 모드

```python
model.eval()

with torch.inference_mode():
    outputs = model(inputs)
```

`torch.inference_mode()`는 그래디언트 계산만 억제하는 것이 아니라 autograd 전체를 비활성화하므로 `torch.no_grad()`보다 빠릅니다.

## 흔한 실수 체크리스트

1. CrossEntropyLoss 전에 softmax를 적용함(내부에 log_softmax가 포함되어 있음)
2. validation 중 model.eval() 호출을 잊음
3. 텐서를 모델과 같은 device로 옮기는 것을 잊음
4. optimizer.zero_grad()를 호출하지 않음(그래디언트는 기본적으로 누적됨)
5. 학습 중 torch.no_grad()를 사용함(그래디언트 계산을 비활성화함)
6. num_workers를 너무 높게 설정함(프로세스를 너무 많이 만들어 메모리를 thrash함)
7. GPU 학습 중 pin_memory=True를 사용하지 않음
8. state_dict 대신 전체 모델 객체를 저장함(리팩터링 시 깨짐)
