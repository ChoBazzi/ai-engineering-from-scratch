---
name: text-encoder-picker
description: 주어진 constraint set에 맞는 text encoder architecture를 고릅니다.
phase: 5
lesson: 08
---

constraint(task, data volume, latency budget, deploy target, compute budget)가 주어지면 다음을 출력합니다.

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, 또는 "pretrained transformer as frozen encoder + small head".
2. Embedding input: random init, frozen GloVe 또는 fastText, contextualized transformer embeddings.
3. 5줄짜리 training recipe: optimizer, learning rate, batch size, epochs, regularization.
4. 하나의 monitoring signal. RNN/CNN model: long-dependency failure를 찾기 위해 per-sequence-length accuracy를 확인합니다. Transformer fine-tune: LR이 너무 높을 때 fine-tuning collapse가 생기는지 감시하고, 첫 100 step 안의 train loss를 확인합니다.

사용자에게 labeled example이 약 500개 미만일 때, TextCNN / BiLSTM baseline이 plateau했다는 것을 먼저 보이지 않고 transformer fine-tuning을 추천하지 않습니다. Edge deployment(phone, microcontroller, browser)는 무엇보다 먼저 architecture decision이 필요하다고 표시합니다.
