# 텍스트용 CNN과 RNN

> Convolution은 n-gram을 학습합니다. Recurrence는 기억합니다. 둘 다 attention에 의해 대체되었습니다. 하지만 constrained hardware에서는 둘 다 여전히 중요합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 minutes

## 문제

TF-IDF와 Word2Vec은 word order를 무시하는 flat vector를 만들었습니다. 그 위에 만든 classifier는 `dog bites man`과 `man bites dog`를 구분할 수 없었습니다. Word order가 signal을 담는 경우가 있습니다.

Transformer가 등장하기 전에, 두 architecture family가 이 간극을 메웠습니다.

**Convolutional nets for text (TextCNN).** Word embedding sequence 위에 1D convolution을 적용합니다. width 3 filter는 학습 가능한 trigram detector입니다. 세 단어를 가로지르며 score를 출력합니다. 여러 width(2, 3, 4, 5)를 쌓아 multi-scale pattern을 탐지합니다. Max-pool로 fixed-size representation을 만듭니다. Flat하고 parallel하며 빠릅니다.

**Recurrent nets (RNN, LSTM, GRU).** Token을 한 번에 하나씩 처리하면서 정보를 앞으로 운반하는 hidden state를 유지합니다. Sequential하고, memory를 가지며, flexible input length를 다룹니다. 2014년부터 2017년까지 sequence modeling을 지배했고, 그 뒤 attention이 등장했습니다.

이 lesson은 둘 다 만든 뒤, attention을 낳게 한 failure를 이름 붙입니다.

## 개념

**TextCNN** (Kim, 2014). Token을 embedding합니다. width-`k` 1D convolution은 연속된 `k`-gram embedding 위로 filter를 slide하며 feature map을 만듭니다. 그 map 위의 global max-pooling은 가장 강한 activation을 고릅니다. 여러 filter width의 max-pooled output을 concatenate합니다. classifier head에 넣습니다.

왜 동작할까요? Filter는 학습 가능한 n-gram입니다. Max-pooling은 position-invariant이므로, "not good"은 review의 시작에 있든 중간에 있든 같은 feature를 발화시킵니다. 각 100개 filter를 가진 세 filter width는 300개의 학습된 n-gram detector를 제공합니다. Training은 parallel이고 sequential dependency가 없습니다.

**RNN.** 각 time step `t`에서 hidden state는 `h_t = f(W * x_t + U * h_{t-1} + b)`입니다. `W`, `U`, `b`를 시간 전체에서 공유합니다. time `T`의 hidden state는 전체 prefix의 summary입니다. Classification에서는 `h_1 ... h_T` 전체를 pool(max, mean, 또는 last)합니다.

Plain RNN은 vanishing gradient를 겪습니다. **LSTM**은 무엇을 forget하고, 무엇을 store하고, 무엇을 output할지 결정하는 gate를 추가해 long sequence에서 gradient를 안정화합니다. **GRU**는 LSTM을 두 gate로 단순화합니다. parameter가 더 적으면서 비슷한 성능을 냅니다.

**Bidirectional RNN**은 하나의 RNN을 forward로, 다른 하나를 backward로 실행하고 hidden state를 concatenate합니다. 모든 token representation은 left context와 right context를 모두 봅니다. Tagging task에는 필수적입니다.

```figure
rnn-unroll
```

## 직접 만들기

### 1단계: PyTorch의 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)`는 `[batch, seq_len, embed_dim]`을 `[batch, embed_dim, seq_len]`으로 바꿉니다. `nn.Conv1d`가 가운데 축을 channel로 보기 때문입니다. Pooled output은 input length와 관계없이 fixed-size입니다.

### 2단계: LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Last-state pool이 아니라 sequence 위의 max-pool을 사용합니다. Classification에서는 보통 max-pooling이 last hidden state를 쓰는 것보다 낫습니다. 긴 sequence의 끝부분 정보가 last state를 지배하기 쉽기 때문입니다.

### 3단계: vanishing gradient demo (직관)

Gating이 없는 plain RNN은 long-range dependency를 학습할 수 없습니다. 장난감 task를 생각해봅시다. sequence 어딘가에 token `A`가 등장했는지 예측합니다. `A`가 position 1에 있고 sequence가 100 token 길이라면, loss의 gradient는 recurrent weight를 99번 곱한 경로를 따라 뒤로 흘러야 합니다. weight가 1보다 작으면 gradient가 사라집니다. 1보다 크면 폭발합니다.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM은 additive interaction만으로 network를 통과하는 **cell state**로 이를 해결합니다. forget gate가 이를 multiplicative하게 scale하지만, gradient는 여전히 "highway"를 따라 흐릅니다. GRU도 더 적은 parameter로 비슷한 일을 합니다. 둘 다 100+ step sequence에서 stable training을 가능하게 합니다.

### 4단계: 그래도 충분하지 않았던 이유

LSTM에서도 세 가지 문제가 남았습니다.

1. **Sequential bottleneck.** length 1000 sequence에서 RNN을 training하려면 1000개의 serial forward/backward step이 필요합니다. 시간 축을 따라 parallelize할 수 없습니다.
2. **Encoder-decoder setup의 fixed-size context vector.** Decoder는 encoder의 final hidden state만 봅니다. 이 state는 전체 input을 압축한 것입니다. 긴 input에서는 detail이 손실됩니다. Lesson 09에서 이를 직접 다룹니다.
3. **Distant-dependency accuracy ceiling.** LSTM은 plain RNN보다 낫지만, 200+ step을 가로질러 특정 정보를 전달하는 데 여전히 어려움을 겪습니다.

Attention은 세 문제를 모두 해결했습니다. Transformer는 recurrence를 완전히 버렸습니다. Lesson 10이 전환점입니다.

## 사용하기

PyTorch의 `nn.LSTM`, `nn.GRU`, `nn.Conv1d`는 production-ready입니다. Training code는 표준적입니다.

Hugging Face는 input layer로 꽂아 쓸 수 있는 pretrained embedding을 제공합니다.

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Constraint에 맞을 때 사용하는 checklist입니다.

- **Edge / on-device inference.** GloVe embedding을 쓰는 TextCNN은 transformer보다 10-100배 작습니다. deploy target이 phone이라면 이 stack입니다.
- **Streaming / online classification.** RNN은 token을 한 번에 하나씩 처리합니다. Transformer는 전체 sequence가 필요합니다. Real-time incoming text에서는 LSTM이 여전히 이깁니다.
- **Tiny models for baselines.** 새 task에서 빠르게 iteration합니다. CPU에서 5분 안에 TextCNN을 training할 수 있습니다.
- **Sequence labeling with limited data.** BiLSTM-CRF(lesson 06)는 1k-10k labeled sentence에서도 여전히 production-grade NER architecture입니다.

그 외에는 모두 transformer로 갑니다.

## 내보내기

`outputs/prompt-text-encoder-picker.md`로 저장하세요.

```markdown
---
name: text-encoder-picker
description: 주어진 constraint set에 맞는 text encoder architecture를 고릅니다.
phase: 5
lesson: 08
---

constraint(task, data volume, latency budget, deploy target, compute budget)가 주어지면 다음을 출력합니다.

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, 또는 "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, frozen GloVe / fastText, 또는 contextualized transformer embeddings.
3. 5줄짜리 training recipe: optimizer, learning rate, batch size, epochs, regularization.
4. 하나의 monitoring signal. RNN/CNN model: attention mechanism이 없으면 long-range dep를 놓칩니다. per-length accuracy를 확인합니다. Transformer: LR이 너무 높으면 fine-tuning collapse가 발생합니다. train loss를 확인합니다.

data가 약 500 labeled example 미만일 때, TextCNN / BiLSTM baseline이 plateau했다는 것을 보이지 않고 transformer fine-tuning을 추천하지 않습니다. Edge deployment는 무엇보다 먼저 architecture가 필요하다고 표시합니다.
```

## 연습 문제

1. **Easy.** 3-class toy dataset을 TextCNN으로 training하세요. data는 직접 만듭니다. filter width (2, 3, 4)가 단일 width (3)보다 평균 F1에서 나은지 확인하세요.
2. **Medium.** LSTM classifier에 max-pool, mean-pool, last-state pooling을 구현하세요. 작은 dataset에서 비교하고, 어떤 pooling이 이기는지와 그 이유에 대한 가설을 문서화하세요.
3. **Hard.** BiLSTM-CRF NER tagger를 만드세요(lesson 06과 이번 lesson을 결합). CoNLL-2003에서 training합니다. lesson 06의 CRF-alone baseline 및 BERT fine-tune과 비교하세요. training time, memory, F1을 보고하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| TextCNN | 텍스트용 CNN | Word embedding 위의 1D convolution stack과 global max-pool입니다. Kim (2014). |
| RNN | Recurrent net | 각 time step에서 hidden state를 update합니다. `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | input / forget / output gate와 cell state를 추가합니다. long sequence에서도 안정적으로 training됩니다. |
| GRU | 더 단순한 LSTM | 세 gate 대신 두 gate를 사용합니다. 비슷한 accuracy와 더 적은 parameter를 가집니다. |
| Bidirectional | 양방향 | Forward + backward RNN을 concatenate합니다. 모든 token은 context의 양쪽을 모두 봅니다. |
| Vanishing gradient | Training signal이 사라짐 | Plain RNN에서 1보다 작은 weight가 반복해서 곱해져 early-step gradient가 사실상 0이 됩니다. |

## 더 읽을거리

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN paper입니다. 8쪽이고 읽기 쉽습니다.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM paper입니다. 의외로 명료합니다.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — LSTM을 모두가 이해할 수 있게 만든 diagram입니다.
