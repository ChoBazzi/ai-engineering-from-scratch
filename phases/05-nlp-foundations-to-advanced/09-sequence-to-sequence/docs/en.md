# Sequence-to-Sequence 모델

> 번역기인 척하는 두 RNN. 이들이 부딪히는 병목이 attention이 존재하는 이유다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75분

## 문제

분류는 가변 길이 sequence를 하나의 label로 매핑한다. 번역은 가변 길이 sequence를 또 다른 가변 길이 sequence로 매핑한다. 입력과 출력은 서로 다른 vocabulary, 어쩌면 서로 다른 언어에 속하며, 길이가 같다는 보장도 없다.

seq2seq architecture(Sutskever, Vinyals, Le, 2014)는 의도적으로 단순한 recipe로 이 문제를 풀었다. 두 개의 RNN. 하나는 source sentence를 읽고 fixed-size context vector를 만든다. 다른 하나는 그 vector를 읽고 target sentence를 token by token으로 생성한다. Lesson 08에서 작성한 것과 같은 code를 다르게 붙인 것이다.

이것을 공부할 가치가 있는 이유는 두 가지다. 첫째, context-vector bottleneck은 NLP에서 가장 교육적으로 유용한 실패다. attention과 transformer가 잘하는 모든 것을 동기화한다. 둘째, training recipe(teacher forcing, scheduled sampling, inference 시 beam search)는 LLM을 포함한 모든 현대 generation system에도 여전히 적용된다.

## 개념

**Encoder.** source sentence를 읽는 RNN. 마지막 hidden state가 **context vector**다. 즉 전체 입력의 fixed-size summary다. source만 잃지 않으면 아무것도 잃지 않는다는 가정이다.

**Decoder.** context vector로 초기화되는 또 다른 RNN. 각 step에서 이전에 생성된 token을 입력으로 받고 target vocabulary에 대한 distribution을 만든다. sample 또는 argmax로 다음 token을 고른다. 다시 입력으로 넣는다. `<EOS>` token이 생성되거나 max length에 도달할 때까지 반복한다.

**Training:** 각 decoder step에서 cross-entropy loss를 계산하고 sequence 전체에 대해 합산한다. 두 network 모두에 대해 standard backprop through time을 수행한다.

**Teacher forcing.** Training 중 step `t`에서 decoder의 입력은 decoder 자신의 이전 prediction이 아니라 position `t-1`의 *ground-truth* token이다. 이는 training을 안정화한다. 이것이 없으면 초기 실수가 cascade되어 model이 전혀 배우지 못한다. Inference에서는 model 자신의 prediction을 써야 하므로 train/inference distribution gap이 항상 존재한다. 이 gap을 **exposure bias**라고 한다.

**병목.** Encoder가 source에 대해 학습한 모든 것은 하나의 context vector 안으로 압축되어야 한다. 긴 sentence는 detail을 잃는다. Rare word는 흐려진다. Reordering(chat noir vs. black cat)은 계산되는 것이 아니라 암기되어야 한다.

Attention(Lesson 10)은 decoder가 마지막 encoder hidden state만이 아니라 *모든* encoder hidden state를 보게 해서 이를 고친다. 이것이 전체 핵심이다.

```figure
lstm-gates
```

## 직접 만들기

### 1단계: encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs`의 shape는 `[batch, seq_len, hidden_dim]`이다. 입력 position마다 하나의 hidden state가 있다. `hidden`의 shape는 `[1, batch, hidden_dim]`이다. 마지막 step이다. Lesson 08에서는 "classification을 위해 outputs를 pool하라"고 했다. 여기서는 마지막 hidden state를 context vector로 보관하고 per-step outputs는 무시한다.

### 2단계: decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder는 한 번에 한 step씩 호출된다. 입력은 single token batch와 현재 hidden state다. 출력은 다음 token에 대한 vocabulary logits와 업데이트된 hidden state다.

### 3단계: teacher forcing을 사용하는 training loop

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

이름 붙일 가치가 있는 knob이 두 개 있다. `ignore_index=0`은 padding token에 대한 loss를 건너뛴다. `teacher_forcing_ratio`는 각 step에서 true token과 model prediction 중 true token을 사용할 확률이다. 1.0(full teacher forcing)에서 시작하고 training 동안 ~0.5까지 anneal하여 exposure-bias gap을 줄인다.

### 4단계: inference loop(greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding은 매 step에서 가장 probability가 높은 token을 고른다. 이것은 빗나갈 수 있다. 한 token에 commit하면 되돌릴 수 없다. **Beam search**는 top-`k` partial sequence를 계속 살려 두고 마지막에 가장 높은 score의 complete sequence를 고른다. Beam width 3-5가 standard다.

### 5단계: 병목을 시연하기

Toy copy task에서 model을 train한다. source `[a, b, c, d, e]`, target `[a, b, c, d, e]`. Sequence length를 늘린다. Accuracy를 관찰한다.

```text
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

하나의 GRU hidden state는 40-token input을 lossless하게 암기할 수 없다. 정보는 모든 encoder step에 있지만 decoder는 마지막 state만 본다. Attention은 이를 직접 고친다.

## 사용하기

PyTorch에는 `nn.Transformer`와 `nn.LSTM` 기반 seq2seq template이 있다. Hugging Face의 `transformers` library는 수십억 token으로 train된 full encoder-decoder model(BART, T5, mBART, NLLB)을 제공한다.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

현대 encoder-decoder는 RNN을 버리고 transformer를 사용한다. High-level shape(encoder, decoder, generate-token-by-token)는 2014 seq2seq paper와 동일하다. 각 block 내부의 mechanism만 다르다.

### RNN-based seq2seq를 여전히 선택할 때

새 project에서는 거의 없다. 구체적인 예외는 다음과 같다.

- Bounded memory로 input을 token by token 소비하는 streaming translation.
- Transformer memory cost가 과도한 on-device text generation.
- 교육. Encoder-decoder bottleneck을 이해하는 것이 transformer가 왜 이겼는지 이해하는 가장 빠른 길이다.

### Exposure bias와 완화책

- **Scheduled sampling.** Training 동안 teacher forcing ratio를 anneal하여 model이 자신의 실수에서 회복하는 법을 배우게 한다.
- **Minimum risk training.** Token-level cross-entropy 대신 sentence-level BLEU score로 train한다. 실제로 원하는 것에 더 가깝다.
- **Reinforcement learning fine-tuning.** Sequence generator에 metric으로 reward를 준다. 현대 LLM RLHF에서 사용된다.

세 가지 모두 transformer-based generation에도 여전히 적용된다.

## 내보내기

`outputs/prompt-seq2seq-design.md`로 저장한다.

```markdown
---
name: seq2seq-design
description: 주어진 task를 위한 sequence-to-sequence pipeline을 설계한다.
phase: 5
lesson: 09
---

Task(translation, summarization, paraphrase, question rewrite)가 주어지면 다음을 출력한다.

1. 아키텍처. Pretrained transformer encoder-decoder(BART, T5, mBART, NLLB)가 default다. RNN-based seq2seq는 특정 constraint(streaming, edge inference, pedagogy)가 있을 때만 사용한다.
2. 시작 checkpoint. 이름을 명시한다(`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Checkpoint를 task와 language coverage에 맞춘다.
3. Decoding 전략. Deterministic output에는 greedy, quality에는 beam search(width 4-5), diversity에는 temperature를 쓰는 sampling을 사용한다. 한 문장으로 근거를 제시한다.
4. Shipping 전에 확인할 failure mode 하나. Exposure bias는 긴 output에서 generation drift로 나타난다. 90th-percentile length의 output 20개를 sample하고 눈으로 확인한다.

Parallel example이 ~1M개 미만이면 seq2seq를 scratch에서 training하라고 추천하지 않는다. User-facing content에 greedy decoding을 쓰는 pipeline은 fragile하다고 flag한다(greedy는 repeat와 loop를 만든다).
```

## 연습문제

1. **쉬움.** Toy copy task를 구현하라. Target이 source와 같은 input-output pair로 GRU seq2seq를 train하라. Length 5, 10, 20에서 accuracy를 측정하라. Bottleneck을 재현하라.
2. **중간.** Beam width 3의 beam search decoding을 추가하라. 작은 parallel corpus에서 greedy 대비 BLEU를 측정하라. Beam search가 이기는 지점(보통 마지막 token)과 차이를 만들지 못하는 지점을 기록하라.
3. **어려움.** `facebook/bart-base`를 10k-pair paraphrase dataset으로 fine-tune하라. Held-out input에서 fine-tuned model의 beam-4 output을 base model과 비교하라. BLEU를 보고하고 qualitative example 10개를 고르라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|----------|
| Encoder | Input RNN | Source를 읽는다. Per-step hidden state와 final context vector를 만든다. |
| Decoder | Output RNN | Context vector로 초기화된다. Target token을 한 번에 하나씩 생성한다. |
| Context vector | 요약 | Final encoder hidden state. Fixed size. Attention이 해결하는 bottleneck. |
| Teacher forcing | True token 사용 | Training time에 ground-truth previous token을 feed한다. Learning을 안정화한다. |
| Exposure bias | Train/test gap | True token으로 train된 model은 자신의 실수에서 회복하는 연습을 한 적이 없다. |
| Beam search | 더 나은 decoding | Greedy하게 commit하는 대신 각 step에서 top-k partial sequence를 살려 둔다. |

## 더 읽을거리

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 원래 seq2seq paper. 네 페이지다.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — GRU와 encoder-decoder framing을 도입했다.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention paper. 이 lesson 직후 읽어라.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — build 가능한 seq2seq + attention code.
