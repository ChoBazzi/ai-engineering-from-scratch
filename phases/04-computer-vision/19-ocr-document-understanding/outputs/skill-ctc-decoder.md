---
name: skill-ctc-decoder
description: length normalisation을 포함해 greedy 및 beam-search CTC decoder를 처음부터 작성하기
version: 1.0.0
phase: 4
lesson: 19
tags: [ocr, ctc, decoding, sequence-models]
---

# CTC Decoder

CTC output을 위한 두 decoding routine을 만듭니다. greedy는 빠르고, beam은 noisy input에서 더 좋습니다.

## 사용 시점

- custom CRNN output에서 OCR inference를 실행할 때.
- pretrained OCR model을 여러 decoder와 비교 benchmark할 때.
- ctcdecode를 끌어오지 않고 단순한 beam search를 구현할 때.

## Inputs

- `log_probs`: vocab에 대한 (T, N, C) log-softmax(index 0 = convention상 blank).
- `vocab`: C개 character의 list.
- `beam_width`(beam only): 보통 5-10.

## Greedy decoder

```python
def greedy_ctc_decode(log_probs, vocab, blank=0):
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(vocab[idx])
            prev = idx
        out.append("".join(decoded))
    return out
```

## Beam search decoder

```python
import heapq
import math

def beam_ctc_decode(log_probs, vocab, beam_width=5, blank=0):
    T, N, C = log_probs.shape
    lp = log_probs.cpu()
    results = []
    for n in range(N):
        beams = {("",): (0.0, -math.inf)}  # (prefix_tuple) -> (p_blank, p_nonblank)
        for t in range(T):
            logits_t = lp[t, n]
            new_beams = {}
            for prefix, (p_b, p_nb) in beams.items():
                for c in range(C):
                    p = logits_t[c].item()
                    if c == blank:
                        nb = p_b + p
                        nnb = p_nb + p
                        upd = new_beams.get(prefix, (-math.inf, -math.inf))
                        new_beams[prefix] = (
                            _logsumexp(upd[0], _logsumexp(nb, nnb)),
                            upd[1],
                        )
                    else:
                        last = prefix[-1] if prefix else ""
                        char = vocab[c]
                        if char == last:
                            # Case 1: stay on same prefix (collapse from p_nb)
                            upd = new_beams.get(prefix, (-math.inf, -math.inf))
                            new_beams[prefix] = (upd[0], _logsumexp(upd[1], p_nb + p))
                            # Case 2: extend prefix via blank-separated repeat ("a_a" -> "aa")
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], p_b + p))
                        else:
                            new_prefix = prefix + (char,)
                            upd = new_beams.get(new_prefix, (-math.inf, -math.inf))
                            nb = _logsumexp(p_b, p_nb) + p
                            new_beams[new_prefix] = (upd[0], _logsumexp(upd[1], nb))
            beams = dict(heapq.nlargest(
                beam_width,
                new_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
            ))
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        results.append("".join(best))
    return results


def _logsumexp(a, b):
    if a == -math.inf: return b
    if b == -math.inf: return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))
```

## Rules

- PyTorch의 `nn.CTCLoss`에서 CTC blank index는 convention상 0입니다.
- Beam search는 low-confidence input에서 accuracy를 개선합니다. clean input에서는 improvement가 <1% CER입니다.
- beam을 5 아래로 prune하지 않습니다. 그 아래에서는 accuracy-latency trade가 평평해집니다.
- tight latency budget 안에서 beam search를 실행해야 한다면 greedy로 내립니다. 대부분의 production OCR data에서는 quality hit가 작습니다.
- large vocabulary(CJK의 3000+ character)에서는 위 pure Python version 대신 `ctcdecode`(C++)로 전환합니다. Python beam은 빠르게 bottleneck이 됩니다.
