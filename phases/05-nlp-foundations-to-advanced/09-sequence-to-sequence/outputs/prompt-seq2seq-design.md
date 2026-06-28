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
