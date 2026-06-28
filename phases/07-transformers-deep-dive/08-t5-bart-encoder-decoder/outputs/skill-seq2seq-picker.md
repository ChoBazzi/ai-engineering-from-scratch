---
name: seq2seq-picker
description: 새로운 sequence-to-sequence task에 대해 encoder-decoder와 decoder-only 중 무엇을 쓸지 고릅니다.
version: 1.0.0
phase: 7
lesson: 8
tags: [transformers, t5, bart, seq2seq]
---

seq2seq task(translation / summarization / speech-to-text / structured extraction / rewrite), input과 output length distribution, quality vs latency priority가 주어지면 다음을 출력하세요.

1. architecture. encoder-decoder(T5 / BART / Whisper-style), decoder-only instruction-tuned, encoder-only + prompt template 중 하나를 고릅니다. 한 문장 이유를 제시합니다.
2. pretraining objective. Span corruption(T5), denoising(BART), next-token(decoder-only), 또는 "pretraining을 건너뛰고 기존 checkpoint를 fine-tune" 중 하나. checkpoint 이름을 명시합니다.
3. input formatting. task prefix string(T5 style), system prompt(decoder-only), raw tokens(BART) 중 하나. BOS/EOS handling을 포함합니다.
4. decoding strategy. translation/summary에는 beam search width와 length penalty, chat-like task에는 nucleus/min-p를 제시합니다. 해당 task에 무엇을 쓰는지 명시합니다.
5. 평가. task에 맞는 metric: BLEU / ROUGE / WER / F1 / exact match. test split size를 포함합니다.

generative output에 encoder-only를 권장하지 마세요. input이 이미 conversation인 경우 encoder-decoder를 권장하지 마세요. decoder-only가 conversation memory에 자연스럽게 맞습니다. speech-to-text에 decoder-only를 고르면서 이겨야 할 baseline으로 Whisper를 언급하지 않으면 표시하세요.
