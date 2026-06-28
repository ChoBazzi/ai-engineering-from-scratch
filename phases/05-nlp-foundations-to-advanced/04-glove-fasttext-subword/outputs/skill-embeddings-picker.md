---
name: skill-embeddings-picker
description: 새 language model 또는 text pipeline을 위한 tokenization approach를 선택합니다.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

task와 dataset description이 주어지면 다음을 출력합니다.

1. Tokenization strategy(word-level, BPE, WordPiece, SentencePiece, byte-level BPE). 한 문장짜리 이유.
2. Vocabulary size target. English-only LM: 32k. Multilingual: 64k-100k. Code: 50k-100k.
3. 정확한 training command가 포함된 library call. library 이름(Hugging Face `tokenizers`, `sentencepiece`)을 말하세요. argument를 quote하세요.
4. reproducibility pitfall 하나. Tokenizer-model mismatch는 production에서 가장 흔한 silent bug입니다. 어떤 tokenizer가 어떤 pretrained checkpoint와 짝을 이루는지 말하고, 바꾸지 말라고 경고하세요.

사용자가 pretrained LLM을 fine-tuning하는 경우 custom tokenizer 학습을 권장하지 마세요(fine-tune은 pretrained tokenizer를 사용해야 합니다). production inference path를 목표로 하는 어떤 model에도 word-level tokenization을 권장하지 마세요. non-English 또는 multi-script corpus에는 byte fallback이 있는 SentencePiece가 필요하다고 표시하세요.
