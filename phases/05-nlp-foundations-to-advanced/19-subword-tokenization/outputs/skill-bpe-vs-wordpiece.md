---
name: skill-bpe-vs-wordpiece
description: 주어진 corpus와 deployment target에 맞는 tokenizer algorithm, vocab size, library를 선택합니다.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

corpus(size, languages, domain)와 deployment target(training from scratch / fine-tuning / API-compatible inference)이 주어지면 다음을 출력합니다.

1. Algorithm. BPE, Unigram, 또는 WordPiece. 한 문장 이유.
2. Library. SentencePiece, HF Tokenizers, 또는 tiktoken. 이유.
3. Vocab size. 가장 가까운 1k로 반올림. model size와 language coverage에 연결된 이유.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. held-out set의 average tokens-per-word, OOV rate, compression ratio, round-trip decode equality.

rare-script content가 있는 corpus에서 character-coverage <0.995 tokenizer를 훈련하는 것을 거부하세요. CI에서 고정된 `tokenizer.json` hash check 없이 vocab을 배포하는 것을 거부하세요. vocab 16k 미만의 monolingual tokenizer는 likely under-spec으로 flag하세요.
