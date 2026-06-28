---
name: multilingual-picker
description: multilingual NLP task의 source language, target model, evaluation plan을 고른다.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

requirement(target language, task type, language별 available labeled data)가 주어지면 다음을 output하라:

1. fine-tuning을 위한 source language. default는 English다. target language에 typologically close한 high-resource language가 있으면 LANGRANK 또는 qWALS를 check한다.
2. Base model. XLM-R(classification), mT5(generation), NLLB(translation), Aya-23(generative LLM).
3. Few-shot budget. 가능하다면 target-language example 100-500개로 시작한다. labeling이 infeasible할 때만 zero-shot을 사용한다.
4. Evaluation 계획. per-language accuracy(aggregate 아님), cross-lingual consistency, non-Latin script의 entity-level F1.

per-language evaluation 없이 multilingual model을 ship하는 것을 거부하라. aggregate metric은 long-tail failure를 숨긴다. tokenization coverage가 낮은 script(Amharic, Tigrinya, 여러 African language)는 byte-fallback을 갖춘 model(SentencePiece with byte_fallback=True, 또는 GPT-2 같은 byte-level tokenizer)이 필요하다고 flag하라.
