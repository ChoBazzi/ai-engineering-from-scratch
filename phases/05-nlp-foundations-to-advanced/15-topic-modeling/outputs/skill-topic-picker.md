---
name: topic-picker
description: corpus에 맞는 LDA 또는 BERTopic을 고릅니다. library, knobs, evaluation을 지정합니다.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

corpus 설명(document count, avg length, domain, language, compute budget)이 주어지면 다음을 출력하세요:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. 한 문장짜리 이유.
2. Configuration. Number of topics(`~sqrt(n_docs)`에서 시작), `min_df` / `max_df` filters, neural approaches용 embedding model.
3. Evaluation. `gensim.models.CoherenceModel`을 통한 topic coherence (c_v), topic diversity, 그리고 20-sample human read.
4. Failure mode to probe. LDA에서는 stopwords와 frequent terms를 흡수하는 "junk topics"입니다. BERTopic에서는 ambiguous documents를 삼키는 -1 outlier cluster입니다.

chunking strategy 없이 embedding model의 context window보다 긴 documents에 BERTopic을 쓰는 것은 거절하세요. coherence가 무너지므로 매우 짧은 text(tweets, 10 tokens 미만 reviews)에 LDA를 쓰는 것도 거절하세요. 5 미만 또는 200 초과의 `n_topics` 선택은 실제 data에서는 likely wrong으로 flag하세요.
