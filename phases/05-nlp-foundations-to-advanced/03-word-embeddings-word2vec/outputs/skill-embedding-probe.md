---
name: embedding-probe
description: word2vec model을 점검합니다. analogy를 실행하고, neighbor를 찾고, quality를 진단합니다.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

학습된 word embedding이 제대로 작동하는지 확인합니다. `gensim.models.KeyedVectors` object와 vocabulary가 주어지면 다음을 실행합니다.

1. 세 가지 canonical analogy test. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. top-1 결과와 cosine을 보고합니다.
2. 사용자가 제공한 domain-specific word에 대해 다섯 가지 nearest-neighbor test. top-5 neighbor와 cosine을 출력합니다.
3. symmetry check 하나. float precision 범위 안에서 `similarity(a, b) == similarity(b, a)`인지 확인합니다.
4. degenerate check 하나. embedding의 norm이 0.01보다 작거나 100보다 크면 모델에 training bug가 있습니다. 표시합니다.

analogy accuracy만으로 모델이 좋다고 선언하지 않습니다. Analogy benchmark는 쉽게 맞춤형으로 조작될 수 있고 downstream task로 전이되지 않습니다. intrinsic evaluation과 downstream evaluation을 함께 권장합니다.
