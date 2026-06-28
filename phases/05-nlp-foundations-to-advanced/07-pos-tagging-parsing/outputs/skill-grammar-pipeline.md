---
name: grammar-pipeline
description: downstream NLP task를 위한 고전적 POS + dependency pipeline을 설계합니다.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

downstream task(information extraction, rewrite validation, query decomposition, lemmatization)가 주어지면 다음을 출력합니다.

1. Tagset. 영어 전용 legacy pipeline에는 Penn Treebank, multilingual 또는 cross-lingual에는 Universal Dependencies를 사용합니다.
2. Library. 대부분의 production에는 spaCy(`en_core_web_sm` / `_lg` / `_trf`), academic-grade multilingual에는 stanza, 가장 높은 UD accuracy에는 trankit을 사용합니다.
3. Integration snippet. library를 호출하고 `.pos_`, `.dep_`, `.head`를 소비하는 3-5줄입니다.
4. 테스트할 failure mode. Noun-verb ambiguity(`saw`, `book`, `can`)와 PP-attachment ambiguity는 고전적인 함정입니다. output 20개를 sample로 뽑아 눈으로 확인합니다.

직접 parser를 만들라는 추천은 거부합니다. parser를 처음부터 만드는 일은 application task가 아니라 research project입니다. lowercase / uppercase variant를 처리하지 않고 POS tag를 소비하는 pipeline은 fragile하다고 표시합니다.
