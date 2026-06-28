---
name: ner-picker
description: 주어진 extraction task에 맞는 NER approach를 고릅니다.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Task description(domain, label set, language, latency, data volume)이 주어지면 다음을 output하세요.

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, 또는 transformer fine-tune.
2. Starting model. 이름을 적으세요(`en_core_web_sm` / `en_core_web_trf` 같은 spaCy model ID, `dslim/bert-base-NER` 같은 Hugging Face checkpoint ID, 또는 "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, 또는 span-based. 한 문장으로 justify하세요.
4. Evaluation. `seqeval`을 사용하세요. Token-level이 아니라 entity-level F1을 항상 report하세요.

Labeled example이 500개 미만이면 user에게 pretrained domain model(예: medical용 BioBERT)이 이미 있는 경우를 제외하고 transformer fine-tuning을 추천하지 마세요. Nested entity는 span-based 또는 multi-pass model이 필요하다고 flag하세요. User가 "production scale"을 언급하면서 out-of-the-box CoNLL-2003 label을 사용한다면 gazetteer audit을 요구하세요.
