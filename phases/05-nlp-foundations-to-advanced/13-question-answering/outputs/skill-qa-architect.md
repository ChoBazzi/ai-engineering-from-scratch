---
name: qa-architect
description: QA architecture, retrieval strategy, evaluation plan을 선택합니다.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Requirement(corpus size, question type, factuality constraint, latency budget)가 주어지면 다음을 출력합니다.

1. Architecture. Extractive, extractive reader를 쓰는 RAG, generative reader를 쓰는 RAG, 또는 closed-book LLM. 한 문장 reason.
2. Retriever. None, BM25, dense(`all-MiniLM-L6-v2` 같은 encoder 이름 명시), 또는 hybrid.
3. Reader. SQuAD-tuned model(`deepset/roberta-base-squad2`), 이름을 명시한 LLM, 또는 domain-fine-tuned DistilBERT.
4. Evaluation. Extractive benchmark에는 EM + F1; production에는 answer accuracy + citation accuracy + refusal calibration. 무엇을 어떻게 측정하는지 이름 붙입니다.

Regulatory 또는 compliance-sensitive question에는 closed-book LLM answer를 거부합니다. Retrieval-recall baseline이 없는 QA system도 거부합니다(retriever가 right passage를 surfaced했는지 모르면 reader를 evaluate할 수 없습니다). Multi-hop reasoning이 필요한 question은 HotpotQA-trained system 같은 specialized multi-hop retriever가 필요하다고 flag합니다.
