---
name: summary-picker
description: extractive 또는 abstractive를 고르고, library와 factuality check를 지정합니다.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

task(document type, compliance requirement, length, compute budget)가 주어지면 다음을 출력하라.

1. 접근법. Extractive 또는 abstractive. 이유를 한 문장으로 설명한다.
2. 시작 model / library. 이름을 명시한다. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, 또는 LLM prompt.
3. 평가 계획. ROUGE-1, ROUGE-2, ROUGE-L(`rouge-score`와 stemming 사용). Abstractive라면 factuality check도 포함한다.
4. 조사할 failure mode 하나. Entity swap은 abstractive news summarization에서 가장 흔하다. Source entity가 summary에 나타나지 않는 sample을 flag한다.

Factuality gate 없이 medical, legal, financial, 또는 regulated content에 abstractive summarization을 거부하라. Model context window를 넘는 input은 단순 truncation이 아니라 chunked map-reduce summarization이 필요하다고 표시하라.
