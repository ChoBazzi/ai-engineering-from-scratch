---
name: sentiment-baseline
description: 새 dataset을 위한 sentiment analysis baseline을 설계합니다.
phase: 5
lesson: 05
---

Task description(domain, language, size, label granularity, latency budget)이 주어지면 다음을 output하세요.

1. Feature extraction recipe. Tokenizer, n-gram range, stopword policy(보통 keep), negation handling(scoped prefix 또는 bigram)을 명시하세요.
2. Classifier. Baseline에는 Naive Bayes, production에는 logistic regression, domain에 sarcasm, aspect-based output, 또는 cross-lingual coverage가 필요할 때만 transformer를 사용하세요.
3. Evaluation plan. Precision, recall, F1, confusion matrix, per-class error sample을 report하세요. Imbalanced data에서는 accuracy alone을 절대 report하지 마세요.
4. Post-deployment에서 monitor할 failure mode 하나. Domain drift와 sarcasm이 상위 두 가지입니다. Weekly sample audit을 제안하세요.

Sentiment task에서 stopword 제거를 추천하지 마세요. Class가 imbalanced일 때 accuracy를 유일한 metric으로 report하지 마세요. Subword가 많은 language(German, Finnish, Turkish)는 word-level TF-IDF보다 FastText 또는 transformer embedding이 필요하다고 flag하세요.
