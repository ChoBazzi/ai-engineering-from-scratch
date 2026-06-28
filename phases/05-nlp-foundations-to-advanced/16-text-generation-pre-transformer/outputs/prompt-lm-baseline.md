---
name: lm-baseline
description: neural LM을 training하기 전에 reproducible n-gram language model baseline을 만듭니다.
phase: 5
lesson: 16
---

corpus와 target use(next-word prediction, rescoring, perplexity baseline)가 주어지면 다음을 출력하세요:

1. N-gram order. 일반 English에는 trigram, corpus가 크면 4-gram, speech rescoring에는 5-gram.
2. Smoothing. Modified Kneser-Ney가 default입니다. Laplace는 teaching용으로만 사용합니다.
3. Library. production에는 `kenlm`, teaching에는 `nltk.lm`, math를 배우기 위한 경우에만 직접 구현합니다.
4. Evaluation. train과 test sets 사이에 consistent tokenization을 사용한 held-out perplexity.

비교 중인 systems 사이에 서로 다른 tokenization으로 계산한 perplexity는 보고하지 마세요. perplexity numbers는 identical tokenization에서만 비교 가능합니다. test set의 OOV rate를 flag하세요. training 중 special `<UNK>` token을 reserve하지 않으면 KN은 OOV를 잘 처리하지 못합니다.
