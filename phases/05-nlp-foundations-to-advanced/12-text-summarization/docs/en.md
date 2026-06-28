# 텍스트 요약

> Extractive system은 문서가 말한 내용을 알려 준다. Abstractive system은 저자가 뜻한 바를 알려 준다. 서로 다른 과제이고, 서로 다른 함정이 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 minutes

## 문제

2,000단어짜리 뉴스 기사가 피드에 들어온다. 이를 담아내는 120단어가 필요하다. 기사에서 가장 중요한 세 문장을 고를 수도 있고(extractive), 내용을 자신의 말로 다시 쓸 수도 있다(abstractive). 둘 다 summarization이라고 불린다. 하지만 완전히 다른 문제다.

Extractive summarization은 ranking problem이다. 모든 sentence에 점수를 매기고 top-`k`를 반환한다. Output은 원문에서 그대로 들어 올렸기 때문에 항상 문법적이다. 위험은 기사 전체에 흩어진 내용을 놓치는 것이다.

Abstractive summarization은 generation problem이다. Transformer가 input을 조건으로 새로운 text를 만든다. Output은 유창하고 압축적이지만 source에 없던 사실을 hallucinate할 수 있다. 위험은 자신감 있는 fabrication이다.

이 수업은 두 방식을 모두 만들고, 각 방식이 고유하게 가진 failure mode를 다룬다.

## 개념

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive.** 기사를 sentence가 node이고 similarity가 edge인 graph로 다룬다. Graph 위에서 PageRank(또는 비슷한 알고리즘)를 실행해 sentence가 다른 모든 것과 얼마나 연결되어 있는지로 점수를 매긴다. 가장 높은 점수의 sentence가 summary다. 표준 구현은 **TextRank**(Mihalcea and Tarau, 2004)다.

**Abstractive.** Transformer encoder-decoder(BART, T5, Pegasus)를 document-summary pair로 fine-tune한다. Inference 때 모델은 document를 읽고 cross-attention을 통해 token-by-token으로 summary를 생성한다. 특히 Pegasus는 gap-sentence pretraining objective를 사용해 많은 fine-tuning 없이도 summarization에 강하다.

평가는 **ROUGE**(Recall-Oriented Understudy for Gisting Evaluation)를 사용한다. ROUGE-1과 ROUGE-2는 unigram과 bigram overlap을 점수화한다. ROUGE-L은 longest common subsequence를 점수화한다. 높을수록 좋지만 ROUGE-L 40은 "good", 50은 "exceptional"이다. 모든 논문은 세 가지를 모두 보고한다. `rouge-score` package를 사용하라.

## 직접 만들기

### 1단계: TextRank(extractive)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

이름 붙여 둘 만한 점이 두 가지 있다. Similarity function은 log-normalized word overlap을 사용하며, 이는 원래 TextRank variant다. TF-IDF vector의 cosine도 쓸 수 있다. Damping factor 0.85와 iteration count는 PageRank default다.

### 2단계: BART로 abstractive 요약

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN은 CNN/DailyMail corpus로 fine-tune되어 있다. 바로 news-style summary를 생성한다. 다른 domain(scientific papers, dialog, legal)에서는 해당 Pegasus checkpoint를 쓰거나 target data로 fine-tune하라.

### 3단계: ROUGE 평가

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

항상 stemming을 사용하라. 그렇지 않으면 "running"과 "run"이 서로 다른 단어로 계산되어 ROUGE가 실제 overlap을 과소계산한다.

### ROUGE를 넘어서(2026 summarization eval)

ROUGE는 20년 동안 summarization의 지배적인 metric이었지만, 2026년에는 단독으로 충분하지 않다. NLG 논문에 대한 대규모 meta-analysis는 다음을 보여 주었다.

- **BERTScore**(contextual embedding similarity)는 2023년까지 입지를 넓혔고, 이제 대부분의 summarization paper에서 ROUGE와 함께 보고된다.
- **BARTScore**는 evaluation을 generation으로 다룬다. Pretrained BART가 source를 조건으로 summary에 얼마나 높은 likelihood를 부여하는지 점수화한다.
- **MoverScore**(contextual embedding 위의 Earth Mover's Distance)는 ROUGE보다 semantic overlap을 더 잘 포착하기 때문에 2025년 summarization benchmark에서 최상위에 올랐다.
- **FactCC**와 **QA-based faithfulness**는 2021-2023년에 흔했지만, 이제는 종종 **G-Eval**(coherence, consistency, fluency, relevance를 chain-of-thought reasoning으로 점수화하는 GPT-4 prompt chain)로 대체된다.
- **G-Eval**과 유사한 LLM-judge 접근은 rubric이 잘 설계되면 약 80%의 경우 인간 판단과 맞는다.

Production recommendation은 legacy comparison에는 ROUGE-L, semantic overlap에는 BERTScore, coherence와 factuality에는 G-Eval을 보고하는 것이다. 50-100개의 human-labeled summary에 맞춰 보정하라.

### 4단계: factuality 문제

Abstractive summary는 hallucination에 취약하다. Extractive summary는 output이 source에서 그대로 들어 올려지기 때문에 hallucination risk가 훨씬 낮다. 다만 source sentence가 맥락에서 떼어져 있거나, 오래되었거나, 인용 순서가 바뀌면 여전히 오해를 줄 수 있다. 이것이 production system이 compliance-adjacent content에 여전히 extractive method를 선호하는 가장 큰 이유다.

이름 붙여야 할 hallucination type은 다음과 같다.

- **Entity swap.** Source는 "John Smith"라고 말한다. Summary는 "John Brown"이라고 말한다.
- **Number drift.** Source는 "25,000"이라고 말한다. Summary는 "25 million"이라고 말한다.
- **Polarity flip.** Source는 "rejected the offer"라고 말한다. Summary는 "accepted the offer"라고 말한다.
- **Fact invention.** Source는 CEO를 언급하지 않는다. Summary는 CEO가 승인했다고 말한다.

작동하는 평가 접근법은 다음과 같다.

- **FactCC.** Source sentence와 summary sentence 사이의 entailment로 학습한 binary classifier. Factual/not-factual을 예측한다.
- **QA-based factuality.** 답이 source에 있는 질문을 QA model에 묻는다. Summary가 다른 답을 지지하면 flag한다.
- **Entity-level F1.** Source와 summary의 named entity를 비교한다. Summary에만 있는 entity는 의심스럽다.

Factuality가 중요한 user-facing 영역(news, medical, legal, financial)에서는 extractive가 더 안전한 default다. Abstractive에는 루프 안에 factuality check가 필요하다.

## 사용하기

2026년 stack은 다음과 같다.

| 사용 사례 | 권장 항목 |
|---------|-------------|
| News, 3-5 sentence summary, English | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` 또는 tuned T5 |
| Multi-document, long-form | 32k+ context를 가진 prompted LLM |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| 구조상 hallucination risk가 낮은 extractive | TextRank 또는 `sumy`의 LSA / LexRank |

Long context를 가진 LLM은 compute가 제약이 아닐 때 2026년에 specialized model을 자주 이긴다. Tradeoff는 cost와 reproducibility다. Specialized model은 더 일관된 output을 제공한다.

## 배포하기

`outputs/skill-summary-picker.md`로 저장하라.

```markdown
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
```

## 연습문제

1. **쉬움.** 뉴스 기사 5개에 TextRank를 실행하라. Top-3 sentence를 reference summary와 비교하라. ROUGE-L을 측정하라. CNN/DailyMail-style article에서는 ROUGE-L 30-45가 보일 것이다.
2. **중간.** Entity-level factuality를 구현하라. Source와 summary에서 named entity를 추출하고(spaCy), summary 안 source entity recall과 source 대비 summary entity precision을 계산하라. High precision과 low recall은 안전하지만 짧다는 뜻이고, low precision은 hallucinated entity를 의미한다.
3. **어려움.** CNN/DailyMail article 50개에서 BART-large-CNN과 LLM(Claude 또는 GPT-4)을 비교하라. ROUGE-L, factuality(entity F1 기준), summary당 cost를 보고하라. 각각 어디서 이기는지 문서화하라.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Extractive | Pick sentences | Source의 sentence를 그대로 반환한다. Hallucination을 만들지 않는다. |
| Abstractive | Rewrite | Source를 조건으로 새 text를 생성한다. Hallucination이 가능하다. |
| ROUGE | Summary metric | System output과 reference 사이의 N-gram / LCS overlap. |
| TextRank | Graph-based extractive | Sentence similarity graph 위의 PageRank. |
| Factuality | Is it right | Summary claim이 source로 뒷받침되는지 여부. |
| Hallucination | Made-up content | Source가 뒷받침하지 않는 summary content. |

## 더 읽기

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — extractive canonical paper.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART paper.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus와 gap-sentence objective.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE paper.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — factuality landscape paper.
