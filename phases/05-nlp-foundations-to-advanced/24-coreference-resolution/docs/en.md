# Coreference Resolution

> "She called him. He did not answer. The doctor was at lunch." 이름이 나오지 않아도 세 reference가 두 사람을 가리킨다. Coreference resolution은 누가 누구인지 알아낸다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## 문제

300-word article에서 Apple Inc.의 모든 mention을 추출하라. article이 "Apple"이라고 말하면 쉽다. "the company", "they", "Cupertino's technology giant", "Jobs's firm"이라고 말하면 어렵다. 이 mention들을 같은 entity로 resolve하지 않으면 NER pipeline은 mention의 60-80%를 놓친다.

Coreference resolution은 같은 real-world entity를 가리키는 모든 expression을 하나의 cluster로 연결한다. 이는 surface-level NLP(NER, parsing)와 downstream semantics(IE, QA, summarization, KG)를 이어 주는 접착제다.

2026년에 중요한 이유:

- Summarization: "The CEO announced..."와 "Tim Cook announced..." 중 summary는 CEO의 이름을 말해야 한다.
- Question answering: "Who did she call?"에는 "she"를 resolve해야 한다.
- Information extraction: "PER1 founded Apple"과 "Jobs founded Apple"이 별도 entry로 있는 knowledge graph는 틀렸다.
- Multi-document IE: 같은 event를 다루는 여러 article의 mention을 merge하는 것은 cross-document coreference다.

## 개념

![Coreference clustering: mentions → entities](../assets/coref.svg)

**Task.** Input: document. Output: 각 cluster가 하나의 entity를 가리키는 mention(span) clustering.

**Mention types.**

- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

**Architectures.**

1. **Rule-based (Hobbs, 1978).** grammar rule을 사용한 syntactic-tree-based pronoun resolution. 좋은 baseline이다. pronoun에서는 놀랄 만큼 이기기 어렵다.
2. **Mention-pair classifier.** 모든 mention pair (m_i, m_j)에 대해 corefer 여부를 예측한다. transitive closure로 cluster를 만든다. 2016년 이전의 표준.
3. **Mention-ranking.** 각 mention마다 candidate antecedent("no antecedent" 포함)를 rank한다. top을 고른다.
4. **Span-based end-to-end (Lee et al., 2017).** Transformer encoder. length cap까지 모든 candidate span을 enumerate한다. mention score를 예측한다. 각 span의 antecedent-probability를 예측한다. greedily cluster한다. 현대적 default다.
5. **Generative (2024+).** LLM에 prompt한다. "List every pronoun in this text and its antecedent." 쉬운 case에서는 잘 동작하지만, long document와 rare referent에서는 흔들린다.

**Evaluation metrics.** clustering quality를 하나의 metric만으로 포착할 수 없어서 다섯 표준 metric(MUC, B³, CEAF, BLANC, LEA)을 쓴다. 앞의 세 개 평균을 CoNLL F1으로 report한다. 2026년 CoNLL-2012 state-of-the-art: 약 83 F1.

**Known hard cases.**

- 몇 페이지 전에 소개된 entity를 가리키는 definite description.
- Bridging anaphora("the wheels" → 앞서 언급된 car).
- Chinese와 Japanese 같은 언어의 zero anaphora.
- Cataphora(pronoun이 referent보다 먼저 나옴): "When **she** walked in, Mary smiled."

## 직접 만들기

### 1단계: pretrained neural coreference (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

더 긴 document에서는 다음과 같은 결과를 얻는다.
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### 2단계: rule-based pronoun resolver (teaching)

stdlib-only implementation은 `code/main.py`를 보라.

1. mention 추출: named entities(capitalized spans), pronouns(dict lookup), definite descriptions("the X").
2. 각 pronoun마다 이전 K mentions를 살펴보고 다음으로 score한다.
   - gender/number agreement(heuristic)
   - recency(가까울수록 이김)
   - syntactic role(subject 선호)
3. 가장 높은 score의 antecedent에 link한다.

neural model과 경쟁할 수준은 아니다. 하지만 search space와 end-to-end model이 내려야 하는 decision을 보여준다.

### 3단계: coreference에 LLM 사용하기

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

주의할 failure mode가 두 가지 있다. 첫째, LLM은 서로 다른 사람을 가리키는 "him"과 "her"를 over-merge한다. 둘째, LLM은 long document에서 mention을 조용히 누락한다. 항상 span-offset check로 verify하라.

### 4단계: evaluation

표준 conll-2012 script는 MUC, B³, CEAF-φ4를 계산하고 평균을 report한다. in-house eval은 annotated test set에서 span-level precision과 recall로 시작한 뒤 mention-linking F1을 추가하라.

## 함정

- **Singleton explosion.** 일부 system은 모든 mention을 자기만의 cluster로 report한다. B³는 관대하다. MUC는 이를 벌한다. 항상 세 metric을 모두 확인하라.
- **Long context의 pronoun.** 2,000 tokens를 넘는 document에서는 performance가 약 15 F1 떨어진다. 신중하게 chunk하라.
- **Gender assumptions.** hard-coded gender rule은 non-binary referent, organization, animal에서 깨진다. learned model 또는 neutral scoring을 사용하라.
- **Long doc에서 LLM drift.** 단일 API call로 50+ paragraphs 전체의 mention을 안정적으로 cluster할 수 없다. sliding-window + merge를 사용하라.

## 사용하기

2026년 stack:

| Situation | Pick |
|-----------|------|
| English, single document | `en_coreference_web_trf` (spaCy-experimental) 또는 AllenNLP neural coref |
| Multilingual | OntoNotes 또는 Multilingual CoNLL에서 train된 SpanBERT / XLM-R |
| Cross-document event coref | Specialized end-to-end models (2025-26 SOTA) |
| Quick LLM baseline | structured-output coref prompt를 쓰는 GPT-4o / Claude |
| Production dialog systems | Rule-based fallback + neural primary + critical slots에 대한 manual review |

2026년에 실제 배포되는 integration pattern: 먼저 NER를 실행하고, coref를 실행한 뒤, coref cluster를 NER entity로 merge한다. Downstream task는 mention마다 하나의 entity가 아니라 cluster마다 하나의 entity를 본다.

## 배포하기

`outputs/skill-coref-picker.md`로 저장하라.

```markdown
---
name: coref-picker
description: coreference approach, evaluation plan, integration strategy를 고른다.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

use case(single-doc / multi-doc, domain, language)가 주어지면 다음을 출력하라.

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

sliding-window merge 없이 2,000 tokens를 넘는 document에 LLM-only coref를 쓰는 것은 거부하라. mention-level precision-recall report 없이 coref를 실행하는 pipeline은 거부하라. demographically diverse text에 배포된 gender-heuristic system은 flag하라.
```

## 연습문제

1. **Easy.** `code/main.py`의 rule-based resolver를 직접 만든 5개 paragraph에 실행하라. ground truth 대비 mention-link accuracy를 측정하라.
2. **Medium.** pretrained neural coref model을 news article에 사용하라. cluster를 직접 만든 manual annotation과 비교하라. 어디서 실패했는가?
3. **Hard.** coref-enhanced NER pipeline을 만들라. 먼저 NER를 실행하고, coref cluster로 merge하라. 100 articles에서 NER-only 대비 entity-coverage improvement를 측정하라.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Mention | reference | entity(name, pronoun, noun phrase)를 가리키는 text span. |
| Antecedent | "it"이 가리키는 것 | 뒤의 mention과 corefer하는 앞선 mention. |
| Cluster | entity의 mentions | 모두 같은 real-world entity를 가리키는 mention 집합. |
| Anaphora | 뒤쪽 reference | 나중 mention이 앞선 mention을 가리킴("he" → "John"). |
| Cataphora | 앞쪽 reference | 앞선 mention이 나중 mention을 가리킴("When he arrived, John..."). |
| Bridging | implicit reference | "I bought a car. The wheels were bad."(그 car의 wheels.) |
| CoNLL F1 | leaderboard의 숫자 | MUC, B³, CEAF-φ4 F1 score의 평균. |

## 더 읽을거리

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — canonical textbook chapter.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based end-to-end.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — coref를 개선하는 pretraining.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — benchmark.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — rule-based classic.
