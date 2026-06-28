# 자연어 추론 — 텍스트 함의

> "`t`가 `h`를 entail한다"는 것은 `t`를 읽은 사람이 `h`가 참이라고 결론 내릴 수 있다는 뜻입니다. NLI는 entailment / contradiction / neutral을 예측하는 작업입니다. 겉보기에는 단조롭지만, 프로덕션에서는 핵심 하중을 받는 기술입니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## 문제

요약기를 만들었습니다. 요약문이 생성되었습니다. 그 요약문에 hallucination이 들어 있지 않다는 것을 어떻게 알 수 있을까요?

chatbot을 만들었습니다. 챗봇이 "yes"라고 답했습니다. 그 답이 검색된 passage에 의해 뒷받침된다는 것을 어떻게 알 수 있을까요?

뉴스 기사 10,000개를 topic별로 분류해야 합니다. training label은 없습니다. 기존 model을 재사용할 수 있을까요?

이 세 문제는 모두 Natural Language Inference로 환원됩니다. NLI는 premise `t`와 hypothesis `h`가 주어졌을 때, `h`가 `t`에 의해 entailed되는지, contradicted되는지, 아니면 neutral(관련 없음)인지 묻습니다.

- **Hallucination 검사:** `t` = source document, `h` = summary claim. entailment가 아니면 hallucination입니다.
- **Grounded QA:** `t` = retrieved passage, `h` = generated answer. entailment가 아니면 fabrication입니다.
- **Zero-shot classification:** `t` = document, `h` = verbalized label ("This is about sports"). Entailment = predicted label.

하나의 작업, 세 가지 프로덕션 용도입니다. 그래서 모든 RAG evaluation framework는 내부에 NLI model을 탑재합니다.

## 개념

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**세 가지 label.**

- **Entailment.** `t` → `h`. "The cat is on the mat"는 "There is a cat."을 entail합니다.
- **Contradiction.** `t` → ¬`h`. "The cat is on the mat"는 "There is no cat."과 contradict합니다.
- **Neutral.** 어느 쪽으로도 inference가 없습니다. "The cat is on the mat"는 "The cat is hungry."에 대해 neutral입니다.

**논리적 함의가 아닙니다.** NLI는 *natural* language inference입니다. 엄격한 논리가 아니라 일반적인 인간 독자가 추론할 내용을 다룹니다. "John walked his dog"는 NLI에서는 "John has a dog"를 entail하지만, 엄격한 first-order logic에서는 소유 관계를 공리화해야만 그렇게 인정할 수 있습니다.

**Datasets.**

- **SNLI** (2015). 570k개의 human-annotated pair, premise는 image caption입니다. domain이 좁습니다.
- **MultiNLI** (2017). 10개 genre에 걸친 433k pair입니다. 2026년 기준 표준 training corpus입니다.
- **ANLI** (2019). Adversarial NLI입니다. 인간이 기존 model을 깨도록 특별히 설계한 example을 작성했습니다. 더 어렵습니다.
- **DocNLI, ConTRoL** (2020–21). Document-length premise입니다. multi-hop과 long-range inference를 테스트합니다.

**Architecture.** transformer encoder(BERT, RoBERTa, DeBERTa)가 `[CLS] premise [SEP] hypothesis [SEP]`를 읽습니다. `[CLS]` representation이 3-way softmax로 들어갑니다. MNLI로 train하고 held-out benchmark에서 evaluate하면, in-distribution pair에서 90%+ accuracy를 얻습니다.

**NLI를 통한 zero-shot.** document와 candidate label이 주어지면 각 label을 hypothesis("This text is about sports")로 바꿉니다. 각 hypothesis의 entailment probability를 계산합니다. 최댓값을 고릅니다. 이것이 Hugging Face의 `zero-shot-classification` pipeline 뒤에 있는 mechanism입니다.

## 직접 만들기

### 1단계: pretrained NLI model 실행하기

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

프로덕션 NLI에서는 `facebook/bart-large-mnli`와 `microsoft/deberta-v3-large-mnli`가 open default입니다. DeBERTa-v3가 leaderboard 상위권을 차지합니다.

### 2단계: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

기본 template은 "This example is about {label}."입니다. `hypothesis_template`로 customize할 수 있습니다. training data가 필요 없습니다. fine-tuning도 필요 없습니다. 바로 작동합니다.

### 3단계: RAG를 위한 faithfulness check

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

이것이 RAGAS faithfulness의 핵심입니다. generated answer를 atomic claim으로 나눕니다. 각 claim을 retrieved context에 대해 검사합니다. entail되는 비율을 report합니다.

### 4단계: 손으로 만든 NLI classifier(개념용)

`code/main.py`에는 stdlib-only toy가 있습니다. premise와 hypothesis를 lexical overlap + negation detection으로 비교합니다. transformer model과 경쟁할 수준은 아니지만, 이 작업의 형태를 보여 줍니다. text 두 개가 들어가고, 3-way label이 나오며, loss = `{entail, contradict, neutral}`에 대한 cross-entropy입니다.

## 함정

- **Hypothesis-only shortcut.** "not", "nobody", "never"가 contradiction과 상관되어 있기 때문에 model은 hypothesis만 보고도 SNLI에서 약 60% 수준으로 label을 예측할 수 있습니다. label leakage를 감지하는 강한 baseline입니다.
- **Lexical overlap heuristic.** subsequence heuristic("모든 subsequence는 entailed된다")은 SNLI는 통과하지만 HANS/ANLI에서는 실패합니다. adversarial benchmark를 사용하세요.
- **Document-length degradation.** single-sentence NLI model은 document-length premise에서 F1이 20+ 떨어집니다. long context에는 DocNLI-trained model을 사용하세요.
- **Zero-shot template sensitivity.** "This example is about {label}" vs "{label}" vs "The topic is {label}"만 바꿔도 accuracy가 10+ point 흔들릴 수 있습니다. template을 tune하세요.
- **Domain mismatch.** MNLI는 일반 영어로 train됩니다. legal, medical, scientific text에는 domain-specific NLI model이 필요합니다(예: SciNLI, MedNLI).

## 사용하기

2026년 stack:

| 사용 사례 | Model |
|---------|-------|
| General-purpose NLI | `microsoft/deberta-v3-large-mnli` |
| Fast / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification (lightweight) | `facebook/bart-large-mnli` |
| Document-level NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Multilingual | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Hallucination detection in RAG | RAGAS / DeepEval 내부의 NLI layer |

2026년 meta-pattern: NLI는 text understanding의 duct tape입니다. "A가 B를 support하는가?" 또는 "A가 B와 contradict하는가?"가 필요할 때는, 또 다른 LLM call을 하기 전에 NLI부터 사용하세요.

## 출시하기

`outputs/skill-nli-picker.md`로 저장하세요.

```markdown
---
name: nli-picker
description: classification / faithfulness / zero-shot task를 위한 NLI model, label template, evaluation setup을 고릅니다.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

use case(faithfulness check, zero-shot classification, document-level inference)가 주어지면 다음을 출력하세요.

1. 모델. 이름이 있는 NLI checkpoint. domain, length, language와 연결된 이유.
2. 템플릿(zero-shot인 경우). verbalization pattern. 예시.
3. 임계값. decision rule을 위한 entailment cutoff. calibration에 근거한 이유.
4. 평가. held-out labeled set의 accuracy, hypothesis-only baseline, adversarial subset.

100-example labeled sanity check 없이 zero-shot classification을 출시하지 마세요. document-length premise에 sentence-level NLI model을 사용하지 마세요. NLI가 hallucination을 해결한다는 claim은 flag하세요. NLI는 hallucination을 줄일 뿐, 제거하지는 않습니다.
```

## 연습문제

1. **쉬움.** 세 class를 모두 포함하는 hand-crafted `(premise, hypothesis, label)` triple 20개에 `facebook/bart-large-mnli`를 실행하세요. accuracy를 측정하세요. adversarial "subsequence heuristic" trap("I did not eat the cake" vs "I ate the cake")을 추가하고 깨지는지 확인하세요.
2. **보통.** AG News headline 100개에서 zero-shot template `"This text is about {label}"`을 `"The topic is {label}"`, `"{label}"`과 비교하세요. accuracy swing을 report하세요.
3. **어려움.** RAG faithfulness checker를 만드세요. atomic-claim decomposition + claim별 NLI를 사용합니다. gold context가 있는 RAG-generated answer 50개에서 evaluate하세요. hand label 대비 false-positive와 false-negative rate를 측정하세요.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | premise-hypothesis 관계의 3-way classification. |
| RTE | Recognizing Textual Entailment | NLI의 예전 이름이며 같은 task입니다. |
| Entailment | "`t` implies `h`" | 일반적인 독자가 `t`가 주어졌을 때 `h`가 참이라고 결론 내립니다. |
| Contradiction | "`t` rules out `h`" | 일반적인 독자가 `t`가 주어졌을 때 `h`가 거짓이라고 결론 내립니다. |
| Neutral | "undecided" | `t`에서 `h`로 어느 방향의 inference도 없습니다. |
| Zero-shot classification | classifier로 쓰는 NLI | label을 hypothesis로 verbalize하고 max entailment를 고릅니다. |
| Faithfulness | 답이 support되는가? | (retrieved context, generated answer)에 대한 NLI입니다. |

## 더 읽을거리

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI benchmark.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026년 NLI workhorse.
