# 다국어 NLP

> 하나의 model, 100개가 넘는 language, 그중 대부분은 training data가 0에 가깝다. Cross-lingual transfer는 2020년대의 실용적인 기적이다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 minutes

## 문제

English에는 수십억 개의 labeled example이 있다. Urdu에는 수천 개가 있다. Maithili에는 거의 없다. global audience를 대상으로 하는 실용적인 NLP system은 task-specific training data가 없는 long tail language에서도 동작해야 한다.

Multilingual model은 하나의 model을 여러 language에서 동시에 train해 이 문제를 해결한다. shared representation 덕분에 model은 high-resource language에서 배운 skill을 low-resource language로 transfer할 수 있다. English sentiment analysis로 model을 fine-tune하면, Urdu에서도 별도 설정 없이 놀랄 만큼 좋은 sentiment prediction을 만든다. 이것이 zero-shot cross-lingual transfer이며, NLP가 세계에 배포되는 방식을 바꾸었다.

이 lesson은 tradeoff, canonical model, 그리고 multilingual work에 새로 들어온 team이 자주 걸려 넘어지는 한 가지 decision을 다룬다. transfer를 위한 source language 선택이다.

## 개념

![shared multilingual embedding space를 통한 cross-lingual transfer](../assets/multilingual.svg)

**Shared vocabulary.** Multilingual model은 모든 target language의 text로 train한 SentencePiece 또는 WordPiece tokenizer를 사용한다. vocabulary는 공유된다. 같은 subword unit이 관련 language 전반에서 같은 morpheme을 나타낸다. English와 Italian의 `anti-`는 같은 token을 얻는다.

**Shared representation.** 여러 language에 걸친 masked language modeling으로 pretrain된 transformer는 서로 다른 language의 semantically similar sentence가 비슷한 hidden state를 만든다는 것을 학습한다. mBERT, XLM-R, NLLB 모두 이를 보인다. English의 "cat" embedding은 French의 "chat", Spanish의 "gato" 근처에 cluster되고, full-sentence embedding도 마찬가지다.

**Zero-shot transfer.** 하나의 language(보통 English)의 labeled data로 model을 fine-tune한다. inference 때는 model이 support하는 다른 language에서 실행한다. target-language label은 필요 없다. 결과는 typologically related language에서 강하고, distant language에서는 약하다.

**Few-shot fine-tuning.** target language의 labeled example 100-500개를 추가한다. classification task에서 accuracy가 English baseline의 95-98%까지 올라간다. multilingual NLP에서 가장 cost-effective한 단일 lever다.

## 모델

| 모델 | 연도 | 지원 범위 | 비고 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 languages | Wikipedia로 train. 첫 실용 multilingual LM. low-resource에 약함. |
| XLM-R | 2019 | 100 languages | CommonCrawl(Wikipedia보다 훨씬 큼)로 train. cross-lingual baseline을 정립. Base 270M, Large 550M. |
| XLM-V | 2023 | 100 languages | 1M-token vocabulary를 가진 XLM-R(250k 대비). low-resource에서 더 좋음. |
| mT5 | 2020 | 101 languages | multilingual generation을 위한 T5 architecture. |
| NLLB-200 | 2022 | 200 languages | Meta의 translation model. 55개 low-resource language 포함. |
| BLOOM | 2022 | 46 languages + 13 programming | multilingually train된 open 176B LLM. |
| Aya-23 | 2024 | 23 languages | Cohere의 multilingual LLM. Arabic, Hindi, Swahili에서 강함. |

use case에 따라 고른다. Classification은 sane default로 XLM-R-base가 잘 동작한다. Generation task는 translation인지 open generation인지에 따라 mT5 또는 NLLB가 필요하다. LLM-style work는 explicit multilingual prompting을 사용해 Aya-23 또는 Claude와 잘 맞는다.

## source-language 결정(2026 research)

대부분의 team은 fine-tuning source로 English를 기본값으로 둔다. 최근 research(2026)는 이것이 종종 틀렸음을 보인다.

Language similarity는 raw corpus size보다 transfer quality를 더 잘 예측한다. Slavic target에는 German이나 Russian이 English를 이기는 경우가 많다. Indic target에는 Hindi가 English를 이기는 경우가 많다. **qWALS** similarity metric(2026, World Atlas of Language Structures feature 기반)이 이를 quantify한다. **LANGRANK**(Lin et al., ACL 2019)는 linguistic similarity, corpus size, genetic relatedness의 조합으로 candidate source language를 rank하는 별도의 더 이른 method다.

실용 rule: target language에 typologically close한 high-resource relative가 있다면, 먼저 그 language로 fine-tuning해 보고 English fine-tune과 비교하라.

## 직접 만들기

### 1단계: zero-shot cross-lingual classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

하나의 model, 세 language, 같은 API다. NLI data로 train된 XLM-R은 entailment trick을 통해 classification으로 잘 transfer된다.

### 2단계: multilingual embedding space

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

Translation은 embedding space에서 가깝게 놓인다. 다른 English sentence는 더 멀리 놓인다. 이것이 cross-lingual retrieval, clustering, similarity를 가능하게 한다.

### 3단계: few-shot fine-tuning strategy

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

target-language example 100-500개에는 `num_train_epochs=5`와 `learning_rate=2e-5`가 safe default다. 더 높은 learning rate는 multilingual alignment를 collapse시켜 English-only model을 만들 수 있다.

## 실제로 동작하는 evaluation

- **held-out set의 per-language accuracy.** aggregate하지 않는다. aggregate는 long tail을 숨긴다.
- **monolingual baseline과 benchmark.** data가 충분한 language에서는 from scratch로 train한 monolingual model이 multilingual model을 이기기도 한다. test하라.
- **Entity-level test.** target language의 named entity를 확인한다. Multilingual model은 Latin과 거리가 먼 script에서 tokenization이 약한 경우가 많다.
- **Cross-lingual consistency.** 두 language에서 같은 meaning은 같은 prediction을 만들어야 한다. gap을 측정하라.

## 활용하기

2026년 stack:

| 작업 | 권장 |
|-----|-------------|
| Classification, 100 languages | fine-tuned XLM-R-base(~270M) |
| Zero-shot text classification | `joeddav/xlm-roberta-large-xnli` |
| Multilingual sentence embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Translation, 200 languages | `facebook/nllb-200-distilled-600M`(lesson 11 참고) |
| Generative multilingual | Claude, GPT-4, Aya-23, mT5-XXL |
| Low-resource language NLP | XLM-V 또는 related high-resource language에서 domain-specific fine-tune |

performance가 중요하다면 항상 target language fine-tuning 예산을 잡아라. Zero-shot은 출발점이지 최종 답이 아니다.

### tokenization tax(low-resource language에서 무엇이 잘못되는가)

Multilingual model은 모든 language에 하나의 tokenizer를 공유한다. 그 vocabulary는 English, French, Spanish, Chinese, German이 지배적인 corpus로 train된다. dominant set 밖의 language에는 세 가지 tax가 조용히 누적된다.

- **Fertility tax.** Low-resource language text는 English보다 word당 훨씬 많은 token으로 tokenize된다. Hindi sentence는 동등한 English sentence보다 3-5배 token이 필요할 수 있다. 그 3-5배는 context window, training efficiency, latency를 잡아먹는다.
- **Variant recovery tax.** 모든 typo, diacritic variant, Unicode normalization mismatch, case variation은 embedding space에서 cold-start unrelated sequence가 된다. model은 native speaker에게 당연한 orthographic correspondence를 학습하지 못한다.
- **Capacity spillover tax.** tax 1과 2가 context position, layer depth, embedding dimension을 소비한다. 실제 reasoning에 남는 capacity는 같은 model에서 high-resource language가 얻는 것보다 체계적으로 작다.

실용적인 symptom은 이렇다. model은 Hindi에서 정상적으로 train되고, loss curve도 맞아 보이고, eval perplexity도 reasonable해 보이지만, production output은 미묘하게 틀린다. Morphology가 문장 중간에서 collapse한다. rare inflection은 끝내 recover되지 않는다. **깨진 tokenizer는 data scale로 해결할 수 없다.**

Mitigation: target language coverage가 좋은 tokenizer를 고른다(XLM-V의 1M-token vocabulary는 직접적인 fix다). training 전에 held-out target text에서 tokenization fertility를 verify한다. 진짜 long-tail script에는 byte-level fallback(SentencePiece `byte_fallback=True`, GPT-2-style byte-level BPE)을 사용해 어떤 것도 OOV가 되지 않게 한다.

## 배포하기

`outputs/skill-multilingual-picker.md`로 저장하라:

```markdown
---
name: multilingual-picker
description: multilingual NLP task의 source language, target model, evaluation plan을 고른다.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

requirement(target language, task type, language별 available labeled data)가 주어지면 다음을 output하라:

1. fine-tuning을 위한 source language. default는 English다. target language에 typologically close한 high-resource language가 있으면 LANGRANK 또는 qWALS를 check한다.
2. Base model. XLM-R(classification), mT5(generation), NLLB(translation), Aya-23(generative LLM).
3. Few-shot budget. 가능하다면 target-language example 100-500개로 시작한다. labeling이 infeasible할 때만 zero-shot을 사용한다.
4. Evaluation 계획. per-language accuracy(aggregate 아님), cross-lingual consistency, non-Latin script의 entity-level F1.

per-language evaluation 없이 multilingual model을 ship하는 것을 거부하라. aggregate metric은 long-tail failure를 숨긴다. tokenization coverage가 낮은 script(Amharic, Tigrinya, 여러 African language)는 byte-fallback을 갖춘 model(SentencePiece with byte_fallback=True, 또는 GPT-2 같은 byte-level tokenizer)이 필요하다고 flag하라.
```

## 연습문제

1. **쉬움.** English, French, Hindi, Arabic 각각 language별 10개 sentence에서 zero-shot classification pipeline을 실행하라. 각각의 accuracy를 report하라. French는 강하고, Hindi는 괜찮고, Arabic은 variable한 결과를 볼 것이다.
2. **보통.** `paraphrase-multilingual-MiniLM-L12-v2`를 사용해 작은 mixed-language corpus 위에 cross-lingual retriever를 구축하라. English로 query하고 어떤 language의 document든 retrieve하라. recall@5를 측정하라.
3. **어려움.** Hindi classification task에서 English-source와 Hindi-source fine-tuning을 비교하라. 두 regime 모두에서 few-shot fine-tuning에 target-language example 500개를 사용한다. 어떤 source가 Hindi accuracy를 얼마나 더 좋게 만드는지 report하라. 이것은 LANGRANK thesis의 축소판이다.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Multilingual model | 하나의 model, 여러 language | language 전반에서 vocabulary와 parameter를 공유한다. |
| Cross-lingual transfer | 한 language에서 train하고 다른 language에서 실행 | source에서 fine-tune하고 target-language label 없이 target에서 evaluate한다. |
| Zero-shot | target-language label 없음 | target language에서 fine-tuning 없이 transfer한다. |
| Few-shot | 적은 target label | fine-tuning에 사용하는 target-language example 100-500개. |
| mBERT | 첫 multilingual LM | Wikipedia로 pretrained된 104-language BERT. |
| XLM-R | 표준 cross-lingual baseline | CommonCrawl로 pretrained된 100-language RoBERTa. |
| NLLB | Meta의 200-language MT | No Language Left Behind. 55개 low-resource language 포함. |

## 더 읽을거리

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R paper.
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — cross-lingual transfer research line을 시작한 analysis paper.
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 paper.
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Cohere의 multilingual LLM인 Aya.
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK source-language paper.
