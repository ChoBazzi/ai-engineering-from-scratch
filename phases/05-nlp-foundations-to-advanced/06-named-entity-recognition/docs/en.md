# 개체명 인식 (Named Entity Recognition)

> 이름을 뽑아내세요. 모호한 boundary, nested entity, domain jargon을 만나기 전까지는 쉬워 보입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 minutes

## 문제

"Apple sued Google over its iPhone search deal in the US." Entity는 다섯 개입니다. Apple(ORG), Google(ORG), iPhone(PRODUCT), search deal(아마도), US(GPE). 좋은 NER system은 이들을 모두 올바른 type으로 extract합니다. 나쁜 system은 iPhone을 놓치고, 과일 Apple과 회사 Apple을 혼동하며, "US"를 PERSON으로 label합니다.

NER은 모든 structured extraction pipeline 아래에 있는 workhorse입니다. Resume parsing, compliance log scanning, medical record anonymization, search query understanding, chatbot response grounding, legal contract extraction. 눈에 잘 보이지는 않지만 항상 의존하게 됩니다.

이 lesson은 고전적인 경로(rule-based, HMM, CRF)에서 현대적인 경로(BiLSTM-CRF, 그다음 transformer)로 걸어갑니다. 각 단계는 이전 단계의 특정 limitation을 해결합니다. 그 pattern이 이 lesson의 핵심입니다.

## 개념

**BIO tagging**(또는 BILOU)은 entity extraction을 sequence-labeling problem으로 바꿉니다. 각 token에 `B-TYPE`(entity의 beginning), `I-TYPE`(entity 내부), 또는 `O`(어떤 entity에도 속하지 않음)를 label합니다.

```text
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

Multi-token entity는 chain으로 이어집니다. `New B-GPE`, `York I-GPE`, `City I-GPE`. BIO를 이해하는 model은 임의의 span을 extract할 수 있습니다.

Architecture progression은 다음과 같습니다.

- **Rule-based.** Regex + gazetteer lookup입니다. 알려진 entity에서는 high precision이지만, 새 entity에는 coverage가 없습니다.
- **HMM.** Hidden Markov Model입니다. Tag가 주어졌을 때 token의 emission probability, tag-to-tag transition probability, Viterbi decode를 사용합니다. Labeled data로 train합니다.
- **CRF.** Conditional Random Field입니다. HMM과 비슷하지만 discriminative라서 임의의 feature(word shape, capitalization, neighboring word)를 섞을 수 있습니다. 2026년에도 low-resource deployment에서 고전적인 production workhorse입니다.
- **BiLSTM-CRF.** Hand-crafted feature 대신 neural feature를 씁니다. LSTM이 sentence를 양방향으로 읽고, 위의 CRF layer가 일관된 tag sequence를 강제합니다.
- **Transformer-based.** Token-classification head로 BERT를 fine-tune합니다. Accuracy가 가장 좋고 compute가 가장 큽니다.

```figure
ner-bio-tagging
```

## 직접 만들기

### 1단계: BIO tagging helper

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 2단계: hand-crafted feature

고전적인(non-neural) NER에서는 feature가 승부처입니다. 유용한 feature는 다음과 같습니다.

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")`은 `xXxxxx`를 return합니다. `word_shape("USA-2024")`는 `XXX-dddd`를 return합니다. Capitalization pattern은 proper noun에 high-signal입니다.

### 3단계: 간단한 rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

Production gazetteer에는 Wikipedia와 DBpedia에서 긁은 수백만 entry가 있습니다. Coverage는 좋습니다. Disambiguation(`Apple` 회사 vs 과일)은 형편없습니다. 그래서 statistical model이 이겼습니다.

### 4단계: CRF 단계(sketch, full impl 아님)

처음부터 50줄짜리 full CRF를 만드는 것은 probability-theory foundation 없이는 별로 유익하지 않습니다. 대신 `sklearn-crfsuite`를 사용하세요.

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1`과 `c2`는 L1 및 L2 regularization입니다. `all_possible_transitions=True`는 모델이 illegal sequence(예: `O` 뒤의 `I-ORG`)가 unlikely하다는 것을 학습하게 해 줍니다. 이것이 CRF가 constraint를 직접 작성하지 않아도 BIO consistency를 enforce하는 방식입니다.

### 5단계: BiLSTM-CRF가 추가하는 것

Feature가 learned feature가 됩니다. Input은 token embedding(GloVe 또는 fastText)입니다. LSTM은 left-to-right와 right-to-left로 sentence를 읽습니다. Concatenated hidden state가 CRF output layer를 통과합니다. CRF는 여전히 tag-sequence consistency를 enforce하고, LSTM은 hand-crafted feature를 learned feature로 대체합니다.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF layer에는 `torchcrf.CRF`를 사용하세요(`pip install pytorch-crf`). Hand-crafted CRF 대비 gain은 측정 가능하지만, labeled sentence가 수만 개 있지 않으면 생각보다 작습니다.

## 사용하기

spaCy는 production-grade NER을 기본으로 제공합니다.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```text
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

`iPhone`이 `PRODUCT`가 아니라 `ORG`로 label된 점에 주목하세요. spaCy의 small model은 product-entity coverage가 약합니다. Large model(`en_core_web_lg`)은 더 낫습니다. Transformer model(`en_core_web_trf`)은 더 낫습니다.

BERT-based NER에는 Hugging Face를 사용합니다.

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```text
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"`은 contiguous B-X, I-X token을 하나의 span으로 merge합니다. 이것이 없으면 token-level label을 받고 직접 merge해야 합니다.

### LLM-based NER(2026년의 선택지)

Zero-shot 및 few-shot LLM NER은 이제 많은 domain에서 fine-tuned model과 경쟁 가능하며, labeled data가 부족할 때 훨씬 더 좋습니다.

- **Zero-shot prompting.** LLM에 entity type list와 example schema를 줍니다. JSON output을 요청합니다. 바로 작동하지만 novel domain에서 accuracy는 moderate합니다.
- **ZeroTuneBio-style prompting.** Task를 candidate extraction → meaning explanation → judgment → re-check로 분해합니다. One-shot이 아닌 multi-stage prompt는 biomedical NER에서 accuracy를 크게 올립니다. 같은 pattern은 legal, financial, scientific domain에서도 작동합니다.
- **RAG를 사용한 dynamic prompting.** 작은 annotated seed set에서 inference call마다 가장 비슷한 labeled example을 retrieve하고, few-shot prompt를 즉석에서 만듭니다. 2026 benchmark에서 이것은 GPT-4 biomedical NER F1을 static prompting 대비 11-12% 올립니다.
- **Per-entity-type decomposition.** 긴 document에서는 모든 entity type을 한 번에 extract하는 single call이 length가 길어질수록 recall을 잃습니다. Entity type마다 extraction pass를 하나씩 실행하세요. Inference cost는 높지만 accuracy가 상당히 높아집니다. Clinical note와 legal contract의 standard pattern입니다.

2026년 기준 production recommendation: training data를 수집하기 전에 LLM zero-shot baseline으로 시작하세요. F1이 충분히 좋아 fine-tune이 아예 필요 없는 경우가 많습니다.

### 고전 NER이 여전히 이기는 곳

LLM을 사용할 수 있어도 고전 NER은 다음 상황에서 이깁니다.

- Latency budget이 50ms 미만입니다.
- Labeled example이 수천 개 있고 98%+ F1이 필요합니다.
- Domain에 stable ontology가 있고 pretrained CRF 또는 BiLSTM이 잘 transfer됩니다.
- Regulatory constraint가 on-prem, non-generative model을 요구합니다.

### 무너지는 곳

- **Domain shift.** CoNLL-trained NER을 legal contract에 적용하면 gazetteer보다 못합니다. Domain에서 fine-tune하세요.
- **Nested entities.** "Bank of America Tower"는 동시에 ORG이자 FACILITY입니다. Standard BIO는 overlapping span을 표현할 수 없습니다. Nested NER(multi-pass 또는 span-based model)이 필요합니다.
- **Long entities.** "United States Federal Deposit Insurance Corporation." Token-level model은 때때로 이것을 쪼갭니다. `aggregation_strategy`를 사용하거나 post-process하세요.
- **Sparse types.** Medical NER label에는 DRUG_BRAND, ADVERSE_EVENT, DOSE 같은 것이 있습니다. General-purpose model은 전혀 모릅니다. 여기서는 Scispacy와 BioBERT가 출발점입니다.

## 출시하기

`outputs/skill-ner-picker.md`로 저장하세요.

```markdown
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
2. Starting model. 이름을 적으세요(spaCy model ID, Hugging Face checkpoint ID, 또는 "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, 또는 span-based. 한 문장으로 justify하세요.
4. Evaluation. `seqeval`을 사용하세요. Token-level이 아니라 entity-level F1을 항상 report하세요.

Labeled example이 500개 미만이면 user에게 pretrained domain model이 이미 있는 경우를 제외하고 transformer fine-tuning을 추천하지 마세요. Nested entity는 span-based 또는 multi-pass model이 필요하다고 flag하세요. User가 "production scale"을 언급하고 label이 CoNLL-2003에서 변경되지 않았다면 gazetteer audit을 요구하세요.
```

## 연습문제

1. **쉬움.** `bio_to_spans`(`spans_to_bio`의 inverse)를 구현하고 sentence 10개에서 round-trip consistency를 verify하세요.
2. **보통.** 위의 sklearn-crfsuite CRF를 CoNLL-2003 English NER dataset에서 train하세요. `seqeval`로 per-entity F1을 report하세요. Typical result: ~84 F1입니다.
3. **어려움.** Domain-specific NER dataset(medical, legal, 또는 financial)에서 `distilbert-base-cased`를 fine-tune하세요. spaCy small model과 비교하세요. Data leakage check를 문서화하고 무엇이 놀라웠는지 write-up하세요.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|--------------------|-----------|
| NER | 이름 추출 | Token span에 type(PERSON, ORG, GPE, DATE, ...)을 label합니다. |
| BIO | Tagging scheme | `B-X`는 시작, `I-X`는 continuation, `O`는 outside입니다. |
| BILOU | 더 나은 BIO | 더 깔끔한 boundary를 위해 `L-X`(last), `U-X`(unit)를 추가합니다. |
| CRF | Structured classifier | Emission만이 아니라 label 간 transition을 model합니다. Valid sequence를 enforce합니다. |
| Nested NER | Overlapping entity | 한 span이 그 sub-span과 다른 entity인 경우입니다. BIO는 이것을 표현할 수 없습니다. |
| Entity-level F1 | 올바른 NER metric | Predicted span이 true span과 정확히 match해야 합니다. Token-level F1은 accuracy를 과장합니다. |

## 더 읽을거리

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF paper입니다. Canonical입니다.
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — standard가 된 token-classification pattern을 소개합니다.
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents`와 `Span`의 모든 attribute에 대한 practical reference입니다.
- [seqeval](https://github.com/chakki-works/seqeval) — 올바른 metric library입니다. 항상 사용하세요.
