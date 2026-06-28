# POS 태깅과 구문 파싱

> 한동안 문법은 유행에서 밀려났습니다. 그러다 모든 LLM 파이프라인이 구조화 추출을 검증해야 하게 되면서 다시 돌아왔습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제

Lesson 01에서는 표제어 추출에 part-of-speech tag가 필요하다고 했습니다. `running`이 동사라는 것을 모르면 lemmatizer는 그것을 `run`으로 줄일 수 없습니다. `better`가 형용사라는 것을 모르면 `good`으로 줄일 수 없습니다.

그 말 뒤에는 하나의 하위 분야 전체가 숨어 있었습니다. Part-of-speech tagging은 문법 범주를 할당합니다. Syntactic parsing은 문장의 트리 구조를 복원합니다. 어떤 단어가 어떤 단어를 수식하는지, 어떤 동사가 어떤 argument를 지배하는지 찾습니다. 고전 NLP는 이 둘을 다듬는 데 20년을 보냈습니다. 그러다 딥러닝이 이를 pretrained transformer 위의 token-classification task로 압축했고, 연구 커뮤니티는 다음 주제로 넘어갔습니다.

하지만 적용 커뮤니티는 그렇지 않습니다. 모든 structured-extraction pipeline은 여전히 내부에서 POS와 dependency tree를 사용합니다. LLM이 생성한 JSON은 문법 제약에 맞춰 검증됩니다. Question-answering system은 dependency parse로 query를 분해합니다. Machine translation 품질 평가기는 parse tree의 alignment를 확인합니다.

알아둘 가치가 있습니다. 이 lesson은 tagset, baseline, 그리고 처음부터 구현하기를 멈추고 spaCy를 호출해야 하는 지점을 소개합니다.

## 개념

**POS tagging**은 각 token에 문법 범주를 붙입니다. **Penn Treebank (PTB)** tagset은 영어의 기본값입니다. 36개 tag가 있고, 일반 독자에게는 지나치게 세밀해 보이는 구분을 합니다. `NN`은 단수 명사, `NNS`는 복수 명사, `NNP`는 단수 고유명사, `VBD`는 과거시제 동사, `VBZ`는 3인칭 단수 현재 동사입니다. **Universal Dependencies (UD)** tagset은 더 거칠고(17개 tag) 언어에 독립적입니다. cross-lingual 작업의 기본값이 되었습니다.

```text
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**Syntactic parsing**은 트리를 만듭니다. 주요 방식은 두 가지입니다.

- **Constituency parsing.** 명사구, 동사구, 전치사구가 서로 안에 중첩됩니다. 출력은 non-terminal category(NP, VP, PP)를 가진 트리이고, 단어는 leaf입니다.
- **Dependency parsing.** 각 단어는 자신이 의존하는 head word를 하나 가지며, grammatical relation label이 붙습니다. 출력은 모든 edge가 (head, dependent, relation) triple인 트리입니다.

Dependency parsing은 2010년대에 우세해졌습니다. 특히 어순이 자유로운 언어에서도 언어 간 일반화가 깔끔하기 때문입니다.

```text
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 직접 만들기

### 1단계: most-frequent-tag baseline

동작하는 가장 단순한 POS tagger입니다. 각 단어에 대해 training에서 가장 자주 가졌던 tag를 예측합니다.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

Brown corpus에서 이 baseline은 약 85% accuracy에 도달합니다. 좋지는 않지만, 진지한 model이라면 절대 밑돌면 안 되는 바닥선입니다.

### 2단계: bigram HMM tagger

sequence의 joint probability를 model합니다.

```text
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

두 개의 table을 둡니다. transition probability(previous tag가 주어졌을 때 tag), emission probability(tag가 주어졌을 때 word)입니다. 둘 다 count와 Laplace smoothing으로 추정합니다. Viterbi(tag lattice 위의 dynamic programming)로 decode합니다.

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Brown에서 bigram HMM은 약 93% accuracy에 도달합니다. 85%에서 93%로 뛰는 대부분의 이유는 transition probability입니다. model은 `DET NOUN`은 흔하고 `NOUN DET`은 드물다는 것을 학습합니다.

### 3단계: modern tagger가 이것보다 나은 이유

Transition + emission probability는 local합니다. `"I bought a saw"`에서 `saw`가 명사이고 `"I saw the movie."`에서 동사라는 것을 포착할 수 없습니다. 임의 feature(suffix, word shape, 앞뒤 단어, 단어 자체)를 쓰는 CRF는 약 97%에 도달합니다. BiLSTM-CRF나 transformer는 98%+에 도달합니다.

이 task의 상한은 annotator disagreement가 정합니다. Penn Treebank에서 human annotator는 약 97% 정도 일치합니다. 98%를 넘는 model은 test set에 overfitting하고 있을 가능성이 큽니다.

### 4단계: dependency parsing sketch

Dependency parsing 전체를 처음부터 구현하는 것은 범위 밖입니다. 표준 교과서식 설명은 Jurafsky and Martin에 있습니다. 알아둘 고전 계열은 두 가지입니다.

- **Transition-based** parser(arc-eager, arc-standard)는 shift-reduce parser처럼 동작합니다. token을 읽고 stack에 shift한 뒤, arc를 만드는 reduce action을 적용합니다. Greedy decoding이 빠릅니다. 고전 구현은 MaltParser입니다. modern neural version은 Chen and Manning의 transition-based parser입니다.
- **Graph-based** parser(Eisner's algorithm, Dozat-Manning biaffine)는 가능한 모든 head-dependent edge에 score를 매기고 maximum spanning tree를 고릅니다. 더 느리지만 더 정확합니다.

대부분의 applied work에서는 spaCy를 호출하세요.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```text
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

`dep` column을 아래에서 위로 읽으면 문장의 문법 구조가 드러납니다.

## 사용하기

모든 production NLP library는 표준 pipeline의 일부로 POS와 dependency parser를 제공합니다.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). 빠르고 정확하며 tokenization + NER + lemmatization과 통합되어 있습니다. `token.tag_` (Penn), `token.pos_` (UD), `token.dep_` (dependency relation).
- **Stanford NLP (stanza)**. CoreNLP의 Stanford 후속 도구입니다. 60개 이상 언어에서 state-of-the-art입니다.
- **trankit**. Transformer 기반이며 UD accuracy가 좋습니다.
- **NLTK**. `pos_tag`. 사용할 수 있지만 느리고 오래되었습니다. 교육용으로는 충분합니다.

### 2026년에도 이것이 중요한 곳

- **Lemmatization.** Lesson 01은 올바른 lemmatization을 위해 POS가 필요합니다. 항상 그렇습니다.
- **Structured extraction from LLM outputs.** 생성된 문장이 문법 제약(예: subject-verb agreement, required modifier)을 지키는지 검증합니다.
- **Aspect-based sentiment.** Dependency parse는 어떤 adjective가 어떤 noun을 수식하는지 알려줍니다.
- **Query understanding.** `"movies directed by Wes Anderson starring Bill Murray"`는 parse를 통해 structured constraint로 분해됩니다.
- **Cross-lingual transfer.** UD tag와 dependency relation은 언어에 독립적이어서 새 언어의 zero-shot structured analysis를 가능하게 합니다.
- **Low-compute pipelines.** Transformer를 배포할 수 없다면 POS + dependency parse + gazetteer만으로도 놀라울 만큼 멀리 갈 수 있습니다.

## 내보내기

`outputs/skill-grammar-pipeline.md`로 저장하세요.

```markdown
---
name: grammar-pipeline
description: downstream NLP task를 위한 고전적 POS + dependency pipeline을 설계합니다.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

downstream task(information extraction, rewrite validation, query decomposition, lemmatization)가 주어지면 다음을 출력합니다.

1. 사용할 tagset. 영어 전용 legacy pipeline에는 Penn Treebank, multilingual 또는 cross-lingual에는 Universal Dependencies를 사용합니다.
2. Library. 대부분의 production에는 spaCy, academic-grade multilingual에는 stanza, 가장 높은 UD accuracy에는 trankit을 사용합니다. 구체적인 model ID를 말합니다.
3. Integration pattern. library를 호출하고 필요한 attribute(`.pos_`, `.dep_`, `.head`)를 소비하는 3-5줄을 보여줍니다.
4. 테스트할 failure mode. Noun-verb ambiguity(`saw`, `book`, `can`)와 PP-attachment ambiguity는 고전적인 함정입니다. output 20개를 sample로 뽑아 눈으로 확인합니다.

직접 parser를 만들라는 추천은 거부합니다. parser를 처음부터 만드는 일은 application task가 아니라 research project입니다. lowercase/uppercase variant를 처리하지 않고 POS tag를 소비하는 pipeline은 fragile하다고 표시합니다.
```

## 연습 문제

1. **Easy.** 작은 tagged corpus(예: NLTK의 Brown subset)에서 most-frequent-tag baseline을 사용해 held-out sentence의 accuracy를 측정하세요. 약 85% 결과를 확인하세요.
2. **Medium.** 위의 bigram HMM을 training하고 tag별 precision/recall을 보고하세요. HMM이 가장 자주 혼동하는 tag는 무엇인가요?
3. **Hard.** spaCy의 dependency parse를 사용해 1000문장 sample에서 subject-verb-object triple을 추출하세요. 수동으로 label한 50개 triple에서 평가하세요. extraction이 실패하는 곳을 문서화하세요. passives, coordinations, elided subjects에서 자주 실패합니다.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| POS tag | 단어의 유형 | 문법 범주입니다. PTB에는 36개, UD에는 17개가 있습니다. |
| Penn Treebank | 표준 tagset | 영어에 특화되어 있습니다. 동사 시제와 명사 수를 세밀하게 구분합니다. |
| Universal Dependencies | Multilingual tagset | PTB보다 더 거칠며 언어 중립적입니다. cross-lingual work의 기본값입니다. |
| Dependency parse | 문장 트리 | 각 단어는 하나의 head를 가지고, 각 edge는 grammatical relation을 가집니다. |
| Viterbi | Dynamic programming | emission과 transition이 주어졌을 때 가장 확률이 높은 tag sequence를 찾습니다. |
