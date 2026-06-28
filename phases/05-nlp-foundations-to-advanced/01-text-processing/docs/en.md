# 텍스트 처리 - 토큰화, 스테밍, 표제어 추출

> 언어는 연속적이다. 모델은 이산적이다. 전처리는 그 둘을 잇는 다리다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## 문제

모델은 "The cats were running."을 그대로 읽지 못한다. 모델이 읽는 것은 정수다.

모든 NLP 시스템은 같은 세 가지 질문에서 시작한다. 단어는 어디에서 시작하는가. 단어의 어근은 무엇인가. 도움이 될 때는 "run", "running", "ran"을 어떻게 같은 것으로 다루고, 그렇지 않을 때는 어떻게 다른 것으로 다룰 것인가.

토큰화를 잘못하면 모델은 쓰레기에서 학습한다. tokenizer가 `don't`를 하나의 token으로 보는데 다른 곳에서는 `do n't`를 두 token으로 본다면, 학습 분포가 갈라진다. stemmer가 `organization`과 `organ`을 같은 stem으로 접어 버리면 topic modeling은 망가진다. lemmatizer가 품사 문맥을 필요로 하는데 이를 넘기지 않으면 동사가 명사처럼 처리된다.

이 lesson에서는 세 가지 전처리 단계를 처음부터 만든 다음, NLTK와 spaCy가 같은 일을 어떻게 수행하는지 보여 주어 tradeoff를 볼 수 있게 한다.

## 개념

세 가지 연산이 있다. 각각에는 역할과 실패 모드가 있다.

**Tokenization**은 문자열을 token으로 나눈다. "Token"이라는 말은 일부러 느슨하다. 적절한 단위는 task에 따라 달라지기 때문이다. 고전 NLP에서는 단어 수준. transformer에서는 subword. 공백이 없는 언어에서는 문자 단위.

**Stemming**은 규칙으로 접미사를 잘라낸다. 빠르고, 공격적이고, 단순하다. `running -> run`. `organization -> organ`. 두 번째 예가 실패 모드다.

**Lemmatization**은 문법 지식을 사용해 단어를 사전형으로 줄인다. 더 느리지만 정확하며, lookup table이나 형태소 분석기가 필요하다. `ran -> run` ("ran"이 "run"의 과거형임을 알아야 한다). `better -> good` (비교급 형태를 알아야 한다).

경험칙. 속도가 중요하고 noise를 감수할 수 있으면 stem을 사용한다(search indexing, rough classification). 의미가 중요하면 lemmatize한다(question answering, semantic search, 사용자가 읽을 모든 것).

```figure
edit-distance
```

## 직접 만들기

### 1단계: regex word tokenizer

가장 단순하면서도 쓸 만한 tokenizer는 punctuation을 독립 token으로 유지하면서 비영숫자 문자 기준으로 나눈다. 완벽하지도 최종형도 아니지만 한 줄로 실행된다.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

우선순위대로 세 가지 pattern이 있다. 내부 apostrophe를 선택적으로 허용하는 단어(`don't`, `it's`). 순수 숫자. 공백도 영숫자도 아닌 단일 문자를 독립 token으로 처리하는 pattern(punctuation).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

눈여겨볼 실패 모드. `3pm`은 문자 run과 숫자 run을 번갈아 처리하기 때문에 `['3', 'pm']`으로 나뉜다. 대부분의 task에는 충분하다. URL, email, hashtag는 모두 깨진다. production에서는 일반 pattern보다 앞에 전용 pattern을 추가한다.

### 2단계: Porter stemmer(step 1a만)

전체 Porter algorithm에는 다섯 단계의 규칙이 있다. Step 1a만으로도 가장 흔한 영어 접미사를 다루며 pattern을 배울 수 있다.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

규칙은 위에서 아래로 읽는다. `ies -> i` 규칙 때문에 `ponies -> poni`가 되며, `pony`가 되지 않는다. 실제 Porter에는 이를 고치는 step 1b가 있다. 규칙은 서로 경쟁한다. 앞선 규칙이 이긴다. 순서는 어떤 단일 규칙보다 중요하다.

### 3단계: lookup 기반 lemmatizer

제대로 된 lemmatization에는 morphology가 필요하다. 가르치기 좋은 버전은 작은 lemma table과 fallback을 사용한다.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

마지막 예가 핵심 학습 지점이다. `watched`는 table에 없고 fallback은 `ing`만 처리한다. 실제 lemmatization은 `ed`, 불규칙 동사, 비교급 형용사, 소리 변화가 있는 복수형(`children -> child`)을 다룬다. production system이 WordNet, spaCy의 morphologizer, 또는 전체 morphological analyzer를 쓰는 이유가 이것이다.

### 4단계: 파이프라인으로 묶기

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

빠진 조각은 POS tagger다. Phase 5 · 07 (POS Tagging)에서 하나를 만든다. 지금은 모든 것을 기본값 `NOUN`으로 두고 한계를 인정한다.

## 사용하기

NLTK와 spaCy는 production 버전을 제공한다. 각각 몇 줄이면 된다.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize`는 regex가 놓치는 contractions, Unicode, edge case를 처리한다. `PorterStemmer`는 다섯 단계를 모두 실행한다. `WordNetLemmatizer`는 POS tag를 NLTK의 Penn Treebank scheme에서 WordNet의 abbreviation set으로 번역해야 한다. 위의 변환 코드가 대부분의 tutorial이 건너뛰는 부분이다.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```text
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy는 전체 pipeline을 `nlp(text)` 뒤에 숨긴다. Tokenization, POS tagging, lemmatization이 모두 실행된다. 규모가 커지면 NLTK보다 빠르다. 기본 정확도도 더 높다. tradeoff는 개별 component를 쉽게 바꾸기 어렵다는 점이다.

### 무엇을 고를까

| 상황 | 선택 |
|-----------|------|
| 교육, 연구, component 교체 | NLTK |
| production, 다국어, 속도가 중요함 | spaCy |
| Transformer pipeline(어차피 model tokenizer로 tokenize함) | `tokenizers` / `transformers`를 사용하고 classical preprocessing은 건너뛰기 |

### 아무도 경고하지 않는 두 가지 실패 모드

대부분의 tutorial은 algorithm을 가르치고 멈춘다. 실제 preprocessing pipeline을 물어뜯는 두 가지가 있는데, 거의 다루지 않는다.

**Reproducibility drift.** NLTK와 spaCy는 version 사이에서 tokenization과 lemmatizer behavior를 바꾼다. spaCy 2.x에서 `['do', "n't"]`를 만들던 것이 3.x에서는 `["don't"]`를 만들 수 있다. 모델은 한 분포에서 학습했다. inference는 이제 다른 분포에서 실행된다. 정확도는 조용히 떨어지고 아무도 이유를 모른다. `requirements.txt`에서 library version을 고정한다. 샘플 문장 20개의 expected tokenization을 고정하는 preprocessing regression test를 작성한다. upgrade할 때마다 실행한다.

**Training / inference mismatch.** 공격적 preprocessing(lowercase, stopword removal, stemming)으로 학습하고, raw user input으로 배포하면 성능이 무너진다. 이것은 production NLP에서 가장 흔한 실패다. 학습 때 preprocess했다면 inference 때도 동일한 함수를 실행해야 한다. preprocessing을 serving team이 다시 쓰는 notebook cell이 아니라 model package 내부의 function으로 배포한다.

## 내보내기

세 권의 교과서를 읽지 않고도 engineer가 preprocessing strategy를 고를 수 있게 돕는 재사용 가능한 prompt다.

`outputs/prompt-preprocessing-advisor.md`로 저장한다:

```markdown
---
name: preprocessing-advisor
description: NLP task에 맞는 tokenization, stemming, lemmatization 설정을 추천한다.
phase: 5
lesson: 01
---

당신은 classical NLP preprocessing에 대해 조언한다. task description이 주어지면 다음을 출력한다:

1. Tokenization 선택(regex, NLTK word_tokenize, spaCy, 또는 transformer tokenizer). 이유를 설명한다.
2. Stem, lemmatize, 둘 다, 또는 둘 다 하지 않을지. 이유를 설명한다.
3. 구체적인 library call. function 이름을 말한다. NLTK가 관련되어 있으면 POS-tag translation을 인용한다.
4. 사용자가 test해야 할 실패 모드 하나.

사용자에게 보이는 text에는 stemming을 추천하지 않는다. POS tag 없이 lemmatization을 추천하지 않는다. 비영어 input은 다른 pipeline이 필요하다고 표시한다.
```

## 연습문제

1. **Easy.** URL을 단일 token으로 유지하도록 `tokenize`를 확장하라. Test: `tokenize("Visit https://example.com today.")`는 URL token 하나를 만들어야 한다.
2. **Medium.** Porter step 1b를 구현하라. 단어에 vowel이 있고 `ed` 또는 `ing`로 끝나면 제거한다. double-consonant rule을 처리하라(`hopping -> hop`, `hopp`가 아님).
3. **Hard.** WordNet을 lookup table로 사용하되 WordNet에 entry가 없으면 Porter stemmer로 fallback하는 lemmatizer를 만들라. tagged corpus에서 plain WordNet 및 plain Porter와 accuracy를 비교하라.

## 핵심 용어

| 용어 | 사람들이 말하는 뜻 | 실제 의미 |
|------|-----------------|-----------------------|
| Token | 단어 | 모델이 소비하는 어떤 단위든 가능하다. word, subword, character, byte일 수 있다. |
| Stem | 단어의 어근 | 규칙 기반 suffix stripping의 결과. 항상 실제 단어는 아니다. |
| Lemma | 사전형 | 사전에서 찾아볼 형태. 정확히 계산하려면 문법적 문맥이 필요하다. |
| POS tag | 품사 | NOUN, VERB, ADJ 같은 category. 정확한 lemmatization에 필요하다. |
| Morphology | 단어 형태 규칙 | tense, number, case에 따라 단어가 형태를 바꾸는 방식. Lemmatization은 여기에 의존한다. |

## 더 읽을거리

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) - 원 논문. 다섯 쪽이며 여전히 가장 명확한 설명이다.
- [spaCy 101 - linguistic features](https://spacy.io/usage/linguistic-features) - 실제 pipeline이 어떻게 연결되는지 보여 준다.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) - 아직 생각해 보지 못했을 tokenization edge case들.
