# Entity Linking & Disambiguation

> NER가 "Paris"를 찾았습니다. Entity linking은 이것이 무엇인지 결정합니다: Paris, France인가? Paris Hilton인가? Paris, Texas인가? Paris(트로이 왕자)인가? 링크하지 않으면 knowledge graph는 계속 모호한 상태로 남습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 minutes

## 문제

문장이 이렇게 나옵니다: "Jordan beat the press." NER는 "Jordan"을 PERSON으로 태깅합니다. 좋습니다. 하지만 *어떤* Jordan일까요?

- Michael Jordan (basketball)?
- Michael B. Jordan (actor)?
- Michael I. Jordan (Berkeley ML professor - 맞습니다, ML 논문에서도 실제로 생기는 혼동입니다)?
- Jordan (the country)?
- Jordan (Hebrew first name)?

Entity linking (EL)은 각 mention을 knowledge base의 고유 항목으로 해소합니다: Wikidata, Wikipedia, DBpedia, 또는 도메인 KB입니다. 하위 작업은 두 가지입니다.

1. **Candidate generation.** "Jordan"이 주어졌을 때 어떤 KB 항목들이 그럴듯한가?
2. **Disambiguation.** context가 주어졌을 때 어떤 candidate가 맞는가?

두 단계 모두 학습 가능합니다. 둘 다 benchmark가 있습니다. 결합된 pipeline은 10년 동안 안정적이었습니다. 달라지는 것은 disambiguator의 품질입니다.

## 개념

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation.** mention surface form("Jordan")이 주어지면 alias index에서 candidate를 조회합니다. Wikipedia alias dictionary는 대부분의 named entity를 포괄합니다: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK airport, JFK (movie). 일반적인 index는 mention마다 10-30개의 candidate를 반환합니다.

**Disambiguation: 세 가지 접근.**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. 잘 작동하고, 빠르며, training이 필요 없습니다.
2. **Embedding-based (ESS / REL / Blink).** mention + context를 encode합니다. 각 candidate의 description도 encode합니다. cosine이 최대인 항목을 고릅니다. 2020-2024년의 기본 선택입니다.
3. **Generative (GENRE, 2021; LLM-based, 2023+).** entity의 canonical name을 token-by-token으로 decode합니다. 유효한 entity name의 trie로 constrained decoding하므로 출력이 유효한 KB id임이 보장됩니다.

**End-to-end vs pipeline.** 최신 모델(ELQ, BLINK, ExtEnD, GENRE)은 NER + candidate generation + disambiguation을 한 번에 수행합니다. 하지만 production에서는 component를 교체할 수 있기 때문에 pipeline system이 여전히 우세합니다.

### 두 가지 측정값

- **Mention recall (candidate gen).** gold mention 중 정답 KB entry가 candidate list에 나타나는 비율입니다. 전체 pipeline의 하한입니다.
- **Disambiguation accuracy / F1.** 올바른 candidate가 주어졌을 때 top-1이 얼마나 자주 맞는가입니다.

항상 둘 다 보고하세요. 80% candidate recall에서 99% disambiguation을 내는 system은 80% pipeline입니다.

## 직접 만들기

### 1단계: Wikipedia redirect에서 alias index 만들기

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia alias data: 약 18M개의 (alias, entity) pair입니다. Wikidata dump에서 내려받습니다. inverted index로 저장하세요.

### 2단계: context 기반 disambiguation

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap은 장난감 예시입니다. embedding의 cosine similarity로 바꾸세요(transformer version은 `code/main.py`의 step-2 참고).

### 3단계: embedding-based (BLINK-style)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Index time에는 모든 KB entity를 한 번씩 embed합니다. Query time에는 mention + context를 한 번 embed하고, candidate pool에 대해 dot product를 계산한 뒤 최댓값을 고릅니다.

### 4단계: generative entity linking (개념)

GENRE는 entity의 Wikipedia title을 character-by-character로 decode합니다. Constrained decoding(lesson 20 참고)은 유효한 title만 출력될 수 있게 보장합니다. KB-backed trie와 긴밀히 통합됩니다. 최신 후속 방식은 REL-GEN과 structured output을 사용하는 LLM-prompted EL입니다.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

whitelist(Outlines `choice`)와 결합하면 2026년에 출하하기 가장 단순한 EL pipeline입니다.

### 5단계: AIDA-CoNLL에서 평가하기

AIDA-CoNLL은 표준 EL benchmark입니다: Reuters article 1,393개, mention 34k개, Wikipedia entity로 구성됩니다. in-KB accuracy(`P@1`)와 out-of-KB NIL-detection rate를 보고하세요.

## 함정

- **NIL handling.** 일부 mention은 KB에 없습니다(새로 등장한 entity, 잘 알려지지 않은 사람). System은 잘못된 entity를 추측하는 대신 NIL을 예측해야 합니다. 별도로 측정합니다.
- **Mention boundary errors.** upstream NER가 partial span을 놓칩니다("Bank of America"가 "Bank"만 태깅됨). EL recall이 떨어집니다.
- **Popularity bias.** 학습된 system은 빈번한 entity를 과다 예측합니다. ML 논문에서 "Michael I. Jordan"이 언급되어도 basketball Jordan으로 링크되는 일이 자주 생깁니다.
- **Cross-lingual EL.** 중국어 text의 mention을 English Wikipedia entity에 매핑합니다. multilingual encoder 또는 translation step이 필요합니다.
- **KB staleness.** 새 회사, 사건, 인물은 작년 Wikipedia dump에 없습니다. Production pipeline에는 refresh loop가 필요합니다.

## 사용하기

2026년 stack:

| Situation | Pick |
|-----------|------|
| General-purpose English + Wikipedia | BLINK or REL |
| Cross-lingual, KB = Wikipedia | mGENRE |
| LLM-friendly, few mentions/day | Prompt Claude/GPT-4 with candidate list + constrained JSON |
| Domain-specific KB (medical, legal) | Custom BERT with KB-aware retrieval + fine-tune on domain AIDA-style set |
| Extremely low-latency | Exact-match prior only (Milne-Witten baseline) |
| Research SOTA | GENRE / ExtEnD / generative LLM-EL |

2026년에 출하되는 production pattern: NER → coref → 각 mention에 EL 적용 → cluster를 cluster당 하나의 canonical entity로 collapse. 출력은 mention마다 하나가 아니라 document의 entity마다 하나의 KB id입니다.

## 출하하기

`outputs/skill-entity-linker.md`로 저장하세요:

```markdown
---
name: entity-linker
description: Entity linking pipeline을 설계합니다 - KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Use case(domain KB, language, volume, latency budget)가 주어지면 다음을 출력하세요:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

mention-recall baseline이 없는 EL pipeline은 거부하세요(candidate gen이 올바른 entity를 surfacing했는지 모르면 disambiguator를 평가할 수 없습니다). 유효한 KB id로 constrained output하지 않는 LLM-prompted EL pipeline도 거부하세요. popularity bias가 minority entity(예: name-clashes)에 영향을 주는데 domain fine-tuning이 없는 system은 표시하세요.
```

## 연습문제

1. **Easy.** `code/main.py`에서 10개의 ambiguous mention(Paris, Jordan, Apple)에 prior+context disambiguator를 구현하세요. 올바른 entity를 손으로 label하세요. accuracy를 측정하세요.
2. **Medium.** sentence transformer로 50개의 ambiguous mention을 encode하세요. 각 candidate description을 embed하세요. embedding-based disambiguation을 Jaccard context overlap과 비교하세요.
3. **Hard.** 1k-entity domain KB를 만드세요(예: 회사의 employee + product). NER + EL end-to-end를 구현하세요. 100개의 held-out sentence에서 precision과 recall을 측정하세요.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Entity linking (EL) | Wikipedia에 링크하기 | mention을 고유한 KB entry에 매핑합니다. |
| Candidate generation | 누구일 수 있나? | mention에 대해 그럴듯한 KB entry shortlist를 반환합니다. |
| Disambiguation | 맞는 것을 고르기 | context를 사용해 candidate를 score하고 winner를 고릅니다. |
| Alias index | lookup table | surface form → candidate entities로 매핑합니다. |
| NIL | KB에 없음 | 일치하는 KB entry가 없다는 명시적 prediction입니다. |
| KB | Knowledge base | Wikidata, Wikipedia, DBpedia, 또는 도메인 KB입니다. |
| AIDA-CoNLL | benchmark | gold entity link가 있는 Reuters article 1,393개입니다. |

## 더 읽을거리

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) - foundational prior+context approach.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) - embedding-based workhorse.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) - constrained decoding을 사용하는 generative EL.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) - benchmark paper.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) - open production stack.
