# Relation Extraction & Knowledge Graph Construction

> NER가 entity를 찾았습니다. Entity linking이 그것들을 고정했습니다. Relation extraction은 그 entity 사이의 edge를 찾습니다. Knowledge graph는 node, edge, 그리고 그 provenance의 합입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 minutes

## 문제

분석가가 읽습니다: "Tim Cook became CEO of Apple in 2011." 여기에는 네 가지 fact가 있습니다.

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction (RE)은 자유 텍스트를 structured triple `(subject, relation, object)`로 바꿉니다. Corpus 전체에서 aggregate하면 knowledge graph가 됩니다. Aggregate하고 query하면 RAG, analytics, 또는 compliance audit을 위한 reasoning substrate가 됩니다.

2026년의 문제는 이렇습니다: LLM은 relation을 열정적으로 추출합니다. 너무 열정적입니다. source text가 뒷받침하지 않는 triple을 hallucinate합니다. provenance가 없으면 실제 triple과 그럴듯한 fiction을 구분할 수 없습니다. 2026년의 답은 AEVS-style anchor-and-verify pipeline입니다.

## 개념

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Triple form.** `(subject_entity, relation_type, object_entity)`. Relation은 closed ontology(Wikidata properties, FIBO, UMLS) 또는 open set(OpenIE-style, anything goes)에서 나옵니다.

**세 가지 extraction 접근.**

1. **Rule / pattern-based.** Hearst pattern: "X such as Y" → `(Y, isA, X)`. 여기에 hand-crafted regex를 더합니다. 취약하지만 precise하고 explainable합니다.
2. **Supervised classifier.** 문장 안의 두 entity mention이 주어지면 fixed set에서 relation을 예측합니다. TACRED, ACE, KBP로 학습합니다. 2015-2022년의 표준입니다.
3. **Generative LLM.** model에 triple을 emit하도록 prompt합니다. 바로 작동합니다. provenance가 필요합니다. 없으면 그럴듯해 보이는 junk를 hallucinate합니다.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** 현재의 hallucination-mitigation framework입니다.

- **Anchor.** 모든 entity span과 relation-phrase span을 exact position으로 식별합니다.
- **Extract.** anchor span에 연결된 triple을 생성합니다.
- **Verify.** 각 triple element를 source text에 다시 match합니다. unsupported 항목은 reject합니다.
- **Supplement.** coverage pass가 anchored span이 누락되지 않게 보장합니다.

Hallucination이 크게 줄어듭니다. compute는 더 필요하지만 audit 가능합니다.

**open-vs-closed tradeoff.**

- **Closed ontology.** Fixed property list(예: Wikidata의 11,000+ properties). Predictable합니다. Queryable합니다. 꾸며내기 어렵습니다.
- **Open IE.** 어떤 verbal phrase든 relation이 됩니다. High recall입니다. Low precision입니다. Query하기 지저분합니다.

Production KG는 보통 섞어서 씁니다: discovery에는 open IE를 사용하고, main graph에 merge하기 전 relation을 closed ontology로 canonicalize합니다.

## 직접 만들기

### 1단계: pattern-based extraction

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

전체 toy extractor는 `code/main.py`를 보세요. Hearst pattern은 debug 가능하기 때문에 domain-specific pipeline에서 여전히 출하됩니다.

### 2단계: supervised relation classification

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL은 seq2seq relation extractor입니다: text를 넣으면 triple을 출력하고, 이미 Wikidata property id 형태입니다. distant-supervision data로 fine-tune되었습니다. 표준 open-weights baseline입니다.

### 3단계: anchoring을 사용하는 LLM-prompted extraction

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

반환된 모든 span을 source와 대조해 verify하세요. `text[start:end] != triple_entity`인 항목은 reject합니다. 이것이 최소 형태의 AEVS "verify" step입니다.

### 4단계: closed ontology로 canonicalize하기

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Canonicalization은 종종 engineering work의 60-80%입니다. 이에 맞게 budget을 잡으세요.

### 5단계: 작은 graph를 만들고 query하기

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

이것이 모든 RAG-over-KG system의 atom입니다. RDF triple store(Blazegraph, Virtuoso), property graph(Neo4j), 또는 vector-augmented graph store로 scale하세요.

## 함정

- **Coreference before RE.** "He founded Apple" - RE는 "he"가 누구인지 알아야 합니다. 먼저 coref를 실행하세요(lesson 24).
- **Entity canonicalization.** "Apple Inc"와 "Apple"은 같은 node로 resolve되어야 합니다. Entity linking을 먼저 하세요(lesson 25).
- **Hallucinated triples.** LLM은 text가 뒷받침하지 않는 triple을 emit합니다. span verification을 강제하세요.
- **Relation canonicalization drift.** Open IE relation은 일관되지 않습니다("was born in," "came from," "is a native of"). canonical id로 collapse하지 않으면 graph를 query할 수 없습니다.
- **Temporal errors.** "Tim Cook is CEO of Apple" - 지금은 참이지만 2005년에는 거짓입니다. 많은 relation은 time-bounded입니다. qualifier(Wikidata의 `P580` start time, `P582` end time)를 사용하세요.
- **Domain mismatch.** REBEL은 Wikipedia로 학습되었습니다. Legal, medical, scientific text에는 domain-fine-tuned RE model이 필요한 경우가 많습니다.

## 사용하기

2026년 stack:

| Situation | Pick |
|-----------|------|
| Fast production, general domain | REBEL or LlamaPred with Wikidata canonicalization |
| Domain-specific (biomed, legal) | SciREX-style domain fine-tune + custom ontology |
| LLM-prompted, audited output | AEVS pipeline: anchor → extract → verify → supplement |
| High-volume news IE | Pattern-based + supervised hybrid |
| Building a KG from scratch | Open IE + manual canonicalization pass |
| Temporal KG | Extract with qualifiers (start/end time, point in time) |

Integration pattern: NER → coref → entity linking → relation extraction → ontology mapping → graph load. 모든 stage가 잠재적인 quality gate입니다.

## 출하하기

`outputs/skill-re-designer.md`로 저장하세요:

```markdown
---
name: re-designer
description: provenance와 canonicalization을 포함한 relation extraction pipeline을 설계합니다.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Corpus(domain, language, volume)와 downstream use(KG-RAG, analytics, compliance)가 주어지면 다음을 출력하세요:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Precision vs recall target에 연결된 reason.
2. Ontology. Closed property list(Wikidata / domain) 또는 canonicalization pass가 있는 open IE.
3. Provenance. 모든 triple은 source char-span + doc id를 포함합니다. Audit에는 타협할 수 없습니다.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. 손으로 label한 triple 200개에서 precision / recall + LLM-extracted sample의 hallucination-rate.

span verification(source provenance)이 없는 LLM-based RE pipeline은 거부하세요. canonicalization 없이 production graph로 흘러가는 open-IE output도 거부하세요. time-bounded relation(employer, spouse, position)에 temporal qualifier가 없는 pipeline은 표시하세요.
```

## 연습문제

1. **Easy.** `code/main.py`의 pattern extractor를 news-article sentence 5개에 실행하세요. precision을 손으로 확인하세요.
2. **Medium.** 같은 sentence에 REBEL(또는 작은 LLM)을 사용하세요. triple을 비교하세요. 어떤 extractor가 precision이 더 높나요? recall은 어느 쪽이 더 높나요?
3. **Hard.** AEVS pipeline을 만드세요: LLM으로 extract하고 source와 span을 대조해 verify합니다. Wikipedia-style sentence 50개에서 verify step 전후 hallucination rate를 측정하세요.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-------------------|-----------|
| Triple | Subject-relation-object | KG의 atomic unit인 `(s, r, o)` tuple입니다. |
| Open IE | 무엇이든 extract | Open-vocabulary relation phrase입니다. high recall, low precision입니다. |
| Closed ontology | Fixed schema | relation type의 bounded set입니다(Wikidata, UMLS, FIBO). |
| Canonicalization | 모든 것을 normalize | surface name / relation을 canonical id로 매핑합니다. |
| AEVS | Grounded extraction | Anchor-Extraction-Verification-Supplement pipeline (2026)입니다. |
| Provenance | Source-of-truth link | 모든 triple은 source로 이어지는 doc id + char-span을 포함합니다. |
| Distant supervision | 저렴한 label | 기존 KG와 text를 align해 training data를 만듭니다. |

## 더 읽을거리

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) - distant-supervision paper.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) - seq2seq RE workhorse.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) - joint IE.
- [AEVS - Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) - 2026 hallucination-mitigation design.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) - canonical graph query.
