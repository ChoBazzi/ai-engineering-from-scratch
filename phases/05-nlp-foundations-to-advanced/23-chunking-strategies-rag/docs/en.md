# RAG를 위한 청킹 전략

> 청킹 설정은 embedding model 선택만큼 retrieval 품질에 영향을 준다(Vectara NAACL 2025). 청킹을 잘못하면 아무리 reranking을 해도 구제하기 어렵다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 minutes

## 문제

50페이지짜리 계약서를 RAG 시스템에 넣었다. 사용자가 묻는다. "해지 조항은 무엇인가?" retriever가 표지를 반환한다. 왜 그럴까? model은 512-token chunk로 학습되었고, 해지 조항은 20페이지 뒤에 있으며, 페이지 구분을 사이에 두고 쪼개져 있고, query와 연결되는 local keyword도 없기 때문이다.

해결책은 "더 나은 embedding model을 사는 것"이 아니다. 해결책은 청킹이다. 얼마나 크게 할 것인가? overlap은 줄 것인가? 어디서 split할 것인가? surrounding context를 붙일 것인가?

2026년 2월 benchmark는 놀라운 결과를 보여준다.

- Vectara의 2026년 연구: recursive 512-token chunking이 semantic chunking을 69% -> 54% accuracy로 앞섰다.
- Natural Questions에서 SPLADE + Mistral-8B: overlap은 측정 가능한 이점을 전혀 제공하지 않았다.
- Context cliff: context가 약 2,500 tokens에 이르면 response quality가 급격히 떨어진다.

"명백한" 답(semantic chunking, 20% overlap, 1000 tokens)은 자주 틀린다. 이 lesson은 여섯 가지 전략에 대한 직관을 만들고, 어떤 상황에서 무엇을 선택해야 하는지 알려준다.

## 개념

![한 passage 위에 시각화한 여섯 가지 청킹 전략](../assets/chunking.svg)

**Fixed chunking.** N characters 또는 tokens마다 split한다. 가장 단순한 baseline이다. 문장 중간에서 끊긴다. compression은 좋지만 coherence는 나쁘다.

**Recursive.** LangChain의 `RecursiveCharacterTextSplitter`. 먼저 `\n\n`로 split을 시도하고, 그다음 `\n`, `.`, space 순서로 시도한다. 깔끔하게 fallback한다. 2026년의 default다.

**Semantic.** 각 sentence를 embed한다. 인접 sentence 사이의 cosine similarity를 계산한다. similarity가 threshold 아래로 떨어지는 지점에서 split한다. topic coherence를 보존한다. 더 느리고, 때로 retrieval을 해치는 40-token짜리 작은 fragment를 만든다.

**Sentence.** sentence boundary에서 split한다. chunk당 sentence 하나를 쓰거나 N sentences window를 쓴다. 비용은 훨씬 낮으면서 약 5k tokens까지 semantic chunking과 비슷하게 동작한다.

**Parent-document.** retrieval에는 작은 child chunk를 저장하고, context에는 더 큰 parent chunk를 저장한다. child로 retrieve하고 parent를 반환한다. 품질이 완만히 저하된다. child chunk가 나빠도 합리적인 parent를 반환한다.

**Late chunking (2024).** 먼저 전체 document를 token level에서 embed한 뒤, token embedding을 chunk embedding으로 pool한다. cross-chunk context를 보존한다. long-context embedder(BGE-M3, Jina v3)와 잘 맞는다. compute가 더 많이 든다.

**Contextual retrieval (Anthropic, 2024).** 각 chunk 앞에 문서 내 위치에 대한 LLM-generated summary를 붙인다("This chunk is section 3.2 of the termination clauses..."). Anthropic 자체 benchmark에서 35-50% retrieval improvement를 보였다. indexing 비용이 비싸다.

### 모든 default를 이기는 규칙

chunk size를 query type에 맞춰라.

| Query type | Chunk size |
|------------|-----------|
| Factoid ("CEO의 이름은 무엇인가?") | 256-512 tokens |
| Analytical / multi-hop | 512-1024 tokens |
| Whole-section comprehension | 1024-2048 tokens |

NVIDIA의 2026년 benchmark다. chunk는 answer와 local context를 담을 만큼 충분히 커야 하고, retriever의 top-K가 context noise보다 answer에 집중할 만큼 충분히 작아야 한다.

## 직접 만들기

### 1단계: fixed와 recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 2단계: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

도메인에 맞춰 `threshold`를 조정하라. 너무 높으면 fragment가 된다. 너무 낮으면 하나의 거대한 chunk가 된다.

### 3단계: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

핵심 통찰: parent를 dedupe하라. 여러 child가 같은 parent에 mapping될 수 있다. 전부 반환하면 context를 낭비한다.

### 4단계: contextual retrieval (Anthropic pattern)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

contextualized chunk를 index하라. query time에는 추가된 주변 signal 덕분에 retrieval이 좋아진다.

### 5단계: evaluate

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

항상 benchmark하라. corpus에 맞는 "best" strategy는 어떤 blog post와도 다를 수 있다.

## 함정

- **Factoid query만으로 chunking을 평가함.** Multi-hop query는 매우 다른 winner를 드러낸다. query-type-stratified eval set을 사용하라.
- **minimum size 없는 semantic chunking.** retrieval을 해치는 40-token fragment를 만든다. 항상 `min_tokens`를 강제하라.
- **Cargo cult로 쓰는 overlap.** 2026년 연구들은 overlap이 종종 아무 이점 없이 index cost만 두 배로 늘린다고 본다. 추정하지 말고 측정하라.
- **Min/max enforcement 없음.** 5 tokens짜리 chunk와 5000 tokens짜리 chunk는 둘 다 retrieval을 망가뜨린다. clamp하라.
- **Cross-doc chunking.** chunk가 두 document에 걸치게 만들지 말라. 항상 doc별로 chunk한 뒤 merge하라.

## 사용하기

2026년 stack:

| Situation | Strategy |
|-----------|----------|
| 첫 build, 알 수 없는 corpus | Recursive, 512 tokens, no overlap |
| Factoid QA | Recursive, 256-512 tokens |
| Analytical / multi-hop | Recursive, 512-1024 tokens + parent-document |
| 강한 cross-reference(contracts, papers) | Late chunking 또는 contextual retrieval |
| Conversational / dialog corpus | Turn-level chunks + speaker metadata |
| 짧은 utterance(tweets, reviews) | One document = one chunk |

recursive 512로 시작하라. 50-query eval set에서 recall@5를 측정하라. 거기서부터 조정하라.

## 배포하기

`outputs/skill-chunker.md`로 저장하라.

```markdown
---
name: chunker
description: 주어진 corpus와 query distribution에 맞는 chunking strategy, size, overlap을 고른다.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

corpus(document types, avg length, domain)와 query distribution(factoid / analytical / multi-hop)이 주어지면 다음을 출력하라.

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

min/max chunk size enforcement가 없는 chunking strategy는 거부하라. 도움이 된다는 ablation 없이 20%를 넘는 overlap은 거부하라. min-token floor가 없는 semantic chunking recommendation은 flag하라.
```

## 연습문제

1. **Easy.** 하나의 20-page document를 fixed(512, 0), recursive(512, 0), recursive(512, 100)으로 chunk하라. chunk 수와 boundary quality를 비교하라.
2. **Medium.** 5 documents 위에 30-query eval set을 만들라. recursive, semantic, parent-document에 대해 recall@5를 측정하라. 무엇이 이기는가? blog post와 일치하는가?
3. **Hard.** contextual retrieval을 구현하라. baseline recursive 대비 MRR improvement를 측정하라. index cost(LLM calls)와 accuracy gain을 함께 보고하라.

## 핵심 용어

| Term | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Chunk | doc의 한 조각 | embed, index, retrieve되는 sub-document unit. |
| Overlap | 안전 여백 | 인접 chunk 사이에 공유되는 N tokens. 2026 benchmark에서는 자주 쓸모없다. |
| Semantic chunking | 똑똑한 chunking | 인접 sentence embedding similarity가 떨어지는 지점에서 split한다. |
| Parent-document | 2단계 retrieval | 작은 child를 retrieve하고 더 큰 parent를 반환한다. |
| Late chunking | embedding 후 chunk | 전체 doc을 token level에서 embed하고 chunk vector로 pool한다. |
| Contextual retrieval | Anthropic의 기법 | indexing 전에 각 chunk 앞에 LLM-generated summary를 붙인다. |
| Context cliff | 2500-token 벽 | RAG에서 약 2.5k context tokens 부근에 관찰된 quality drop(Jan 2026). |

## 더 읽을거리

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — production의 default.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking은 embedding choice만큼 중요하다.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking paper.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — LLM-generated context prefix로 35-50% retrieval improvement.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — query type별 chunk size.
