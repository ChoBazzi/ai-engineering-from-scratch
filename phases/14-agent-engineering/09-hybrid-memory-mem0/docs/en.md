# Hybrid Memory: Vector + Graph + KV (Mem0)

> Mem0(Chhikara et al., 2025)는 memory를 병렬로 놓인 세 store로 다룹니다. Semantic similarity를 위한 vector, 빠른 fact lookup을 위한 KV, entity-relationship reasoning을 위한 graph입니다. Retrieval 시 scoring layer가 셋을 fuse합니다. 이것이 external memory의 2026년 production standard입니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- 단일 store(vector only, graph only, KV only)가 agent memory에 충분하지 않은 이유를 설명할 수 있습니다.
- Mem0의 세 parallel store와 각 store가 최적화하는 대상을 말할 수 있습니다.
- Mem0의 fusion scoring(relevance, importance, recency)과 그것이 hierarchy가 아니라 weighted sum인 이유를 설명할 수 있습니다.
- 모든 store에 write하는 `add()`와 result를 fuse하는 `search()`를 가진 toy three-store memory를 stdlib로 구현할 수 있습니다.

## 문제

하나의 store는 세 query class 중 하나에는 틀립니다.

- **Semantic similarity** — "지난주 agent drift에 대해 무엇을 논의했지?" Vector가 이깁니다. KV와 graph는 놓칩니다.
- **Fact lookup** — "사용자의 전화번호가 뭐지?" KV가 이깁니다. Vector는 낭비이고 graph는 과합니다.
- **Relationship reasoning** — "어떤 customer들이 같은 billing entity를 공유하지?" Graph가 이깁니다. Vector와 KV는 답할 수 없습니다.

Production agent는 한 session에서 셋 모두를 요청합니다. Single-store memory는 항상 그중 둘에 대해 틀립니다. Mem0의 기여는 세 store를 하나의 `add`/`search` surface 뒤에 연결하고, 이들을 fuse하는 scoring function을 제공한 것입니다.

## 개념

### 병렬 세 store

Mem0(arXiv:2504.19413, 2025년 4월)는 `add(text, user_id, metadata)`에서 다음을 수행합니다.

1. Text에서 candidate fact를 추출합니다(LLM-driven step).
2. Semantic search를 위해 각 fact를 vector store(embedding)에 씁니다.
3. O(1) lookup을 위해 각 fact를 (user_id, fact_type, entity)를 key로 KV store에 씁니다.
4. Relationship query를 위해 각 fact를 typed edge로 graph store(Mem0g)에 씁니다.

`search(query, user_id)`에서는 다음을 수행합니다.

1. Vector store가 embedding cosine 기준 top-k를 반환합니다.
2. KV store가 query에서 파생된 (user_id, type, entity) key의 direct hit를 반환합니다.
3. Graph store가 query entity에서 도달 가능한 subgraph를 반환합니다.
4. Scoring layer가 셋을 fuse합니다.

### Fusion scoring

```text
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — vector cosine, KV exact match, graph path weight.
- **Importance** — write 시점에 tag되거나 학습됩니다. 어떤 fact(name, ID, policy)는 더 중요합니다.
- **Recency** — last write 또는 read 이후 시간에 대한 exponential decay.

Weight는 product별로 tune합니다. Chat agent는 `w_recency`를 더 높이고, compliance agent는 `w_importance`를 더 높이며, retrieval agent는 `w_relevance`를 더 높입니다.

### Mem0g와 temporal reasoning

Mem0g는 conflict detector를 추가합니다. 새 fact가 기존 edge와 모순되면 기존 edge는 delete되지 않고 invalid로 표시됩니다. Temporal query("3월에 사용자의 도시는 어디였지?")는 valid-at-time subgraph를 traverse합니다.

이것은 Letta의 invalidation pattern이 일반화하는 compliance-grade behavior입니다.

### Benchmark number

Mem0 paper가 보고한 수치(2025):

- **LoCoMo**(long-form conversation memory): 91.6
- **LongMemEval**(long-horizon episodic memory): 93.4
- **BEAM 1M**(1M-token memory benchmark): 64.1

Comparison baseline(full-context 128k LLM, flat vector store, flat KV)은 모두 10점 이상 뒤처집니다. Benchmark만으로 선택이 정당화되지는 않습니다. Operational shape가 중요합니다. 하지만 이 수치는 fusion design이 rounding error가 아님을 보여 줍니다.

### Scope taxonomy

Mem0는 memory를 scope별로 나눕니다.

- **User memory** — session을 넘어 persist되며 `user_id`로 keying됩니다.
- **Session memory** — 한 thread 안에서 persist됩니다.
- **Agent memory** — agent instance별 state입니다.

모든 write는 하나의 scope를 선택합니다. Retrieval은 scope별 weight를 적용해 여러 scope를 query할 수 있습니다. 생각 없이 scope를 섞으면 "assistant가 Alice에게 Bob의 project를 말해 버린" 사고가 납니다.

### 이 패턴이 잘못되는 지점

- **Embedding drift.** 처음 백 개 query에서는 맞아 보이던 vector result가 corpus가 커질수록 저하됩니다. Top-N-used record를 주기적으로 re-embedding하세요.
- **KV schema creep.** `(user_id, type, entity)`는 단순해 보이지만 모든 팀이 자기만의 `type`을 추가하기 시작합니다. Type set을 분기마다 audit하세요.
- **Graph explosion.** Noisy extractor 하나가 message당 50개 edge를 추가합니다. `add` call당 graph write 수를 제한하고 low-confidence edge는 버리세요.

## 직접 만들기

`code/main.py`는 stdlib로 three-store pattern을 구현합니다.

- `VectorStore` — embedding 대역으로 naive token-overlap similarity를 사용합니다.
- `KVStore` — `(user_id, fact_type, entity)`를 key로 하는 dict입니다.
- `GraphStore` — typed edge(subject, relation, object, valid)입니다.
- `Mem0` — `add()`, `search()`, fusion scoring, scope-aware retrieval을 가진 top-level facade입니다.
- Multi-user, multi-session conversation에 대한 worked trace.

실행:

```bash
python3 code/main.py
```

출력은 서로 다른 세 recall path와 fused top-k를 보여 줍니다. `main()` 상단의 scoring weight를 바꿔 ranking이 어떻게 바뀌는지 보세요.

## 활용하기

- **Mem0(Apache 2.0)** — production-ready입니다. Postgres + Qdrant + Neo4j로 self-host하거나 managed cloud를 사용합니다.
- **Letta** — three-tier core/recall/archival 구조입니다. Vector와 graph backend는 직접 가져옵니다.
- **Zep** — temporal KG와 fact extraction을 갖춘 commercial alternative입니다.
- **Custom builds** — extractor(compliance)나 fusion weight(voice agent처럼 recency가 지배적인 경우)를 정확히 제어해야 할 때 사용합니다.

## 출시하기

`outputs/skill-hybrid-memory.md`는 fusion scorer, scope taxonomy, temporal invalidation이 연결된 three-store memory scaffold를 생성합니다.

## 연습 문제

1. Toy vector similarity를 실제 embedding model(sentence-transformers, Ollama, OpenAI embeddings)로 바꾸세요. Synthetic long conversation에서 recall@10을 측정하세요. 1000 write 이후 ranking이 drift하나요?
2. Temporal query `search(query, as_of=timestamp)`를 추가하세요. 그 시점 또는 그 이전에 valid한 record만 반환하세요. 어느 store가 가장 많은 작업을 필요로 하나요?
3. Conflict detector를 구현하세요. Incoming fact가 graph edge와 모순되면 old edge를 invalidate하고 둘 다 log하세요. "user lives in Berlin" -> "user lives in Lisbon"으로 test하세요.
4. Fusion scorer에 `user_feedback` dimension(retrieved record에 thumbs-up)을 포함하도록 port하세요. Gaming(agent가 이미 좋아한 record만 반환하는 현상)을 어떻게 막나요?
5. Mem0 docs(`docs.mem0.ai`)를 읽으세요. Toy를 `mem0` client call로 port하세요. 같은 20개 test query에서 retrieval quality를 비교하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| Hybrid memory | "Vector plus graph plus KV" | 세 store에 병렬로 write하고 retrieval에서 fuse하는 구조 |
| Fact extraction | "Memory ingestion" | Text를 (entity, relation, fact) tuple로 나누는 LLM step |
| Fusion scoring | "Relevance ranking" | Relevance, importance, recency의 weighted sum |
| Scope | "Memory namespace" | user / session / agent — 누가 무엇을 보는지 결정함 |
| Mem0g | "Memory graph" | Relationship query를 위한 temporal validity가 있는 typed edge |
| Temporal invalidation | "Soft delete" | 모순된 edge를 invalid로 표시하고 delete하지 않음 |
| Embedding drift | "Retrieval rot" | Corpus가 커질수록 vector quality가 저하됨. 주기적으로 re-embed함 |

## 더 읽기

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — original paper
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API, SDK, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — virtual-context predecessor
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — three-tier sibling design
