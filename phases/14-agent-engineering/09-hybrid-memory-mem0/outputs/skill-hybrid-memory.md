---
name: hybrid-memory
description: Fusion scorer, scope taxonomy, temporal invalidation을 갖춘 Mem0 모양의 three-store memory system(vector + KV + graph)을 생성합니다.
version: 1.0.0
phase: 14
lesson: 09
tags: [memory, mem0, vector, graph, kv, fusion, scope]
---

Target runtime, vector backend(Qdrant, pgvector, Chroma, sqlite-vec), KV backend(Postgres, Redis, dict), graph backend(Neo4j, in-memory edges)가 주어지면 fused memory system을 만드세요.

생성할 것:

1. `add(text, user_id, session_id, scope, importance, tags)` facade 뒤의 세 store class. Write 시 extractor는 `text`를 record, KV triple, graph triple로 분해합니다. 어떤 store도 optional이 아닙니다.
2. Fusion scorer `score = w_rel * relevance + w_imp * importance + w_rec * recency`. 세 weight를 모두 config로 노출하세요. Call별이 아니라 product별로 tune하세요.
3. Scope taxonomy: `user`, `session`, `agent`. Retrieval은 반드시 scope를 존중해야 합니다. User query가 다른 사용자의 record를 leak해서는 안 됩니다.
4. Temporal invalidation. Contradiction은 old edge/record를 invalid로 표시하며 절대 delete하지 않습니다. Historical query를 위해 `search(query, as_of=timestamp)`를 노출하세요.
5. Extractor interface. Default는 LLM-driven이어도 됩니다. Test를 위해 deterministic regex fallback을 허용하세요. Explosion을 막기 위해 `add()`당 graph edge 수를 제한하세요.

Hard rejects:

- "Mem0-shaped"라고 설명되는 single-store memory. Vector-only, KV-only, graph-only product 자체는 괜찮지만 hybrid memory는 아닙니다. 잘못 이름 붙이지 마세요.
- Scope별 weight 또는 명시적인 `scope=` filter 없는 cross-scope retrieval. Scope leak은 compliance 및 privacy incident입니다.
- Contradiction에서 delete하는 것. Invalidate하고 timestamp를 남기세요. Deletion은 bug를 숨기고 audit을 깨뜨립니다.

Refusal rules:

- 사용자가 "importance weighting 없음"을 요청하면 거절하세요. 백만 record 위의 flat relevance ranking은 retrieval failure를 기다리는 상태입니다.
- Graph backend에 conflict detector가 없다면 결과 system을 "Mem0-shaped"라고 부르기를 거절하세요. 이름을 낮추세요.
- Product가 PII(medical, legal, HR)를 포함한다면 product owner가 audit하지 않은 extractor로 ship하는 것을 거절하세요.

Output: store마다 파일 하나와 `memory.py`(facade), `config.py`(weights), fusion weight, scope policy, extractor contract, invalidation semantics를 설명하는 `README.md`. 마지막에는 agent가 새 skill을 배워야 하면 Lesson 10을, memory op에 OTel span이 필요하면 Lesson 23을, retrieval의 untrusted-input handling은 Lesson 27을 가리키는 "what to read next"로 끝내세요.
