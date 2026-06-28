---
name: skill-library
description: Registration, similarity 기반 retrieval, compositional execution, failure-driven refinement를 갖춘 Voyager 모양 skill library를 생성합니다.
version: 1.0.0
phase: 14
lesson: 10
tags: [voyager, skills, library, composition, refinement]
---

Target runtime과 domain이 주어지면 Voyager의 세 component인 curriculum hook, retrievable skill store, iterative refinement를 지원하는 skill library를 만드세요.

생성할 것:

1. `name`, `description`, `code`, `version`, `tags`, `depends_on`, `history`를 가진 `Skill` type. 모든 write는 prior code를 기록합니다.
2. `register(skill, dedup=True)`(new 또는 version bump), `search(query, top_k, tag_filter)`, `get(name)`, `topo_order(name)`(dep resolution), `execute(name, context)`(topological run)을 가진 `SkillLibrary`.
3. Retrieval은 full library 위의 LLM scoring이 아니라 반드시 embedding similarity 또는 BM25를 사용해야 합니다. Top-k shortlist에 대한 LLM re-rank는 허용됩니다.
4. Execution은 skill별 exception을 catch하고 refinement loop가 소비할 수 있는 feedback으로 trace에 노출해야 합니다.
5. Refinement hook: failed `execute` 이후 runtime은 (task, skill_name, error, env_state)를 수집해 model에 전달하고 rewritten skill에 대해 `register`를 호출합니다. Version은 bump되고 history는 old code를 보존합니다.

Hard rejects:

- Skill이 code가 아니라 prose string인 library. Skill은 executable입니다. Prose는 `description`에 속합니다.
- Topological sort 없는 composition. Cycle detection 없는 depth-first는 skill DAG에서 깨집니다.
- Silent version overwrite. 모든 refinement는 반드시 `version`을 bump하고 audit을 위해 old code를 `history`에 push해야 합니다.

Refusal rules:

- Target runtime에 skill execution sandbox가 없다면, skill이 production system을 건드리는 domain에서는 거절하세요. Ship 전에 sandbox(Lesson 09 principles)를 요구하세요.
- 사용자가 "refinement 없이 모든 failure에 auto-retry"를 요청하면 거절하세요. Refinement 없는 retry는 bug를 증폭할 뿐 고치지 않습니다.
- Library가 약 200개 skill을 넘는데 flat retrieval만 있다면 "production-ready"라고 부르기를 거절하세요. 먼저 tag filter와 hierarchical namespace를 추가하세요.

Output: `skill.py`, `library.py`, `execute.py`, `refine.py`, 그리고 dedup rule, retrieval backend, refinement prompt, version policy를 설명하는 `README.md`. 마지막에는 Claude Agent SDK integration은 Lesson 17, OpenAI Agents SDK tool translation은 Lesson 16, skill-library quality 평가는 Lesson 30을 가리키는 "what to read next"로 끝내세요.
