---
name: memory-blocks
description: Critical path 밖의 sleep-time consolidation agent를 포함해 Letta 모양의 three-tier memory system(core blocks, recall, archival)을 생성합니다.
version: 1.0.0
phase: 14
lesson: 08
tags: [memory, letta, blocks, sleep-time, consolidation]
---

Target runtime, primary model, 그리고 (가능하면 더 강한) sleep-time model이 주어지면, 명시적인 block type과 async consolidation을 갖춘 three-tier memory system을 만드세요.

생성할 것:

1. `label`, `value`, `limit`, `description`, `version`, `history`를 가진 `Block` type. 모든 write는 version을 올리고 old value를 기록합니다. `near_limit(threshold=0.8)`을 노출하세요.
2. 최소 세 default block을 가진 `BlockStore`: `human`(사용자에 대한 fact), `persona`(agent self-concept), `task`(현재 scope). 사용자 정의 block을 허용하세요.
3. `Recall` store — session별로 pagination되는 turn log. 모든 turn을 자동으로 write합니다. Tail은 cap에서 evict되지만 retrievable 상태로 남습니다.
4. `Archival` store — 최소 두 backend(vector, KV). Insert는 record id를 반환합니다. Contradiction이 있으면 delete가 아니라 invalidate하세요.
5. Turn을 처리하고 raw write만 발행하는 `PrimaryAgent`. Critical path에서 summarization을 하지 마세요.
6. Turn 사이에 실행되는 `SleepTimeAgent`: threshold를 넘은 block을 요약하고, contradicted archival record를 무효화하며, `learned_context`를 shared block에 씁니다.

Hard rejects:

- Direct lookup을 제외하고 user-facing turn 중 동기적으로 실행되는 모든 memory op. Summarization, consolidation, invalidation은 sleep-time pass에 속합니다.
- Contradiction 시 archival record를 delete하는 것. History가 audit 가능하게 남도록 invalidate하세요.
- Review step 없이 Persona 또는 Safety block에 write하는 것. 이 block들은 전역 behavior를 형성하므로 silent write는 bug를 숨깁니다.

Refusal rules:

- Runtime이 session을 넘어 block을 persist할 수 없다면, "memory"라고 설명되는 product의 ship을 거절하세요. Claim을 낮추세요.
- Sleep-time agent에 trace output이 없다면 거절하세요. Silent consolidation은 debugging dead-zone입니다.
- 사용자가 "invalidation 없이 항상 latest write를 trust"하라고 요청하면, historical claim이 중요한 모든 domain(compliance, medical, legal)에서는 거절하세요.

Output: component마다 파일 하나와, default block, sleep-time cadence, contradiction resolution policy를 이름 붙여 설명하는 `README.md`. 마지막에는 agent가 memory 위의 graph reasoning이 필요하면 Lesson 09를, product가 memory op에 OTel span이 필요하면 Lesson 23을 가리키는 "what to read next"로 끝내세요.
