---
name: reflexion-buffer
description: TTL, dedup, scoped scope를 갖춘 verbal RL용 episodic-memory reflection buffer를 유지합니다.
version: 1.0.0
phase: 14
lesson: 03
tags: [reflexion, episodic-memory, self-healing, verbal-rl, sleep-time]
---

task class(반복되는 종류의 agent run, 예: "refactor a function", "close a support ticket")가 주어지면 reflection의 episodic-memory buffer를 유지하세요. 각 reflection은 failure mode와 corrective insight를 자연어로 기록합니다. buffer는 같은 task class의 다음 trial 앞에 prepended됩니다.

생성할 것:

1. Reflection capture. trial이 threshold 아래 evaluator score로 끝나면 "I failed to do X because Y; next time, Z." 형태의 one-line reflection을 emit하세요. external failure(network, upstream 500s)에 대한 reflection은 재현 가능하지 않다면 버리세요.
2. TTL and dedup. reflection은 기본적으로 N trials 후 만료됩니다(10 권장). exact duplicate는 collapse합니다. near-duplicate(작은 embedding model에서 >0.9 cosine, 또는 shared substring >= 80%)는 가장 최근 것만 유지합니다.
3. Scope policy. 세 scope: task-class(task name별), user(같은 user의 task 전체), agent(모든 user 전체). 기본값은 task-class입니다. reflection이 user-specific preference를 참조할 때만 user scope로 escalate하고, agent scope로는 절대 자동 escalate하지 마세요.
4. Compaction. buffer가 budget을 넘으면 sleep-time compaction을 실행하세요. near-duplicate를 cluster, summarize, merge합니다. compaction은 hot path 밖에서 실행됩니다. primary agent response를 지연시키지 마세요.
5. Prompt integration. "What I learned from prior trials"라는 제목의 단일 block과 bulleted list를 emit하세요. prompt에는 6 item으로 cap을 두고 overflow는 별도 summary item("... and 4 older reflections about timeouts")으로 보냅니다.

강한 거부:

- reflection을 "be more careful next time"으로 쓰기. actionable하지 않습니다. concrete next-time instruction을 강제하는 prompt로 reflector를 다시 실행하세요.
- wall-clock time 기준으로 reflection 만료. TTL은 offline-replayable run을 위해 time-scoped가 아니라 trial-scoped여야 합니다.
- secret(API keys, tokens, PII)을 참조하는 reflection 저장. buffer에 commit하기 전에 구체적인 "contains secret" class error로 reject하세요.

거부 규칙:

- evaluator가 붙어 있지 않으면 거부하고 Lesson 05(Self-Refine/CRITIC)를 권하세요. reflection에는 gut feeling이 아니라 signal이 필요합니다.
- task class가 one-shot(반복되지 않음)이면 거부하세요. episodic memory는 반복되지 않는 task에는 아무것도 하지 못합니다.

출력: structured buffer file(JSON reflection object: trial id, task class, scope, text, created_at, ttl_remaining), 다음 trial용 prompt block, 곧 만료될 entry를 나열하는 "stale reflections" report.

마지막에는 buffer가 계속 cap에 도달하면 Lesson 06(context compression), compaction을 hot path 밖으로 옮기려면 Lesson 08(Letta sleep-time compute)을 가리키는 "what to read next" note를 붙이세요.
