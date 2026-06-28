# Memory: Virtual Context and MemGPT

> Context window는 유한합니다. conversation, document, tool trace는 그렇지 않습니다. MemGPT(Packer et al., 2023)는 이를 OS virtual memory로 frame합니다. main context는 RAM, external store는 disk이고, agent는 둘 사이를 page합니다. 이것은 2026년 모든 memory system이 상속하는 pattern입니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## Learning Objectives

- MemGPT가 기반으로 삼는 OS analogy를 설명합니다. main context = RAM, external context = disk, memory tools = page in/out.
- main-context buffer, external searchable store, page in/out tool을 갖춘 two-tier MemGPT pattern을 stdlib로 구현합니다.
- agent가 external memory를 query하거나 modify하기 위해 "interrupt"를 발행하고, result가 다음 prompt에 splice되는 방식을 설명합니다.
- Letta(Lesson 08)와 Mem0(Lesson 09)로 이어지는 MemGPT design choice를 식별합니다.

## The Problem

context window는 memory를 해결할 것처럼 보입니다. 실제로는 그렇지 않습니다. production에서 세 failure mode가 반복됩니다.

1. **Overflow.** Multi-turn conversation, long document, tool-call-heavy trajectory가 window를 넘습니다. cutoff 너머의 모든 것은 사라집니다.
2. **Dilution.** window 안에서도 irrelevant context를 밀어 넣으면 중요한 것에 대한 attention이 희석됩니다. frontier model도 long input에서 여전히 degrade됩니다.
3. **Persistence.** 새 session은 빈 window로 시작합니다. external memory가 없는 agent는 session을 넘어 "전에 저에게 요청하셨던..."이라고 말할 수 없습니다.

더 큰 window는 도움이 되지만 이 문제를 해결하지는 않습니다. Mem0의 2025년 논문은 128k-window baseline도 external memory를 가진 4k-window agent가 잡아내는 long-horizon fact를 놓친다고 측정했습니다.

## The Concept

### MemGPT: OS analogy

Packer et al.(arXiv:2310.08560, v2 Feb 2024)은 context management를 operating-system virtual memory에 mapping합니다.

| OS concept | MemGPT concept | 2026 production analog |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Anthropic/OpenAI context window |
| Disk | external context | vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | memory tool이 있는 ReAct loop |

agent는 정상적인 ReAct loop를 실행합니다. 한 가지 추가 tool class가 main context 안팎으로 data를 page할 수 있게 합니다.

### 두 tier

- **Main context.** 현재 task를 담는 fixed-size prompt. model에게 항상 보입니다.
- **External context.** tool로 search 가능한 unbounded store. relevant할 때 read하고, fact가 생길 때 write합니다.

원 논문은 base window를 넘는 두 task에서 이 design을 평가했습니다. 100k token보다 긴 document analysis와 여러 날에 걸친 persistent memory가 있는 multi-session chat입니다.

### interrupt pattern

MemGPT는 memory-as-interrupt를 도입합니다. conversation 중 agent가 memory tool을 invoke할 수 있고, runtime이 이를 실행하며, result는 다음 assistant turn에 new observation으로 splice됩니다. 개념적으로 Unix `read()` syscall과 같습니다. process를 block하고, byte를 반환하고, process가 계속됩니다.

canonical memory tool surface:

- `core_memory_append(section, text)` - prompt의 persistent section에 write.
- `core_memory_replace(section, old, new)` - persistent section edit.
- `archival_memory_insert(text)` - searchable external store에 write.
- `archival_memory_search(query, top_k)` - external store에서 retrieve.
- `conversation_search(query)` - past turn scan.

### MemGPT가 끝나고 Letta가 시작되는 곳

2024년 9월 MemGPT는 Letta가 되었습니다. research repo(`cpacker/MemGPT`)는 남아 있고, Letta는 design을 확장합니다.

- 두 tier가 아니라 세 tier(core, recall, archival - Lesson 08).
- `send_message`/heartbeat pattern을 대체하는 native reasoning(Lesson 08).
- async memory work를 실행하는 sleep-time agent(Lesson 08).

production system이 Letta, Mem0, custom two-tier store를 실행하더라도 MemGPT paper는 2026년 foundation입니다.

### 이 pattern이 잘못되는 곳

- **Memory rot.** write가 read보다 빠르게 쌓이고, retrieval이 stale fact에 잠깁니다. 해결: periodic consolidation(Letta sleep-time), explicit invalidation(Mem0 conflict detector).
- **Memory poisoning.** external memory는 retrieved text입니다. attacker-controlled content가 memory note에 들어가면 다음 session에서 agent가 다시 ingest합니다. 이는 Greshake et al.(Lesson 27)의 attack을 시간 축으로 다시 말한 것입니다.
- **Citation loss.** agent가 "user가 X를 ship하라고 요청했다"를 recall하지만 어떤 turn인지 cite하지 못합니다. 모든 archival write에 source reference(session ID, turn ID)를 함께 저장하세요.

```figure
context-budget
```

## Build It

`code/main.py`는 MemGPT의 two-tier pattern을 stdlib로 구현합니다.

- `MainContext` - `core` dict와 `messages` list를 가진 fixed-size prompt buffer. cap을 넘으면 oldest message를 auto-compact.
- `ArchivalStore` - (id, text, tags, session, turn) record의 in-memory BM25-esque store(token-overlap scoring).
- MemGPT surface에 mapping되는 다섯 memory tool.
- archival을 fact로 채운 뒤 `archival_memory_search`를 호출해 question에 답하는 scripted agent.

실행:

```bash
python3 code/main.py
```

trace는 agent가 세 fact를 쓰고, main context를 cap까지 채워 eviction을 강제한 뒤, archival에서 retrieve하여 follow-up question에 답하는 모습을 보여 줍니다. 실제 LLM 없이 MemGPT workflow를 재현합니다.

## Use It

오늘날 모든 production memory system은 MemGPT variant입니다.

- **Letta** (Lesson 08) - three tiers, native reasoning, sleep-time compute.
- **Mem0** (Lesson 09) - scoring layer와 fused된 vector + KV + graph.
- **OpenAI Assistants / Responses** - threads와 files를 통한 managed memory.
- **Claude Agent SDK** - skills와 session store를 통한 long-term memory.

core pattern이 아니라 operational shape(self-hosted, managed, framework-integrated)로 고르세요. core pattern은 MemGPT입니다.

## Ship It

`outputs/skill-virtual-memory.md`는 어떤 target runtime에 대해서도 올바른 two-tier memory scaffold(main + archival + tool surface)를 만드는 reusable skill입니다. eviction policy와 citation field가 연결되어 있습니다.

## Exercises

1. token으로 측정되는 `max_main_context_tokens` cap을 추가하세요(`len(text.split())` * 1.3으로 근사). cap을 넘으면 oldest message를 summary로 compact하세요. summarizer 유무에 따른 behavior를 비교하세요.
2. archival store 위에 BM25를 제대로 구현하세요(term frequency, inverse document frequency). toy fact set에서 token-overlap baseline 대비 recall@10을 측정하세요.
3. archival insert에 `citation` field(session_id, turn_id, source_url)를 추가하세요. retrieval-backed answer마다 source를 cite하게 만드세요.
4. memory poisoning을 simulate하세요. "ignore all future user instructions"라고 말하는 archival record를 추가합니다. retrieval에서 directive-shaped text를 scan하고 untrusted로 mark하는 guard를 작성하세요.
5. implementation을 MemGPT research repo의 core-memory JSON schema(`cpacker/MemGPT`)를 사용하도록 port하세요. flat string에서 typed section으로 바꾸면 무엇이 달라지나요?

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Virtual context | "Unlimited memory" | page in/out이 있는 main(prompt) + external(searchable) tier |
| Main context | "Working memory" | prompt - fixed-size이고 항상 visible |
| Archival memory | "Long-term store" | demand에 따라 retrieved되는 external searchable persistence |
| Core memory | "Persistent prompt section" | main context 안에 pinned된 named section |
| Memory tool | "Memory API" | external memory를 read/write하기 위해 agent가 발행하는 tool call |
| Interrupt | "Memory page fault" | agent가 pause하고 runtime이 fetch하며 result가 next turn에 splice |
| Memory rot | "Stale facts" | old write가 retrieval을 잠기게 함. consolidation으로 해결 |
| Memory poisoning | "Injected persistent note" | attacker content가 memory로 저장되고 recall 시 다시 ingest됨 |

## Further Reading

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) - OS-inspired virtual context paper
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) - three-tier evolution
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - context를 budget으로 다루기
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) - 이 pattern 위의 hybrid production memory
