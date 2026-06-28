# Memory Blocks와 Sleep-Time Compute (Letta)

> MemGPT는 2024년에 Letta가 되었습니다. 2026년의 진화는 두 가지 아이디어를 더합니다. 모델이 직접 편집할 수 있는 이산적인 기능형 memory block과, 기본 agent가 유휴 상태일 때 비동기로 memory를 통합하는 sleep-time agent입니다. 이것이 하나의 대화를 넘어 memory를 확장하는 방식입니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 minutes

## Learning Objectives

- Letta가 사용하는 세 가지 memory tier(core, recall, archival)의 이름과 각각의 역할을 말할 수 있습니다.
- memory-block 패턴을 설명할 수 있습니다. Human block, Persona block, 사용자 정의 block은 일급 typed object입니다.
- sleep-time compute가 무엇인지, 왜 critical path 밖에 있어야 하는지, 왜 기본 agent보다 강한 모델을 실행할 수 있는지 설명할 수 있습니다.
- primary agent가 응답을 제공하고 sleep-time agent가 turn 사이에 block을 통합하는 scripted two-agent loop를 구현할 수 있습니다.

## 문제

MemGPT(Lesson 07)는 virtual-memory control flow를 해결했습니다. 하지만 프로덕션에서는 세 가지 문제가 드러났습니다.

1. **Latency.** 모든 memory operation이 critical path 위에 있습니다. 사용자가 기다리는 동안 agent가 가지치기, 요약, 조정을 해야 하면 tail latency가 커집니다.
2. **Memory rot.** write가 누적됩니다. 반박된 fact도 남아 있습니다. retrieval은 오래된 content에 묻힙니다.
3. **Structure loss.** 평평한 archival store는 "Human block은 항상 prompt 안에 있다. Persona block도 항상 prompt 안에 있다. Task block은 session마다 교체된다" 같은 구조를 표현할 수 없습니다.

Letta(letta.com)는 2026년의 재작성입니다. Memory block은 구조를 명시적으로 만들고, sleep-time compute는 통합 작업을 critical path 밖으로 옮깁니다.

## 개념

### 세 가지 tier

| Tier | 범위 | 위치 | 작성 주체 |
|------|------|------|-----------|
| Core | 항상 visible | main prompt 내부 | Agent tool call + sleep-time rewrite |
| Recall | Conversation history | Retrievable | Automatic turn logging |
| Archival | 임의의 fact | Vector + KV + graph | Agent tool call + sleep-time ingest |

Core는 MemGPT의 core입니다. Recall은 축출된 tail을 포함하는 conversation buffer입니다. Archival은 external store입니다. 이 분리는 MemGPT의 two-tier overloading을 정리합니다.

### Memory block

Block은 core tier에 속하는 typed, persistent, editable section입니다. 원래 MemGPT 논문은 두 가지를 정의했습니다.

- **Human block** — 사용자에 대한 fact(name, role, preference, goal).
- **Persona block** — agent의 self-concept(identity, tone, constraint).

Letta는 이를 임의의 사용자 정의 block으로 일반화합니다. 현재 목표를 위한 `Task` block, codebase fact를 위한 `Project` block, hard constraint를 위한 `Safety` block을 둘 수 있습니다. 각 block에는 `id`, `label`, `value`, `limit`(문자 수 상한), `description`(모델이 언제 편집해야 하는지 알게 해 주는 설명)이 있습니다.

Block은 tool surface를 통해 편집할 수 있습니다.

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — 한도에 가까운 block을 압축합니다.

### Sleep-time compute

2025년 Letta 추가 사항입니다. critical path 밖의 background에서 두 번째 agent를 실행합니다. Sleep-time agent는 conversation transcript와 codebase context를 처리하고, `learned_context`를 shared block에 쓰며, archival record를 통합하거나 무효화합니다.

그 결과 다음 속성이 생깁니다.

- **Latency cost 없음.** Primary response는 memory op를 기다리지 않습니다.
- **더 강한 모델 허용.** Sleep-time agent는 latency 제약이 없기 때문에 더 비싸고 느린 모델일 수 있습니다.
- **자연스러운 consolidation window.** 사용자가 기다리지 않는 동안 dedup, summary, 반박된 fact 무효화를 수행합니다.

이 모양은 인간이 일하는 방식과 닮았습니다. 작업을 하고, 잠을 자며, 장기 기억이 밤새 정리됩니다.

### Letta V1과 native reasoning

Letta V1(`letta_v1_agent`, 2026)은 native reasoning을 위해 `send_message`/heartbeat와 inline `Thought:` token을 폐기합니다. Responses API(OpenAI)와 extended thinking을 켠 Messages API(Anthropic)는 별도 channel로 reasoning을 내보내고, turn을 지나 전달됩니다(프로덕션에서는 provider 사이에서 암호화). Control loop는 여전히 ReAct입니다. Thought trace는 prompt 모양이 아니라 구조적입니다.

### 이 패턴이 잘못되는 지점

- **Block bloat.** 무한 `block_append`는 빠르게 limit에 닿습니다. cap을 넘기는 write 전에 block summarizer를 연결하세요.
- **Silent drift.** Sleep-time agent가 block을 다시 썼는데 primary agent가 알아차리지 못합니다. Block에 version을 붙이고 trace에 diff를 노출하세요.
- **Poisoned consolidation.** Sleep-time agent가 공격자가 도달할 수 있는 content를 core로 처리합니다. Lesson 27은 sleep-time surface에도 적용됩니다.

## 직접 만들기

`code/main.py`는 다음을 구현합니다.

- `Block` — id, label, value, limit, description.
- `BlockStore` — CRUD + `near_limit(label)` helper.
- 두 scripted agent — `PrimaryAgent`는 turn을 처리하고, `SleepTimeAgent`는 turn 사이에 통합합니다.
- Block write가 있는 세 turn 대화와, block을 요약하고 stale fact를 무효화하는 sleep-time pass를 보여 주는 trace.

실행:

```bash
python3 code/main.py
```

Transcript는 분리를 보여 줍니다. Primary turn은 빠르고 raw write를 생성하며, sleep pass가 압축하고 정리합니다.

## 활용하기

- **Letta**(letta.com): reference implementation입니다. Self-host 또는 managed cloud로 사용합니다.
- **Claude Agent SDK skills**: block 모양의 knowledge입니다. Skill은 agent가 필요할 때 load하는 named, versioned, retrievable instruction block입니다.
- **Custom builds**: storage backend를 직접 제어하려는 팀에 적합합니다. 나중에 migration할 수 있도록 Letta API contract를 사용하세요.

## 출시하기

`outputs/skill-memory-blocks.md`는 safety rule과 citation wiring을 포함해, 어떤 runtime에도 쓸 수 있는 sleep-time hook이 달린 Letta 모양 block system을 생성합니다.

## 연습 문제

1. `near_limit`이 true를 반환할 때 block value를 model-generated summary로 교체하는 `block_summarize` tool을 추가하세요. Summary call과 block overflow를 모두 최소화하는 trigger threshold는 무엇인가요?
2. Archival에 대해 sleep-time dedup을 구현하세요. Text의 token overlap이 90%를 넘는 두 record는 하나로 합칩니다. Critical path에서는 절대 하지 말고 sleep pass에서만 수행하세요.
3. Block에 version을 붙이세요. 모든 write마다 old value와 diff를 기록하세요. Operator가 "왜 agent가 X를 잊었는가"를 debug할 수 있도록 `block_history(label)`을 노출하세요.
4. Sleep-time agent를 untrusted writer로 취급하세요. Persona 또는 Safety block을 건드리면 commit 전에 second-agent review를 요구하세요.
5. 예제를 Letta API(`letta_v1_agent`)를 쓰도록 port하세요. Block schema에서 무엇이 바뀌며, native reasoning은 trace shape를 어떻게 바꾸나요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| Memory block | "Editable prompt section" | Core memory의 typed, persistent, LLM-editable segment |
| Human block | "User memory" | Core에 고정된 사용자 관련 fact |
| Persona block | "Agent identity" | Core에 고정된 self-concept, tone, constraint |
| Sleep-time compute | "Async memory work" | Critical path 밖에서 두 번째 agent가 수행하는 consolidation |
| Core / Recall / Archival | "Tiers" | 항상 visible / conversation / external로 나뉜 three-layer memory split |
| Block limit | "Cap" | Block별 문자 제한이며 summarization을 강제함 |
| Native reasoning | "Thinking channel" | Prompt-level `Thought:`가 아닌 provider-level reasoning output |
| Learned context | "Sleep output" | Sleep-time agent가 shared block에 쓰는 fact |

## 더 읽기

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — block pattern
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — async consolidation
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — native reasoning rewrite
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — origin
