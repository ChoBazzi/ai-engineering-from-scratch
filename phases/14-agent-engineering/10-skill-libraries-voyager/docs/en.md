# Skill Library와 Lifelong Learning (Voyager)

> Voyager(Wang et al., TMLR 2024)는 executable code를 skill로 다룹니다. Skill은 named, retrievable, composable하며 environment feedback으로 refined됩니다. 이것이 Claude Agent SDK skills, skillkit, 2026년 skill-library pattern의 reference architecture입니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 minutes

## Learning Objectives

- Voyager의 세 component(automatic curriculum, skill library, iterative prompting)와 각 역할을 말할 수 있습니다.
- Voyager가 action space를 primitive command가 아니라 code로 만드는 이유를 설명할 수 있습니다.
- Registration, retrieval, composition, failure-driven refinement를 지원하는 stdlib skill library를 구현할 수 있습니다.
- Voyager pattern을 2026년 Claude Agent SDK skills와 skillkit ecosystem에 대응시킬 수 있습니다.

## 문제

모든 session에서 capability를 처음부터 다시 만드는 agent는 세 가지를 잘못합니다.

1. **Token 낭비.** 모든 task가 같은 reasoning을 다시 elicitation합니다.
2. **Progress 손실.** Session A에서 배운 correction이 session B로 전달되지 않습니다.
3. **Long-horizon composition 실패.** 복잡한 task에는 capability hierarchy가 필요합니다. One-shot prompt는 이를 표현할 수 없습니다.

Voyager의 답은 재사용 가능한 capability 각각을 library에 저장된 named code chunk로 다루는 것입니다. 이 skill은 similarity로 retrieval되고, 다른 skill과 composition되며, execution feedback으로 refined됩니다.

## 개념

### 세 component

Voyager(arXiv:2305.16291)는 agent를 다음 세 요소 중심으로 구조화합니다.

1. **Automatic curriculum.** Curiosity-driven proposer가 agent의 현재 skill set과 environment state를 바탕으로 다음 task를 고릅니다. Exploration은 bottom-up입니다.
2. **Skill library.** 각 skill은 executable code입니다. 새 skill은 task가 성공했을 때 추가됩니다. Skill은 query-to-description similarity로 retrieval됩니다.
3. **Iterative prompting mechanism.** 실패하면 agent는 execution error, environment feedback, self-verification output을 받고 skill을 refine합니다.

Minecraft evaluation(Wang et al., 2024): baseline 대비 unique item 3.3배, stone tool 8.5배 빠름, iron tool 6.4배 빠름, map traversal 2.3배 길어짐. 수치는 Minecraft-specific이지만 pattern은 이전됩니다.

### Action space = code

대부분의 agent는 primitive command를 emit합니다. Voyager는 JavaScript function을 emit합니다. Skill은 다음과 같습니다.

```javascript
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Sub-skill에서 composition됩니다. Description과 embedding으로 keying되어 저장됩니다. Prompt가 아니라 program으로 retrieval됩니다.

이것이 2026년 Claude Agent SDK skill입니다. Agent가 필요할 때 load하는 named, retrievable code chunk와 instruction입니다.

### Skill retrieval

새 task: "diamond pickaxe 만들기." Agent는 다음을 수행합니다.

1. Task description을 embed합니다.
2. Skill library에서 top-k similar skill을 query합니다.
3. `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable` 등을 retrieval합니다.
4. Retrieved primitive + new logic으로 새 skill을 composition합니다.

이것이 MCP resource(Phase 13)와 Agent SDK skill이 구현하는 pattern입니다. 현재 task에 scoped된 knowledge/code surface 위의 retrieval입니다.

### Iterative refinement

Voyager의 feedback loop:

1. Agent가 skill을 씁니다.
2. Skill이 environment에서 실행됩니다.
3. 세 signal 중 하나가 반환됩니다. `success`, `error`(stack trace 포함), `self-verification failure`.
4. Agent는 signal을 context로 사용해 skill을 다시 씁니다.
5. Success 또는 max round까지 반복합니다.

이것은 code generation에 environment-grounded verification을 적용한 Self-Refine(Lesson 05)입니다. CRITIC(Lesson 05)은 external tool을 verifier로 쓰는 같은 pattern입니다.

### Curriculum과 exploration

Voyager의 curriculum module은 agent가 가진 것과 아직 하지 않은 것을 바탕으로 "호수 근처에 shelter 짓기" 같은 task를 제안합니다. Proposer는 environment state + skill inventory를 사용해 현재 capability보다 약간 높은 task를 고릅니다. 이것이 exploration sweet spot입니다.

Production agent에서는 "무엇이 빠졌는가" operator로 번역됩니다. 현재 skill library와 domain이 주어졌을 때 아직 cover하지 못하는 skill은 무엇인가요? 팀들은 보통 이를 curriculum review로 수동 구현합니다.

### 이 패턴이 잘못되는 지점

- **Skill library rot.** 거의 같은 description을 가진 동일 skill이 10번 추가됩니다. Write 시 deduplication을 추가하고 retrieval은 하나만 반환하게 하세요.
- **Composed-skill drift.** Parent skill이 refined된 child에 의존합니다. Skill에 version을 붙이세요. v1에 pin된 parent가 v3을 마법처럼 가져오면 안 됩니다.
- **Retrieval quality.** Skill description 위의 vector retrieval은 library가 수백 개를 넘으면 저하됩니다. Tag filter와 hard constraint("`category=tooling`인 skill만")를 보완하세요.

## 직접 만들기

`code/main.py`는 stdlib skill library를 구현합니다.

- `Skill` — name, description, code(string), version, tags, dependencies.
- `SkillLibrary` — register, search(token overlap), compose(dep의 topological sort), refine(update 시 version bump).
- 세 primitive skill을 register하고, 네 번째를 compose하며, failure를 만나 refine하는 scripted agent.

실행:

```bash
python3 code/main.py
```

Trace는 library write, retrieval, composition, failed execution, v2 refinement를 보여 줍니다. Voyager loop의 end-to-end입니다.

## 활용하기

- **Claude Agent SDK skills**(Anthropic) — 2026년 reference입니다. 각 skill에는 description, code, instruction이 있고 agent session 중 필요할 때 load됩니다.
- **skillkit**(npm: skillkit) — 32개 이상의 AI coding agent를 위한 cross-agent skill management입니다.
- **Custom skill libraries** — domain-specific합니다(SQL skills for data agents, Terraform skills for infra agents). Voyager pattern은 작은 규모에도 맞습니다.
- **OpenAI Agents SDK `tools`** — 낮은 수준에서는 각 tool이 lightweight skill입니다.

## 출시하기

`outputs/skill-skill-library.md`는 target runtime 어디에나 registration, retrieval, versioning, refinement가 연결된 Voyager 모양 skill library를 생성합니다.

## 연습 문제

1. `compose()`에 dependency-cycle detector를 추가하세요. Skill A가 B에 의존하고 B가 A에 의존하면 어떻게 되나요? Error인가요 warning인가요?
2. Skill별 version pinning을 구현하세요. Parent skill이 child `crafting@1`을 compose할 때, `crafting@2`로 refinement되어도 parent가 조용히 upgrade되면 안 됩니다.
3. Token-overlap retrieval을 sentence-transformers embedding(또는 BM25 stdlib impl)으로 바꾸세요. 50-skill toy library에서 retrieval@5를 측정하세요.
4. "Curriculum" agent를 추가하세요. 현재 library와 domain description이 주어졌을 때 빠진 skill 5개를 제안하게 하세요. 매주 호출하세요.
5. Anthropic의 Claude Agent SDK skill docs를 읽으세요. Toy library를 SDK의 skill schema로 port하세요. Discoverability에 무엇이 바뀌나요?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|--------------|-----------|
| Skill | "Reusable capability" | Similarity로 retrieval되는 named code chunk + description |
| Skill library | "Agent memory of how-to" | Searchable하고 composable한 persistent skill store |
| Curriculum | "Task proposer" | 현재 capability gap이 이끄는 bottom-up goal generator |
| Composition | "Skill DAG" | Skill이 skill을 호출하며 execution에서 topological sort됨 |
| Iterative refinement | "Self-correcting loop" | Env feedback + error + self-verification이 다음 version으로 되먹임됨 |
| Action-space-as-code | "Programmatic actions" | Temporally extended behavior를 위해 primitive command가 아니라 function을 emit함 |
| Dedup on write | "Skill collapse" | 거의 중복인 description을 하나의 canonical skill로 합침 |

## 더 읽기

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — original skill-library paper
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — 2026년 productization으로서의 skills
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — 실무에서의 skills와 subagents
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — Voyager 아래의 refinement loop
