# Skills와 Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP는 "어떤 tool이 있는가"를 말합니다. Skill은 "task를 어떻게 수행하는가"를 말합니다. 2026년 stack은 둘을 함께 layer합니다. Anthropic의 Agent Skills(open standard, 2025년 12월)는 progressive disclosure가 있는 SKILL.md로 제공됩니다. OpenAI의 Apps SDK는 MCP plus widget metadata입니다. AGENTS.md(이제 60,000개 이상 repo에 있음)는 repo root에서 project-level agent context로 작동합니다. 이 lesson은 각 layer가 무엇을 담당하는지 설명하고, agent 사이를 이동할 수 있는 minimal SKILL.md + AGENTS.md bundle을 만듭니다.

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 학습 목표

- 세 layer를 구분합니다. AGENTS.md(project context), SKILL.md(reusable know-how), MCP(tools).
- YAML frontmatter와 progressive disclosure가 있는 SKILL.md를 작성합니다.
- filesystem-style로 skill을 agent runtime에 load합니다.
- skill을 MCP server 및 AGENTS.md와 compose해 하나의 package가 Claude Code, Cursor, Codex에서 작동하게 합니다.

## 문제

한 engineer가 release-notes-writing workflow를 multi-step prompt로 정리합니다. "최근 merge된 PR을 읽어라. area별로 group하라. 각각을 summarize하라. 팀 style을 따라 changelog entry를 작성하라. Slack draft로 post하라." 그들은 이것을 팀의 Notion doc에 넣습니다.

이제 이 workflow를 Claude Code, Cursor, Codex CLI에서 쓰고 싶습니다. 각 agent는 instruction을 load하는 방식이 다릅니다. Claude Code slash-command, Cursor rules, Codex `.codex.md`. engineer는 workflow를 세 번 복사하고 세 copy를 유지합니다.

AGENTS.md와 SKILL.md가 함께 이 문제를 해결합니다.

- **AGENTS.md**는 repo root에 있습니다. compatible agent는 session start 때 이를 읽습니다. "이 project는 어떻게 동작하는가? convention은 무엇인가? test command는 무엇인가?"
- **SKILL.md**는 portable bundle입니다. YAML frontmatter(name, description) + markdown body + optional resource. skill을 지원하는 agent는 필요할 때 name으로 load합니다.
- **MCP**(Phase 13 · 06-14)는 skill이 invoke해야 하는 tool을 처리합니다.

세 layer, 하나의 portable artifact입니다.

## 개념

### AGENTS.md (agents.md)

2025년 말 출시되어 2026년 4월까지 60,000개 이상 repo가 채택했습니다. repo root의 파일 하나입니다. 형식:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

Agent는 session start 때 이를 읽고 해당 project에 맞춰 behavior를 보정합니다. 2026년의 모든 coding agent가 AGENTS.md를 지원합니다. Claude Code, Cursor, Codex, Copilot Workspace, opencode, Windsurf, Zed.

### SKILL.md format

Anthropic의 Agent Skills(2025년 12월 open standard로 공개):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## 참고

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

Frontmatter는 skill의 identity를 선언합니다. body는 skill이 load될 때 model에 보여주는 prompt입니다.

### Progressive disclosure

Skill은 agent가 필요할 때만 fetch하는 sub-resource를 reference할 수 있습니다. 예:

```text
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md가 "style rule은 style-guide.md를 보라"고 말합니다. agent는 skill이 실제로 실행될 때만 style-guide.md를 가져옵니다. 이렇게 하면 model이 필요 없을 수도 있는 detail로 prompt를 부풀리지 않습니다.

### Filesystem discovery

Agent runtime은 알려진 directory에서 SKILL.md 파일을 scan합니다.

- `~/.anthropic/skills/*/SKILL.md`
- Project `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

Loading은 folder name과 frontmatter `name`으로 합니다. Claude Code, Anthropic Claude Agent SDK, SkillKit(cross-agent)이 모두 이 pattern을 따릅니다.

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`(TypeScript)와 `claude-agent-sdk`(Python)는 session start 때 skill을 load하고 runtime 안에서 callable "agents"로 노출합니다. agent loop는 사용자가 skill을 invoke하면 그 skill로 dispatch합니다.

### OpenAI Apps SDK

2025년 10월 출시되었고 MCP 위에 직접 build되었습니다. OpenAI의 이전 Connectors와 Custom GPT Actions를 단일 developer surface로 통합합니다. Apps SDK app은 다음입니다.

- MCP server(tools, resources, prompts).
- ChatGPT UI를 위한 widget metadata.
- interactive surface를 위한 optional MCP Apps `ui://` resource.

같은 protocol, 더 풍부한 UX입니다.

### Cross-agent portability via SkillKit

SkillKit 같은 도구와 유사한 cross-agent distribution layer는 하나의 SKILL.md를 32개 이상의 AI agent(Claude Code, Cursor, Codex, Gemini CLI, OpenCode 등)의 native format으로 translate합니다. 하나의 source of truth, 여러 consumer입니다.

### The three-layer stack

| 계층 | 파일 | 로드 시점 | 목적 |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | project-level convention |
| SKILL.md | skills directory | skill invoked | reusable workflow |
| MCP server | external process | tools needed | callable actions |

셋은 모두 compose됩니다. agent가 session start 때 AGENTS.md를 읽고, 사용자가 skill을 invoke하고, skill instruction은 MCP tool call을 포함하며, agent는 MCP client를 통해 dispatch합니다.

## 사용하기

`code/main.py`는 stdlib SKILL.md parser와 loader를 제공합니다. `./skills/` 아래 skill을 discover하고, YAML frontmatter와 markdown body를 parse해 skill name을 key로 하는 dict를 생성합니다. 그런 다음 agent loop를 simulate해 `release-notes-writer`를 name으로 invoke합니다.

볼 부분:

- minimal stdlib parser로 YAML frontmatter parse(no `pyyaml` dependency).
- Skill body는 verbatim으로 저장되고, agent는 invocation 시 system prompt 앞에 붙입니다.
- `read_subresource` function이 referenced file을 필요할 때 가져오며 progressive disclosure를 demo합니다.

## 산출물

이 lesson은 `outputs/skill-agent-bundle.md`를 만듭니다. workflow가 주어지면 이 skill은 agent 사이에서 portable한 SKILL.md + AGENTS.md + MCP-server-blueprint bundle을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. `skills/` 아래 두 번째 skill을 추가하고 loader가 이를 감지하는지 확인하세요.

2. 이 course repo의 AGENTS.md를 작성하세요. testing command, style convention, Phase 13 mental model을 포함하세요.

3. 팀 내부 문서의 multi-step workflow를 SKILL.md로 port하세요. Claude Code에서 load되는지 확인하세요.

4. skill을 Cursor와 Codex의 native rule format으로 손수 translate하세요. format 간 diff를 세어 보세요. 이것이 SkillKit이 automate하는 translation surface입니다.

5. Anthropic Agent Skills blog post를 읽으세요. 이 lesson의 loader가 다루지 않는 Claude Agent SDK feature 하나를 식별하세요. (힌트: agent sub-invocation.)

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| SKILL.md | "The skill file" | agent runtime이 load하는 YAML frontmatter plus markdown body |
| AGENTS.md | "Repo-root agent context" | session start 때 읽는 project-level convention file |
| Progressive disclosure | "Lazy-load sub-resources" | skill body가 필요할 때만 가져오는 file을 reference |
| Frontmatter | "YAML block at top" | `---` delimiter 안의 metadata(name, description) |
| Claude Agent SDK | "Anthropic's skill runtime" | skill을 load하고 route하는 `@anthropic-ai/claude-agent-sdk` |
| OpenAI Apps SDK | "MCP + widget meta" | MCP와 ChatGPT UI hook 위에 세운 OpenAI developer surface |
| Skill discovery | "Filesystem scan" | 알려진 dir에서 SKILL.md를 walk하고 name으로 keying |
| Cross-agent portability | "One skill many agents" | SkillKit-style tool로 하나의 SKILL.md를 32개 이상 agent에 translate |
| Agent Skill | "Portable know-how" | MCP의 tool concept 바깥에 있는 reusable task template |
| Apps SDK | "MCP plus ChatGPT UI" | Connector와 Custom GPT가 MCP 위에 통합됨 |

## 더 읽을거리

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025년 12월 launch
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md format reference
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — ChatGPT용 MCP-based developer platform
- [agents.md](https://agents.md/) — AGENTS.md format과 adoption list
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — official skill examples
