---
name: agent-bundle
description: Claude Code, Cursor, Codex 및 호환 agent에서 load할 수 있는 workflow용 portable SKILL.md + AGENTS.md + MCP-server blueprint를 작성합니다.
version: 1.0.0
phase: 13
lesson: 21
tags: [skills, agents-md, apps-sdk, cross-agent, portability]
---

workflow 설명이 주어지면 agent bundle을 작성하세요.

다음을 산출하세요.

1. SKILL.md. `name`과 `description`이 있는 YAML frontmatter, 번호가 매겨진 단계가 있는 markdown body. body가 길면 progressive-disclosure subresource reference를 포함하세요.
2. AGENTS.md entry. skill이 의존하는 convention(linter command, test command)을 반영해 repo의 AGENTS.md에 추가할 몇 줄.
3. MCP server blueprint. skill이 MCP를 통해 호출하는 도구, name, description(Use-when pattern), input schema.
4. Cross-agent translations. 이 SKILL.md가 Cursor rules, Codex `.codex.md`, Windsurf rules로 어떻게 매핑되는지에 대한 SkillKit-style note.
5. Loading path. agent가 이 bundle을 발견할 위치: `~/.anthropic/skills/`, `./skills/`, `~/.claude/skills/`.

강한 거부 조건:
- `name`이 `kebab-case`가 아닌 SKILL.md. discovery가 깨집니다.
- frontmatter에 `description`이 없는 SKILL.md. agent runtime이 건너뜁니다.
- MCP 도구 이름이 Phase 13 · 05 규칙을 따르지 않는 bundle.

거부 규칙:
- workflow가 단일 one-shot prompt라면 skill 생성을 거절하고 inline prompt-engineering을 권장하세요.
- workflow가 OAuth(예: Slack post)를 요구하면 MCP 서버의 first-run elicitation이 이를 처리해야 한다고 표시하세요.
- 대상 agent가 SKILL.md를 지원하지 않는다면(일부 IDE) SkillKit 또는 유사 도구를 통한 translation을 권장하세요.

산출물: 세 파일의 sketch, cross-agent translation note, loading path를 담은 한 페이지 bundle. 마지막에는 이 bundle을 가장 먼저 테스트할 단일 agent를 적으세요.
