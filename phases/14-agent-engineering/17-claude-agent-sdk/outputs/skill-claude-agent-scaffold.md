---
name: claude-agent-scaffold
description: subagent, lifecycle hook, session store, MCP server attachment, W3C trace propagation을 갖춘 Claude Agent SDK 앱을 scaffold한다.
version: 1.0.0
phase: 14
lesson: 17
tags: [claude-agent-sdk, subagents, hooks, session-store, mcp]
---

product domain과 MCP server 목록이 주어지면 Claude Agent SDK 앱을 scaffold한다.

다음을 만든다.

1. instructions, 내장 도구 접근(read_file, write_file, shell, grep, glob, web fetch), custom function tool을 가진 main agent definition.
2. parallelization과 context isolation을 위한 subagent spawner. orchestrator가 아니면 context budget을 터뜨릴 때 사용한다.
3. 등록된 lifecycle hook: audit를 위한 PreToolUse + PostToolUse, setup을 위한 SessionStart, teardown을 위한 SessionEnd, rule enforcement를 위한 UserPromptSubmit(pro-workflow pattern 참조).
4. subagent tree를 렌더링하도록 `list_subkeys`가 연결된 session store(SQLite 기본).
5. 외부 tool/resource surface를 위한 MCP server attachment.
6. caller의 OTel span이 CLI까지 이어지도록 하는 W3C trace context propagation.

Hard reject:

- single-tool task에 subagent를 생성하는 것. subagent는 parallelization 또는 context isolation을 위한 것이지 "read_file 한 번 호출"을 위한 것이 아니다.
- synchronous expensive work가 있는 hook. hook은 microsecond에서 millisecond여야 한다. 긴 작업은 subagent에 둔다.
- cascade-delete policy가 없는 session store. orphaned subagent session은 storage를 부풀린다.

거부 규칙:

- product에 장시간 비동기 작업(hours-to-days)이 필요하면 self-hosted SDK를 거부하고 Claude Managed Agents로 route한다.
- 사용자가 shared location에 `--session-mirror`를 요구하면 거부한다. session transcript에는 PII가 있다. per-user encrypted storage로 mirror한다.
- agent가 tool use 없이 UX용 raw LLM streaming에 의존한다면 Agent SDK를 거부하고 Client SDK 직접 사용을 권장한다.

출력: `agent.py`, `tools.py`, `hooks.py`, `session.py`, 그리고 subagent policy, hook registry, session backend, MCP attachment, OTel wiring을 설명하는 `README.md`. 마지막에는 voice handoff를 위한 Lesson 22, OTel span attribution을 위한 Lesson 23, product에 production runtime shape가 필요하면 Lesson 18을 가리키는 "what to read next"로 끝낸다.
