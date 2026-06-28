# Claude Agent SDK: Subagent와 Session Store

> Claude Agent SDK는 Claude Code harness의 라이브러리 형태다. 내장 도구, 컨텍스트 격리를 위한 subagent, hook, W3C trace propagation, session store parity를 제공한다. Claude Managed Agents는 장시간 비동기 작업을 위한 hosted 대안이다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 minutes

## 학습 목표

- Anthropic Client SDK(raw API)와 Claude Agent SDK(harness 형태)의 차이를 설명한다.
- subagent가 하는 일, 즉 parallelization과 context isolation을 설명하고 언제 쓸지 판단한다.
- Python SDK의 session store 표면(`append`, `load`, `list_sessions`, `delete`, `list_subkeys`)과 `--session-mirror`의 역할을 말한다.
- 내장 도구, 격리된 컨텍스트의 subagent 생성, lifecycle hook, session store를 갖춘 stdlib harness를 구현한다.

## 문제

raw LLM API는 왕복 호출 한 번을 제공한다. 프로덕션 에이전트에는 도구 실행, MCP 서버, lifecycle hook, subagent 생성, session persistence, trace propagation이 필요하다. Claude Agent SDK는 Claude Code가 쓰는 것과 같은 harness 형태를 라이브러리로 제공하여 custom agent에서 쓸 수 있게 한다.

## 개념

### Client SDK vs Agent SDK

- **Client SDK(`anthropic`).** Raw Messages API. loop, 도구, 상태를 직접 소유한다.
- **Agent SDK(`claude-agent-sdk`).** 내장 도구 실행, MCP 연결, hook, subagent 생성, session store. Claude Code loop를 라이브러리로 제공한다.

### 내장 도구

SDK는 파일 읽기/쓰기, shell, grep, glob, web fetch 등 10개 이상의 도구를 기본 제공한다. custom tool은 표준 tool-schema interface로 등록한다.

### Subagent

Anthropic 문서가 설명하는 두 가지 목적:

1. **Parallelization.** 독립적인 작업을 동시에 실행한다. "이 20개 모듈 각각에 대한 테스트 파일을 찾아라"는 20개의 병렬 subagent task다.
2. **Context isolation.** Subagent는 자기 context window를 사용한다. orchestrator에는 결과만 돌아온다. orchestrator의 예산이 보존된다.

Python SDK의 최근 추가 기능: subagent transcript를 읽기 위한 `list_subagents()`, `get_subagent_messages()`.

### Session store

TypeScript와 protocol parity를 가진다.

- `append(session_id, message)` - 턴을 추가한다.
- `load(session_id)` - 대화를 복원한다.
- `list_sessions()` - 열거한다.
- `delete(session_id)` - subagent session까지 cascade 삭제한다.
- `list_subkeys(session_id)` - subagent key를 나열한다.

`--session-mirror`(CLI flag)는 디버깅을 위해 transcript를 스트리밍하면서 외부 파일에 mirror한다.

### Hook

등록할 수 있는 lifecycle hook:

- `PreToolUse`, `PostToolUse` - 도구 호출을 gate하거나 audit한다.
- `SessionStart`, `SessionEnd` - setup과 teardown을 수행한다.
- `UserPromptSubmit` - 모델이 보기 전에 사용자 입력에 대해 동작한다.
- `PreCompact` - context compaction 전에 실행된다.
- `Stop` - agent 종료 시 cleanup한다.
- `Notification` - side-channel alert.

hook은 pro-workflow(Phase 14 curriculum reference)와 유사한 시스템이 cross-cutting behavior를 추가하는 방식이다.

### W3C trace context

호출자에서 활성화된 OTel span은 W3C trace context header를 통해 CLI subprocess로 전파된다. 전체 multi-process trace가 백엔드에서 하나의 trace로 보인다.

### Claude Managed Agents

Hosted 대안(beta header `managed-agents-2026-04-01`). 장시간 비동기 작업, 내장 prompt caching, 내장 compaction을 제공한다. 제어권을 managed infrastructure와 맞바꾼다.

### 이 패턴이 잘못되는 지점

- **Subagent over-spawn.** 100개의 작은 task에 100개의 subagent를 생성한다. overhead가 지배한다. 대신 batch하라.
- **Hook creep.** 모든 팀이 hook을 추가하고 startup time이 부풀어 오른다. 분기마다 hook을 리뷰하라.
- **Session bloat.** session이 누적되고 크기가 커진다. `list_sessions`와 expiry policy를 사용하라.

## 직접 만들기

`code/main.py`는 SDK 형태를 stdlib로 구현한다.

- `read_file`, `write_file`, `list_dir` 내장 도구를 가진 `Tool`, `ToolRegistry`.
- `Subagent` - private context, 격리된 실행, 반환되는 결과.
- `SessionStore` - append, load, list, delete, list_subkeys.
- `Hooks` - `pre_tool_use`, `post_tool_use`, `session_start`, `session_end`.
- demo: main agent가 3개의 subagent를 병렬로 생성하고(각각 격리됨), 결과를 집계하고, session을 영속화한다.

실행:

```bash
python3 code/main.py
```

trace는 subagent context isolation(orchestrator context size가 bounded로 유지됨), hook 실행, session persistence를 보여 준다.

## 활용하기

- **Claude Agent SDK**는 Claude Code harness 형태를 원하는 Claude 우선 제품에 사용한다.
- **Claude Managed Agents**는 hosted 장시간 비동기 작업에 사용한다.
- **OpenAI Agents SDK**(Lesson 16)는 OpenAI 우선 counterpart에 사용한다.
- **LangGraph + custom tools**는 graph-shaped state machine을 원할 때 사용한다.

## 출시하기

`outputs/skill-claude-agent-scaffold.md`는 subagent, hook, session store, MCP server attachment, W3C trace propagation을 갖춘 Claude Agent SDK 앱을 scaffold한다.

## 연습

1. 20개의 task를 5개 병렬 subagent 그룹으로 batch하는 subagent spawner를 추가하라. task마다 하나를 생성하는 방식과 orchestrator context size를 비교하라.
2. `write_file` 호출을 rate limit하는 `PreToolUse` hook을 구현하라(session마다 분당 5회). 동작을 trace하라.
3. `list_subkeys`를 연결해 subagent tree를 렌더링하라. 깊은 중첩은 어떤 모습인가?
4. 장난감을 실제 `claude-agent-sdk` Python package로 이식하라. 도구 등록에서 무엇이 바뀌는가?
5. Claude Managed Agents 문서를 읽어라. 언제 self-hosted에서 managed로 전환하겠는가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agent SDK | "Claude Code as a library" | 도구, MCP, hook, subagent, session store를 갖춘 harness 형태 |
| Subagent | "Child agent" | 별도 컨텍스트와 자체 예산을 가진다. 결과가 위로 올라온다 |
| Session store | "Conversation DB" | subagent cascade와 함께 턴을 persist, load, list, delete한다 |
| Hook | "Lifecycle callback" | pre/post tool, session, prompt submit, compact, stop |
| W3C trace context | "Cross-process trace" | parent span이 CLI subprocess로 전파된다 |
| Managed Agents | "Hosted harness" | Anthropic-hosted 장시간 비동기 작업 |
| `--session-mirror` | "Transcript mirror" | session turn을 스트리밍하면서 외부 파일에 쓴다 |
| MCP server | "Tool surface" | agent에 붙는 외부 도구/리소스 소스 |

## 더 읽을거리

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude Code의 라이브러리 형태
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) - production pattern
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) - hosted 대안
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) - counterpart
