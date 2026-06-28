# Capstone — 완전한 Tool Ecosystem 만들기

> Phase 13은 모든 piece를 가르쳤습니다. 이 capstone은 그것들을 하나의 production-shaped system으로 연결합니다. tools + resources + prompts + tasks + UI가 있는 MCP server, edge의 OAuth 2.1, RBAC gateway, multi-server client, A2A sub-agent call, collector로 들어가는 OTel tracing, CI의 tool-poisoning detection, AGENTS.md + SKILL.md bundle입니다. 끝까지 마치면 모든 architecture choice를 방어할 수 있습니다.

**Type:** Build
**Languages:** Python (stdlib, end-to-end ecosystem harness)
**Prerequisites:** Phase 13 · 01 through 21
**Time:** ~120 minutes

## 학습 목표

- tools, resources, prompts, `ui://` app이 있는 task를 노출하는 MCP server를 compose합니다.
- OAuth 2.1 gateway를 앞단에 세워 RBAC와 pinned hash를 enforce합니다.
- OTel GenAI attribute로 end-to-end trace하는 multi-server client를 작성합니다.
- workload 일부를 A2A sub-agent에게 delegate하고 opacity가 보존되는지 검증합니다.
- 다른 agent가 drive할 수 있도록 전체 stack을 AGENTS.md + SKILL.md로 package합니다.

## 문제

"research and report" system을 ship합니다.

- User asks: "summarize the three most-cited 2026 arXiv papers on agent protocols."
- System: MCP로 arXiv를 search합니다. A2A로 specialized writer agent에게 paper summarization을 delegate합니다. result를 aggregate합니다. interactive report를 MCP Apps `ui://` resource로 render합니다. 모든 step을 OTel에 log합니다.

Phase 13의 모든 primitive가 등장합니다. 장난감이 아닙니다. Anthropic(Claude Research product), OpenAI(Apps SDK가 있는 GPTs), third party가 2026년에 출시한 production research-assistant system이 정확히 이런 형태입니다.

## 개념

### Architecture

```text
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### Trace hierarchy

```text
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

하나의 trace id입니다. 모든 span에는 올바른 `gen_ai.*` attribute가 있습니다.

### Security posture

- OAuth 2.1 + PKCE와 resource indicator로 audience를 gateway에 pin합니다.
- Gateway가 upstream credential을 보관하고 user는 이를 보지 못합니다.
- RBAC: `alice`는 `research:read`, `research:write`를 갖고 모든 tool을 호출할 수 있습니다. `bob`은 `research:read`만 있고 `generate_report`를 호출할 수 없습니다.
- Pinned description manifest: tool hash가 바뀐 server는 모두 drop됩니다.
- Rule of Two audit: 어떤 tool도 untrusted input, sensitive data, consequential action을 결합하지 않습니다.

### Rendering

최종 `generate_report` task는 content block과 `ui://report/current` resource를 반환합니다. client의 host(Claude Desktop 등)는 sandbox iframe에서 interactive dashboard를 render합니다. dashboard에는 정렬된 paper list, citation count, 사용자가 클릭한 paper에 대해 `host.callTool('summarize_paper', {arxiv_id})`를 호출하는 button이 있습니다.

### Packaging

전체는 다음으로 ship됩니다.

```text
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

사용자는 `docker compose up`으로 deploy합니다. Claude Code, Cursor, Codex, opencode 사용자는 `run-research` skill을 invoke해 system을 drive할 수 있습니다.

### What each Phase 13 lesson contributed

| 레슨 | Capstone에서 사용하는 내용 |
|--------|------------------------|
| 01-05 | Tool interface, provider portability, parallel call, schema, linting |
| 06-10 | MCP primitive, server, client, transport, resources + prompts |
| 11-14 | Sampling, roots + elicitation, async tasks, `ui://` apps |
| 15-17 | Tool poisoning, OAuth 2.1, gateway + registry |
| 18 | A2A sub-agent delegation |
| 19 | OTel GenAI tracing |
| 20 | LLM layer용 routing gateway |
| 21 | SKILL.md + AGENTS.md packaging |

## 사용하기

`code/main.py`는 이전 lesson의 pattern을 하나의 runnable demo로 엮습니다. 모두 stdlib이고 모두 in-process라서 end to end로 읽을 수 있습니다. research-and-report scenario의 전체 flow를 실행합니다. gateway handshake, simulated OAuth 2.1, merged tools/list, task로서의 generate_report, writer로의 A2A call, 반환된 ui:// resource, emit된 OTel span입니다.

볼 부분:

- 모든 hop에 걸친 하나의 trace id.
- gateway policy가 두 번째 user의 write를 block합니다.
- Task lifecycle이 working → completed로 진행되고 text와 ui:// content를 모두 반환합니다.
- A2A call의 inner state는 orchestrator에게 opaque입니다.
- AGENTS.md와 SKILL.md는 다른 agent가 workflow를 재현하는 데 필요한 유일한 파일입니다.

## 산출물

이 lesson은 `outputs/skill-ecosystem-blueprint.md`를 만듭니다. product need(research, summarization, automation)가 주어지면 이 skill은 full architecture를 만듭니다. 어떤 MCP primitive, 어떤 gateway control, 어떤 A2A call, 어떤 telemetry, 어떤 packaging인지 포함합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. single trace id와 span nesting을 확인하세요. demo가 Phase 13 primitive를 몇 개나 건드리는지 세세요.

2. demo를 확장하세요. 두 번째 backend MCP server(예: `bibliography`)를 추가하고 gateway가 같은 namespace로 tool을 merge하는지 확인하세요.

3. fake A2A writer agent를 subprocess에서 실행되는 실제 agent로 교체하세요. Lesson 19 harness를 사용하세요.

4. orchestrator와 LLM 사이의 routing gateway에 PII redaction step을 추가하세요. user query의 email이 scrub되는지 확인하세요.

5. 이 system을 유지할 teammate를 위한 AGENTS.md를 작성하세요. 읽는 데 5분 미만이어야 하며 Cursor 또는 Codex에서 capstone을 drive하는 데 필요한 모든 것을 제공해야 합니다.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Capstone | "Phase-13 integration demo" | 모든 primitive를 사용하는 end-to-end system |
| Research and report | "The scenario" | search, summarize, render pattern |
| Ecosystem | "All the pieces together" | server + client + gateway + sub-agent + telemetry + package |
| Trace hierarchy | "Single trace id" | 모든 hop의 span이 trace를 공유하고 span id로 parent-child 연결 |
| Gateway-issued token | "Transitive auth" | client는 gateway token만 보고 gateway가 upstream creds를 보관 |
| Merged namespace | "All tools in one flat list" | gateway에서 multi-server merge, collision 시 prefix |
| Opacity boundary | "A2A call hides internals" | sub-agent의 reasoning이 orchestrator에게 보이지 않음 |
| Three-layer stack | "AGENTS.md + SKILL.md + MCP" | project context + workflow + tools |
| Defense-in-depth | "Multiple security layers" | pinned hash, OAuth, RBAC, Rule of Two, audit log |
| Spec compliance matrix | "What we ship that the spec requires" | deliverable을 2025-11-25 requirement에 매핑한 checklist |

## 더 읽을거리

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — consolidated reference
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — protocol의 향후 방향
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 reference
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — canonical tracing conventions
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — production agent runtime pattern
