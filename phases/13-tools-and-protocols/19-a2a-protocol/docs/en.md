# A2A — Agent-to-Agent Protocol

> MCP는 agent-to-tool입니다. A2A(Agent2Agent)는 agent-to-agent입니다. 서로 다른 framework 위에 만들어진 opaque agent들이 협업하게 해 주는 open protocol입니다. Google이 2025년 4월 공개했고, 2025년 6월 Linux Foundation에 기증했으며, 2026년 4월 v1.0에 도달했습니다. AWS, Cisco, Microsoft, Salesforce, SAP, ServiceNow를 포함한 150개 이상의 supporter가 있습니다. IBM의 ACP를 흡수했고 AP2 payments extension을 추가했습니다. 이 lesson은 Agent Card, Task lifecycle, 두 transport binding을 살펴봅니다.

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 minutes

## 학습 목표

- agent-to-tool(MCP)과 agent-to-agent(A2A) use case를 구분합니다.
- `/.well-known/agent.json`에 skill 및 endpoint metadata가 있는 Agent Card를 게시합니다.
- Task lifecycle(submitted → working → input-required → completed / failed / canceled / rejected)을 따라갑니다.
- Part(text, file, data)가 있는 Message와 output으로서의 Artifact를 사용합니다.

## 문제

고객 서비스 agent가 report-writing을 전문 writer agent에게 delegate해야 합니다. A2A 이전의 선택지는 다음과 같았습니다.

- Custom REST API. 동작은 하지만 모든 pairing이 one-off입니다.
- Shared codebase. 두 agent가 같은 framework에서 실행되어야 합니다.
- MCP. 맞지 않습니다. MCP는 tool을 호출하기 위한 것이지, 각 agent의 opaque internal reasoning을 보존하면서 두 agent가 협업하기 위한 것이 아닙니다.

A2A는 이 간극을 채웁니다. 한 agent가 다른 agent에게 Task를 보내는 interaction으로 모델링하며 lifecycle, message, artifact를 갖습니다. 호출된 agent의 internal state는 opaque로 유지됩니다. caller는 task state transition과 최종 output만 봅니다.

A2A는 "framework가 다른 agent들이 서로 대화하게 하는" protocol입니다. MCP를 대체하지 않습니다. 둘은 상호 보완적입니다.

## 개념

### Agent Card

모든 A2A-compliant agent는 `/.well-known/agent.json`에 card를 게시합니다.

```json
{
  "schemaVersion": "1.0",
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "url": "https://research.example.com/a2a",
  "version": "1.2.0",
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "inputModes": ["text", "file"],
      "outputModes": ["text", "artifact"]
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true}
}
```

Discovery는 URL 기반입니다. card를 fetch하고, A2A endpoint의 URL을 배우고, skill을 enumerate합니다.

### Signed Agent Cards (AP2)

AP2 extension(2025년 9월)은 Agent Card에 cryptographic signature를 추가합니다. publisher가 자기 card를 JWT로 서명하고 consumer가 검증합니다. impersonation을 방지합니다.

### Task lifecycle

```text
submitted -> working -> completed | failed | canceled | rejected
             -> input_required -> working (loop via message)
```

Client는 `tasks/send`로 시작합니다. 호출된 agent는 state를 전환합니다. client는 SSE로 state update를 subscribe하거나 poll합니다.

### Messages and Parts

Message는 하나 이상의 Part를 담습니다.

- `text` — plain content.
- `file` — mimeType이 있는 base64 blob.
- `data` — typed JSON payload(호출된 agent를 위한 structured input).

예:

```json
{
  "role": "user",
  "parts": [
    {"type": "text", "text": "Summarize this paper."},
    {"type": "file", "file": {"name": "paper.pdf", "mimeType": "application/pdf", "bytes": "..."}},
    {"type": "data", "data": {"targetLength": "3 paragraphs"}}
  ]
}
```

### Artifacts

Output은 raw string이 아니라 Artifact입니다. Artifact는 이름과 type이 있는 output입니다.

```json
{
  "name": "summary",
  "parts": [{"type": "text", "text": "..."}],
  "mimeType": "text/markdown"
}
```

Artifact는 chunk로 stream될 수 있습니다. caller가 누적합니다.

### Two transport bindings

1. **JSON-RPC over HTTP.** `/a2a` endpoint, request에는 POST, streaming에는 optional SSE. 기본 binding입니다.
2. **gRPC.** gRPC가 native인 enterprise environment용입니다.

두 binding은 같은 logical message shape를 전달합니다.

### Opacity preservation

핵심 design principle: 호출된 agent의 internal state는 opaque입니다. caller는 task state와 artifact만 봅니다. 호출된 agent의 chain-of-thought, tool call, sub-agent delegation은 모두 보이지 않습니다. tool call이 transparent한 MCP와 다릅니다.

이유: A2A는 경쟁자끼리도 internals를 공개하지 않고 협업할 수 있게 합니다. A2A는 caller가 service 구현 방식을 알지 않고도 "이 customer-service agent를 호출하라"가 될 수 있습니다.

### Timeline

- **2025-04-09.** Google이 A2A를 발표합니다.
- **2025-06-23.** Linux Foundation에 기증됩니다.
- **2025-08.** IBM의 ACP를 흡수합니다.
- **2025-09.** AP2 extension(Agent Payments)이 출시됩니다.
- **2026-04.** 150개 이상의 지원 조직과 함께 v1.0이 출시됩니다.

### Relationship to MCP

| 비교 기준 | MCP | A2A |
|-----------|-----|-----|
| Use case | Agent-to-tool | Agent-to-agent |
| Opacity | transparent tool call | opaque inner reasoning |
| Typical caller | Agent runtime | 다른 agent |
| State | Tool-call result | lifecycle이 있는 Task |
| Authorization | OAuth 2.1 (Phase 13 · 16) | JWT-signed Agent Cards (AP2) |
| Transport | Stdio / Streamable HTTP | JSON-RPC over HTTP / gRPC |

특정 tool을 호출하려면 MCP를 사용하세요. 전체 task를 다른 agent에게 delegate하려면 A2A를 사용하세요. 많은 production system은 둘 다 사용합니다. agent는 tool layer에 MCP를 쓰고 collaboration layer에 A2A를 씁니다.

## 사용하기

`code/main.py`는 minimal A2A harness를 구현합니다. research agent가 card를 게시하고, writer agent가 PDF와 text instruction을 포함한 part가 있는 `tasks/send`를 받아 working → input_required → working → completed를 거쳐 text artifact를 반환합니다. 전부 stdlib이며 message shape에 집중하도록 in-memory transport를 사용합니다.

볼 부분:

- Agent Card JSON shape.
- Task id assignment와 state transition.
- mixed-type part가 있는 message.
- task 중간의 input-required branch.
- completion 시 artifact return.

## 산출물

이 lesson은 `outputs/skill-a2a-agent-spec.md`를 만듭니다. 다른 agent가 호출할 수 있어야 하는 새 agent가 주어지면 이 skill은 Agent Card JSON, skills schema, endpoint blueprint를 만듭니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 호출된 agent가 clarification을 요청하는 input-required pause를 포함해 전체 Task lifecycle을 trace하세요.

2. signed Agent Card를 추가하세요. card의 canonical JSON에 HMAC으로 서명하세요. verifier를 작성하고 mutate된 card에서 실패하는지 확인하세요.

3. task streaming을 구현하세요. writer agent가 SSE로 세 개의 incremental artifact chunk를 emit하고 caller가 누적하게 하세요.

4. MCP 서버를 wrap하는 A2A agent를 설계하세요. 각 MCP tool을 A2A skill에 매핑하세요. 어떤 opacity가 사라지는지 trade-off를 기록하세요.

5. A2A v1.0 announcement를 읽고 2026년 4월 기준 어떤 framework도 아직 구현하지 않은 feature 하나를 식별하세요. (힌트: multi-hop task delegation과 관련 있습니다.)

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| A2A | "Agent-to-Agent protocol" | opaque agent collaboration을 위한 open protocol |
| Agent Card | "`.well-known/agent.json`" | agent의 skill과 endpoint를 설명하는 published metadata |
| Skill | "A callable unit" | agent가 지원하는 named operation(MCP tool의 analog) |
| Task | "Unit of delegation" | lifecycle과 final artifact가 있는 work item |
| Message | "Task input" | Part(text, file, data)를 운반함 |
| Part | "Typed chunk" | message의 `text` / `file` / `data` element |
| Artifact | "Task output" | completion 시 반환되는 named, typed output |
| AP2 | "Agent Payments Protocol" | trust와 payment를 위한 Signed Agent Cards extension |
| Opacity | "Black-box collaboration" | 호출된 agent의 internals가 caller에게 숨겨짐 |
| Input-required | "Task pause" | agent가 추가 정보를 필요로 할 때의 lifecycle state |

## 더 읽을거리

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — canonical A2A specification
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025년 6월 governance transfer
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — roadmap와 partner momentum
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 release notes 및 backward-compat guidance
