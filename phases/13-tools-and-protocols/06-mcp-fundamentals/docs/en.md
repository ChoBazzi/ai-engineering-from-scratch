# MCP 기초 — 프리미티브, 생명주기, JSON-RPC 기반

> MCP 이전의 모든 통합은 일회성 연결이었다. 2024년 11월 Anthropic이 처음 공개했고 지금은 Linux Foundation의 Agentic AI Foundation이 관리하는 Model Context Protocol은 discovery와 invocation을 표준화해 어떤 client든 어떤 server와도 통신할 수 있게 한다. 2025-11-25 사양은 여섯 가지 프리미티브(server 세 가지, client 세 가지), 세 단계 생명주기, JSON-RPC 2.0 wire format을 정의한다. 이것들을 익히면 이 phase의 나머지 MCP 장은 사양을 읽어 나가는 일이 된다.

**Type:** Learn
**Languages:** Python (stdlib, JSON-RPC parser)
**Prerequisites:** Phase 13 · 01 through 05 (the tool interface and function calling)
**Time:** ~45 minutes

## 학습 목표

- 여섯 가지 MCP 프리미티브(server의 tools, resources, prompts; client의 roots, sampling, elicitation)를 모두 말하고 각각의 사용 사례를 하나씩 제시한다.
- 세 단계 생명주기(initialize, operation, shutdown)를 따라가며 각 단계에서 누가 어떤 메시지를 보내는지 설명한다.
- JSON-RPC 2.0 request, response, notification envelope을 파싱하고 내보낸다.
- `initialize`에서 이루어지는 capability negotiation이 무엇이며, 그것이 없으면 무엇이 깨지는지 설명한다.

## 문제

MCP 이전에는 tool을 사용하는 agent마다 자체 프로토콜이 있었다. Cursor에는 MCP처럼 생겼지만 호환되지 않는 tool 시스템이 있었다. Claude Desktop은 다른 시스템을 제공했다. VS Code의 Copilot 확장에는 또 다른 시스템이 있었다. "Postgres query" tool을 만든 팀은 같은 tool을 세 번 작성해야 했고, 각각 다른 host API에 맞춰야 했다. 재사용하려면 코드를 복사해야 했다.

그 결과 일회성 통합이 캄브리아기 폭발처럼 늘어났고, 생태계가 성장하는 속도에는 천장이 생겼다.

MCP는 wire format을 표준화해 이 문제를 해결한다. 하나의 MCP server는 Claude Desktop, ChatGPT, Cursor, VS Code, Gemini, Goose, Zed, Windsurf 등 모든 MCP client에서 동작한다. 2026년 4월 기준 client는 300개 이상, SDK 월간 다운로드는 1억 1천만 건, 공개 server는 10,000개 이상이다. Linux Foundation은 2025년 12월 새 Agentic AI Foundation 아래에서 관리 주체가 되었다.

이 phase에서 사용하는 사양 개정판은 **2025-11-25**다. 이 버전은 async Tasks(SEP-1686), URL-mode elicitation(SEP-1036), sampling with tools(SEP-1577), incremental scope consent(SEP-835), OAuth 2.1 resource-indicator semantics를 추가한다. Phase 13 · 09부터 16까지가 이 확장들을 다룬다. 이 lesson은 기반까지만 다룬다.

## 개념

### 세 가지 server 프리미티브

1. **Tools.** 호출 가능한 action이다. Phase 13 · 01의 네 단계 loop와 같다.
2. **Resources.** 노출된 data다. URI로 주소를 지정할 수 있는 read-only content이며 `file:///path`, `db://query/...`, custom scheme을 쓴다.
3. **Prompts.** 재사용 가능한 template이다. host UI의 slash-command로 나타나며, server가 template을 제공하고 client가 argument를 채운다.

### 세 가지 client 프리미티브

4. **Roots.** server가 접근할 수 있는 URI 집합이다. client가 선언하고 server가 이를 따른다.
5. **Sampling.** server가 client의 model에게 completion 수행을 요청한다. server-side API key 없이 server-hosted agent loop를 가능하게 한다.
6. **Elicitation.** server가 진행 중에 client의 사용자에게 structured input을 요청한다. form 또는 URL(SEP-1036)을 사용한다.

MCP의 모든 capability는 정확히 이 여섯 가지 중 하나에 속한다. Phase 13 · 10부터 14까지가 각각을 깊게 다룬다.

### Wire format: JSON-RPC 2.0

모든 메시지는 다음 field를 가진 JSON object다.

- Requests: `{jsonrpc: "2.0", id, method, params}`.
- Responses: `{jsonrpc: "2.0", id, result | error}`.
- Notifications: `{jsonrpc: "2.0", method, params}` — `id`가 없고 response를 기대하지 않는다.

기본 사양에는 프리미티브별로 묶인 method가 약 15개 있다. 중요한 것들은 다음과 같다.

- `initialize` / `initialized` (handshake)
- `tools/list`, `tools/call`
- `resources/list`, `resources/read`, `resources/subscribe`
- `prompts/list`, `prompts/get`
- `sampling/createMessage` (server-to-client)
- `notifications/tools/list_changed`, `notifications/resources/updated`, `notifications/progress`

### 세 단계 생명주기

**Phase 1: initialize.**

client는 자신의 `capabilities`와 `clientInfo`를 담아 `initialize`를 보낸다. server는 자신의 `capabilities`, `serverInfo`, 그리고 자신이 사용하는 사양 버전으로 응답한다. client는 response를 처리한 뒤 `notifications/initialized`를 보낸다. 이 시점부터 양쪽은 협상된 capability에 따라 request를 보낼 수 있다.

**Phase 2: operation.**

양방향이다. client는 discovery를 위해 `tools/list`를 호출한 다음 invocation을 위해 `tools/call`을 호출한다. server는 해당 capability를 선언했다면 `sampling/createMessage`를 보낼 수 있다. server는 tool set이 변경되면 `notifications/tools/list_changed`를 보낼 수 있다. client는 사용자가 root scope를 바꾸면 `notifications/roots/list_changed`를 보낼 수 있다.

**Phase 3: shutdown.**

어느 쪽이든 transport를 닫는다. MCP에는 구조화된 shutdown method가 없다. transport(stdio 또는 Streamable HTTP, Phase 13 · 09)가 연결 종료 신호를 전달한다.

### Capability negotiation

`initialize` handshake의 `capabilities`가 계약이다. server 예시는 다음과 같다.

```json
{
  "tools": {"listChanged": true},
  "resources": {"subscribe": true, "listChanged": true},
  "prompts": {"listChanged": true}
}
```

server는 `tools/list_changed` notification을 보낼 수 있고 `resources/subscribe`를 지원한다고 선언한다. client는 자신의 capability를 선언해 이에 동의한다.

```json
{
  "roots": {"listChanged": true},
  "sampling": {},
  "elicitation": {}
}
```

client가 `sampling`을 선언하지 않으면 server는 `sampling/createMessage`를 호출하면 안 된다. 반대도 같다. server가 `resources.subscribe`를 선언하지 않으면 client는 subscribe를 시도하면 안 된다.

이 규칙이 생태계의 표류를 막는다. sampling을 지원하지 않는 client도 여전히 유효한 MCP client다. `sampling`을 호출하지 않는 server도 여전히 유효한 MCP server다. 둘은 단지 그 feature를 함께 사용하지 않을 뿐이다.

### Structured content와 error shape

`tools/call`은 typed block으로 이루어진 `content` array를 반환한다. block type은 `text`, `image`, `resource`다. Phase 13 · 14는 여기에 MCP Apps(`ui://` interactive UI)를 추가한다.

error는 JSON-RPC error code를 사용한다. 사양에서 추가로 정의한 것은 `-32002` "Resource not found", `-32603` "Internal error", 그리고 `error.data`에 담는 MCP-specific error data다.

### Client capabilities와 tool call 세부사항

흔한 혼동이 있다. `capabilities.tools`는 client가 tool-list-changed notification을 지원하는지 여부다. client가 특정 tool을 실제로 호출할지는 capability flag가 아니라 model이 runtime에 내리는 선택이다. capability flag는 사양 수준의 계약이다. model의 선택은 별개의 문제다.

### 왜 REST가 아니라 JSON-RPC인가?

JSON-RPC 2.0(2010)은 가벼운 양방향 프로토콜이다. REST는 client-initiated 방식이다. MCP에는 server-initiated message(sampling, notifications)가 필요했으므로 대칭적인 request/response 형태를 가진 JSON-RPC가 자연스럽게 맞았다. JSON-RPC는 HTTP request 형태를 다시 발명하지 않고도 stdio와 WebSocket/Streamable HTTP 위에서 깔끔하게 조합된다.

```figure
mcp-tool-call
```

## 사용하기

`code/main.py`는 최소 JSON-RPC 2.0 parser와 emitter를 제공한 뒤, `initialize` → `tools/list` → `tools/call` → `shutdown` sequence를 손으로 따라가며 모든 메시지를 출력한다. 실제 transport는 없고 message shape만 있다. 각 envelope이 맞는지 확인하려면 Further Reading의 사양과 비교하라.

볼 것:

- `initialize`는 양쪽 capability를 선언한다. response에는 `serverInfo`와 `protocolVersion: "2025-11-25"`가 있다.
- `tools/list`는 `tools` array를 반환한다. 각 entry에는 `name`, `description`, `inputSchema`가 있다.
- `tools/call`은 `params.name`과 `params.arguments`를 사용한다.
- response의 `content`는 `{type, text}` block의 array다.

## 산출물

이 lesson은 `outputs/skill-mcp-handshake-tracer.md`를 만든다. MCP client-server 상호작용의 pcap 스타일 transcript가 주어지면, 이 skill은 각 메시지가 어떤 프리미티브, 어떤 생명주기 단계, 어떤 capability에 의존하는지 annotation한다.

## 연습 문제

1. `code/main.py`를 실행하라. capability negotiation이 일어나는 줄을 찾고, server가 `tools.listChanged`를 선언하지 않았다면 무엇이 달라지는지 설명하라.

2. parser를 확장해 `notifications/progress`를 처리하라. message shape은 `{method: "notifications/progress", params: {progressToken, progress, total}}`다. 오래 실행되는 `tools/call`이 진행 중일 때 이를 emit하고, client handler가 progress bar를 표시할 수 있음을 확인하라.

3. MCP 2025-11-25 사양을 처음부터 끝까지 읽어라. 문서 전체는 약 80쪽이다. 대부분의 server가 필요로 하지 않는 capability flag 하나를 찾아라. 힌트: resource subscription과 관련 있다.

4. 가상의 "cron job" feature가 어떤 프리미티브에 속할지 종이에 스케치하라. 힌트: server는 client가 정해진 시간에 그것을 invoke하기를 원한다. 현재 여섯 프리미티브 중 어느 것도 딱 맞지 않는다. MCP의 2026 roadmap에는 이에 대한 draft SEP가 있다.

5. GitHub의 공개 MCP server에서 session log 하나를 파싱하라. request, response, notification 메시지 수를 세어라. traffic 중 lifecycle과 operation의 비율을 계산하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|-----------|
| MCP | "Model Context Protocol" | model-to-tool discovery와 invocation을 위한 open protocol |
| Server primitive | "server가 노출하는 것" | tools(action), resources(data), prompts(template) |
| Client primitive | "client가 server에게 사용하게 해 주는 것" | roots(scope), sampling(LLM callback), elicitation(user input) |
| JSON-RPC 2.0 | "wire format" | 대칭적인 request/response/notification envelope |
| `initialize` handshake | "Capability negotiation" | 첫 message pair이며, server와 client가 자신이 지원하는 feature를 선언한다 |
| `tools/list` | "Discovery" | client가 server에게 현재 tool set을 요청한다 |
| `tools/call` | "Invocation" | client가 server에게 argument와 함께 tool 실행을 요청한다 |
| `notifications/*_changed` | "Mutation events" | server가 자신의 primitive list가 바뀌었음을 client에게 알린다 |
| Content block | "Typed result" | tool result 안의 `{type: "text" \| "image" \| "resource" \| "ui_resource"}` |
| SEP | "Spec Evolution Proposal" | 이름이 붙은 draft proposal(예: async Tasks를 위한 SEP-1686) |

## 더 읽을거리

- [Model Context Protocol — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — canonical spec document
- [Model Context Protocol — Architecture concepts](https://modelcontextprotocol.io/docs/concepts/architecture) — 여섯 프리미티브 mental model
- [Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) — 2024년 11월 launch post
- [MCP blog — First MCP anniversary](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 1년 회고와 2025-11-25 사양 변경 사항
- [WorkOS — MCP 2025-11-25 spec update](https://workos.com/blog/mcp-2025-11-25-spec-update) — SEP-1686, 1036, 1577, 835, 1724 요약
