# MCP Server 만들기 — Python + TypeScript SDK

> 대부분의 MCP tutorial은 stdio hello-world만 보여 준다. 실제 server는 tools, resources, prompts를 노출하고, capability negotiation을 처리하며, structured error를 내보내고, SDK가 달라도 같은 방식으로 동작한다. 이 lesson은 notes server를 끝까지 만든다. stdlib stdio transport, JSON-RPC dispatch, 세 가지 server 프리미티브, 그리고 이후 Python SDK의 FastMCP나 TypeScript SDK로 옮겨도 그대로 넣을 수 있는 pure-function 스타일을 다룬다.

**Type:** Build
**Languages:** Python (stdlib, stdio MCP server)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals)
**Time:** ~75 minutes

## 학습 목표

- `initialize`, `tools/list`, `tools/call`, `resources/list`, `resources/read`, `prompts/list`, `prompts/get` method를 구현한다.
- stdin에서 JSON-RPC message를 읽고 stdout에 response를 쓰는 dispatch loop를 작성한다.
- JSON-RPC 2.0 사양과 MCP의 추가 code에 맞는 structured error response를 내보낸다.
- tool logic을 다시 쓰지 않고 stdlib 구현을 FastMCP(Python SDK)나 TypeScript SDK로 옮긴다.

## 문제

remote transport(Phase 13 · 09)나 auth layer(Phase 13 · 16)를 쓰기 전에 깔끔한 local server가 필요하다. local은 stdio를 뜻한다. server는 client가 child process로 spawn하고, message는 stdin/stdout 위에서 newline-delimited 방식으로 흐른다.

2025-11-25 사양은 stdio message를 명시적인 `\n` separator가 붙은 JSON object로 encode한다고 규정한다. 여기에는 SSE가 없다. SSE는 예전 remote mode였고 2026년 중반에 제거되는 중이다(Atlassian의 Rovo MCP server는 2026년 6월 30일에, Keboola는 2026년 4월 1일에 deprecated 처리했다). stdio에서는 줄마다 JSON object 하나가 wire format의 전부다.

notes server는 세 가지 server 프리미티브를 모두 연습하기 좋다. Tools는 mutation을 수행한다(`notes_create`). Resources는 data를 노출한다(`notes://{id}`). Prompts는 template을 제공한다(`review_note`). 이 lesson의 구조는 어떤 domain에도 일반화된다.

## 개념

### Dispatch loop

```text
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

세 가지 규칙:

- JSON-RPC envelope이 아닌 것은 stdout에 출력하지 않는다. debug log는 stderr로 보낸다.
- 모든 request는 같은 `id`를 가진 response와 반드시 짝지어져야 한다.
- notification에는 절대 response를 보내면 안 된다.

### `initialize` 구현하기

```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

지원하는 것만 선언하라. client는 feature를 gate하기 위해 capability set에 의존한다.

### `tools/list`와 `tools/call` 구현하기

`tools/list`는 각 entry에 `name`, `description`, `inputSchema`가 있는 `{tools: [...]}`를 반환한다. `tools/call`은 `{name, arguments}`를 받고 `{content: [blocks], isError: bool}`을 반환한다.

content block에는 type이 있다. 가장 흔한 예시는 다음과 같다.

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

tool error에는 두 가지 형태가 있다. Protocol-level error(unknown method, bad params)는 JSON-RPC error다. Tool-level error(유효한 call이지만 tool이 실패함)는 `{content: [...], isError: true}`로 반환한다. 이렇게 하면 model이 자신의 context 안에서 failure를 볼 수 있다.

### resources 구현하기

Resources는 설계상 read-only다. `resources/list`는 manifest를 반환하고, `resources/read`는 content를 반환한다. URI는 `file://...`, `http://...`, 또는 `notes://` 같은 custom scheme일 수 있다.

data를 tool이 아니라 resource로 노출하면 다음이 달라진다.

- model은 그것을 "call"하지 않는다. client가 사용자 요청에 따라 context에 inject할 수 있다.
- subscription을 통해 resource가 바뀔 때 server가 update를 push할 수 있다(Phase 13 · 10).
- Phase 13 · 14는 interactive resource를 위한 `ui://`로 이를 확장한다.

### prompts 구현하기

Prompts는 이름 있는 argument를 가진 template이다. host는 이를 slash-command로 노출한다. `review_note` prompt는 `note_id` argument를 받아 client가 model에 전달할 multi-message prompt template을 만들 수 있다.

### Stdio transport의 세부사항

- Newline-delimited JSON이다. length-prefixed framing이 아니다.
- buffer에 남겨 두지 않는다. write마다 `sys.stdout.flush()`를 호출한다.
- lifetime은 client가 제어한다. stdin이 닫히면(EOF) 깨끗하게 종료한다.
- SIGPIPE를 조용히 무시하지 않는다. log를 남기고 종료한다.

### Annotations

각 tool은 safety property를 설명하는 `annotations`를 가질 수 있다.

- `readOnlyHint: true` — 순수 read이며 retry해도 안전하다.
- `destructiveHint: true` — 되돌릴 수 없는 side effect가 있으므로 client가 확인해야 한다.
- `idempotentHint: true` — 같은 input은 같은 output을 만든다.
- `openWorldHint: true` — external system과 상호작용한다.

client는 이를 사용해 UX(confirmation dialog, status indicator)와 routing(Phase 13 · 17)을 결정한다.

### Graduation path

`code/main.py`의 stdlib server는 약 180줄이다. FastMCP(Python)는 같은 logic을 decorator 스타일로 줄인다.

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK도 같은 형태를 가진다. 준비가 되면 graduation path는 drop-in 방식이다. concepts(capabilities, dispatch, content blocks)는 그대로다.

## 사용하기

`code/main.py`는 stdlib만 사용하는 완전한 stdio 기반 notes MCP server다. 세 tool(`notes_list`, `notes_search`, `notes_create`)에 대한 `initialize`, `tools/list`, `tools/call`, 각 note에 대한 `resources/list`와 `resources/read`, 그리고 `review_note` prompt를 처리한다. JSON-RPC message를 pipe로 넣어 구동할 수 있다.

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

볼 것:

- dispatcher는 method name을 key로 삼는 `dict[str, Callable]`이다.
- 모든 tool executor는 bare string이 아니라 content block list를 반환한다.
- executor가 raise하면 `isError: true`가 설정된다.

## 산출물

이 lesson은 `outputs/skill-mcp-server-scaffolder.md`를 만든다. domain(notes, tickets, files, database)이 주어지면, 이 skill은 올바른 tools / resources / prompts 분리와 SDK graduation path를 갖춘 MCP server를 scaffold한다.

## 연습 문제

1. `code/main.py`를 실행하고 손으로 만든 JSON-RPC message로 구동하라. `notes_create`를 실행한 다음 `resources/read`로 새 note를 가져오라.

2. `annotations: {destructiveHint: true}`가 있는 `notes_delete` tool을 추가하라. client가 confirmation dialog를 표시할 수 있음을 확인하라. 이는 실제 host가 필요하며 Claude Desktop으로 가능하다.

3. note가 수정될 때마다 server가 `notifications/resources/updated`를 push하도록 `resources/subscribe`를 구현하라. keepalive task를 추가하라.

4. server를 FastMCP로 port하라. Python file은 80줄 미만으로 줄어야 한다. wire behavior는 동일해야 하며, 같은 JSON-RPC test harness로 검증하라.

5. 사양의 `server/tools` section을 읽고 이 lesson의 server에 구현되지 않은 tool definition field 하나를 찾아라. 힌트: 여러 개가 있으니 하나를 골라 추가하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|-----------|
| MCP server | "tool을 노출하는 것" | stdio 또는 HTTP 위에서 MCP JSON-RPC를 말하는 process |
| stdio transport | "child process model" | server가 client에 의해 spawn되고 stdin/stdout으로 통신한다 |
| Dispatcher | "Method router" | JSON-RPC method name에서 handler function으로 가는 map |
| Content block | "Tool result chunk" | tool response의 `content` array에 들어 있는 typed element |
| `isError` | "Tool-level failure" | tool이 실패했음을 알리며 JSON-RPC error와 구분한다 |
| Annotations | "Safety hints" | readOnly / destructive / idempotent / openWorld flag |
| FastMCP | "Python SDK" | MCP protocol 위의 decorator 기반 high-level framework |
| Resource URI | "주소 지정 가능한 data" | resource를 식별하는 `file://`, `db://`, 또는 custom scheme |
| Prompt template | "Slash-command brief" | host UI를 위한 argument slot이 있는 server-supplied template |
| Capability declaration | "Feature toggle" | `initialize`에서 선언되는 프리미티브별 flag |

## 더 읽을거리

- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — reference Python implementation
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — 병렬 TS implementation
- [FastMCP — server framework](https://gofastmcp.com/) — MCP server를 위한 decorator-style Python API
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — 두 SDK 중 하나를 사용하는 end-to-end tutorial
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — tools/* message의 complete reference
