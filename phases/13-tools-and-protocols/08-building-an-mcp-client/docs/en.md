# MCP Client 만들기 — Discovery, Invocation, Session 관리

> 대부분의 MCP content는 server tutorial을 제공하고 client 쪽은 대충 넘어간다. 어려운 orchestration은 client code에 있다. process spawning, capability negotiation, 여러 server의 tool list merge, sampling callback, reconnection, namespace collision resolution이 모두 여기에 속한다. 이 lesson은 세 개의 서로 다른 MCP server를 model이 사용할 하나의 flat tool namespace로 끌어올리는 multi-server client를 만든다.

**Type:** Build
**Languages:** Python (stdlib, multi-server MCP client)
**Prerequisites:** Phase 13 · 07 (building an MCP server)
**Time:** ~75 minutes

## 학습 목표

- MCP server를 child process로 spawn하고, `initialize`를 완료한 뒤 `notifications/initialized`를 보낸다.
- server별 session state(capabilities, tool list, 마지막으로 본 notification id)를 유지한다.
- 여러 server의 tool list를 collision handling과 함께 하나의 namespace로 merge한다.
- tool call을 해당 tool을 소유한 server로 route하고 response를 다시 조립한다.

## 문제

실제 agent host(Claude Desktop, Cursor, Goose, Gemini CLI)는 여러 MCP server를 한 번에 load한다. 사용자는 filesystem server, Postgres server, GitHub server를 동시에 실행하고 있을 수 있다. client의 일은 다음과 같다.

1. 각 server를 spawn한다.
2. 각 server와 독립적으로 handshake한다.
3. 각 server에 `tools/list`를 호출하고 결과를 flatten한다.
4. model이 `notes_search`를 emit하면 merged namespace에서 찾아 올바른 server로 route한다.
5. 어떤 server에서 온 notification(`tools/list_changed`)이든 blocking 없이 처리한다.
6. transport failure 시 reconnect한다.

이 모든 것을 직접 굴릴 수 있느냐가 "toy"와 "serviceable"을 가른다. 공식 SDK가 이것을 감싸 주지만, mental model은 여러분의 것이어야 한다.

## 개념

### Child-process spawning

`stdin=PIPE, stdout=PIPE, stderr=PIPE`로 `subprocess.Popen`을 사용한다. `bufsize=1`을 설정하고 line-by-line read를 위해 text mode를 사용한다. 각 server는 하나의 process이며, client는 server마다 하나의 `Popen` handle을 가진다.

### server별 session state

server마다 하나의 `Session` object가 다음을 가진다.

- `process` — Popen handle.
- `capabilities` — server가 `initialize`에서 선언한 것.
- `tools` — 마지막 `tools/list` 결과.
- `pending` — request id에서 response를 기다리는 promise/future로 가는 map.

request는 본질적으로 async다. server B가 call 중일 때 server A로 보낸 `tools/call`이 block되면 안 된다. queue를 쓰는 thread 또는 asyncio를 사용하라.

### Merged namespace

client가 aggregate tool list를 보면 name이 충돌할 수 있다. 두 server가 모두 `search`를 노출할 수 있다. client에는 세 가지 선택지가 있다.

1. **server name으로 prefix를 붙인다.** `notes/search`, `files/search`. 명확하지만 보기 좋지는 않다.
2. **조용한 first-come.** 나중 server의 `search`가 먼저 온 것을 override한다. 위험하며 collision을 숨긴다.
3. **Collision rejection.** 두 번째 server load를 거부하고 사용자에게 알린다. security-sensitive host에는 가장 안전하다.

Claude Desktop은 prefix-by-server를 사용한다. Cursor는 명확한 error와 함께 collision rejection을 사용한다. VS Code MCP도 prefix-by-server를 채택한다.

### Routing

merge 후 dispatch table은 `tool_name -> session`을 map한다. model은 name으로 call을 emit한다. client는 session을 찾고 해당 server의 stdin에 `tools/call` message를 쓴 다음 response를 기다린다.

### Sampling callback

server가 `initialize`에서 `sampling` capability를 선언했다면, client에게 LLM 실행을 요청하는 `sampling/createMessage`를 보낼 수 있다. client는 다음을 해야 한다.

1. sample이 resolve될 때까지 해당 server로 가는 추가 request를 block하거나, 구현이 concurrency를 지원한다면 pipeline한다.
2. 자신의 LLM provider를 호출한다.
3. response를 server로 다시 보낸다.

Lesson 11은 sampling을 end-to-end로 다룬다. 이 lesson에서는 완결성을 위해 stub만 둔다.

### Notification handling

`notifications/tools/list_changed`는 `tools/list`를 다시 호출하라는 뜻이다. `notifications/resources/updated`는 사용 중인 resource라면 다시 읽으라는 뜻이다. notification은 response를 만들면 안 된다. ack하려고 하지 말라.

흔한 client bug가 있다. stream 안에 notification이 있는데 `tools/call`에서 read loop를 block하는 것이다. 모든 message를 queue에 push하는 background reader thread를 사용하고, main thread가 dequeue해서 dispatch하게 하라.

### Reconnection

transport는 실패할 수 있다. server가 crash하거나, OS가 process를 kill하거나, stdio pipe가 끊어질 수 있다. client는 stdout의 EOF를 감지하고 session을 dead로 취급한다. 선택지는 다음과 같다.

- server를 조용히 restart하고 다시 handshake한다. 순수 read-only server에는 괜찮다.
- failure를 사용자에게 드러낸다. 사용자에게 보이는 session을 가진 stateful server에는 괜찮다.

Phase 13 · 09는 Streamable HTTP reconnection semantics를 다룬다. stdio는 더 단순하다.

### Keepalive와 session id

Streamable HTTP는 `Mcp-Session-Id` header를 사용한다. stdio에는 session id가 없다. process identity가 곧 session이다. keepalive ping은 선택 사항이다. stdio pipe는 비활성 상태라고 해서 끊어지지 않는다.

## 사용하기

`code/main.py`는 세 개의 simulated MCP server를 subprocess로 spawn하고, 각각 handshake하고, tool list를 merge하고, tool call을 올바른 server로 route한다. 여기서 "servers"는 실제로 toy responder를 실행하는 다른 Python process다(실제 LLM은 없다). 실행해서 다음을 확인하라.

- 각자 capability set을 가진 세 번의 initialization.
- 세 개의 `tools/list` 결과가 7-tool namespace로 merge되는 과정.
- tool name을 기반으로 한 routing decision.
- namespace prefixing으로 방지된 collision.

볼 것:

- `Session` dataclass가 server별 state를 깔끔하게 보관한다.
- background reader thread는 main thread를 block하지 않고 stdout의 모든 line을 dequeue한다.
- dispatch table은 단순한 `dict[str, Session]`이다.
- collision handling은 명시적이다. 두 server가 같은 name을 선언하면 나중 항목이 prefix가 붙은 이름으로 rename된다.

## 산출물

이 lesson은 `outputs/skill-mcp-client-harness.md`를 만든다. MCP server의 declarative list(name, command, args)가 주어지면, 이 skill은 server를 spawn하고 tool list를 merge하며 collision resolution이 포함된 routing function을 제공하는 harness를 만든다.

## 연습 문제

1. `code/main.py`를 실행하고 server spawn log를 관찰하라. simulated server process 하나를 SIGTERM으로 kill하고, client가 EOF를 감지해 해당 session을 dead로 표시하는 방식을 관찰하라.

2. namespace prefixing을 구현하라. 두 server가 `search`를 노출하면 두 번째 것을 `<server>/search`로 rename하라. dispatch table을 업데이트하고 tool call이 올바르게 route되는지 검증하라.

3. server restart를 위한 connection-pool-style backoff를 추가하라. 연속 failure에는 exponential backoff를 적용하고, 30초로 cap을 두며, 세 번 실패한 뒤 사용자에게 notification을 emit하라.

4. 100개의 concurrent MCP server를 지원하는 client를 스케치하라. 단순 dispatch dict를 어떤 data structure로 대체할까? 힌트: prefix namespacing을 위한 trie와 server별 tool-count metric.

5. client를 공식 MCP Python SDK로 port하라. SDK는 `stdio_client`와 `ClientSession`을 감싼다. multi-server routing을 유지하면서 code는 약 200줄에서 약 40줄로 줄어야 한다.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|-----------|
| MCP client | "agent host" | server를 spawn하고 tool call을 orchestrate하는 process |
| Session | "server별 state" | capabilities, tool list, pending-request bookkeeping |
| Merged namespace | "하나의 tool list" | 모든 active server에 걸친 flat tool name set |
| Namespace collision | "두 server가 같은 tool" | client가 duplicate를 prefix, reject, first-come 중 하나로 처리해야 한다 |
| Routing | "이 call은 누가 받는가?" | tool name에서 소유 server로 dispatch하는 것 |
| Background reader | "Non-blocking stdout" | server stdout을 queue로 비우는 thread 또는 task |
| Sampling callback | "LLM-as-a-service" | server에서 온 `sampling/createMessage`를 처리하는 client handler |
| `notifications/*_changed` | "Primitive mutated" | client가 다시 discover하거나 다시 read해야 한다는 신호 |
| Reconnection policy | "server가 죽었을 때" | transport가 실패했을 때의 restart semantics |
| Stdio session | "Process = session" | session id가 없고 child process lifetime이 session이다 |

## 더 읽을거리

- [Model Context Protocol — Client spec](https://modelcontextprotocol.io/specification/2025-11-25/client) — canonical client behavior
- [MCP — Quickstart client guide](https://modelcontextprotocol.io/quickstart/client) — Python SDK를 쓰는 hello-world client tutorial
- [MCP Python SDK — client module](https://github.com/modelcontextprotocol/python-sdk) — reference `ClientSession`와 `stdio_client`
- [MCP TypeScript SDK — Client](https://github.com/modelcontextprotocol/typescript-sdk) — TS parallel
- [VS Code — MCP in extensions](https://code.visualstudio.com/api/extension-guides/ai/mcp) — VS Code가 하나의 editor host 안에서 여러 MCP server를 multiplex하는 방식
