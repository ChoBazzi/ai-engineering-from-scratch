# MCP Transports — stdio, Streamable HTTP, SSE 마이그레이션

> stdio는 local에서만 동작하고 그 밖에는 맞지 않는다. Streamable HTTP(2025-03-26)는 remote 표준이다. 오래된 HTTP+SSE transport는 deprecated되었고 2026년 중반에 제거되는 중이다. 잘못된 transport를 고르면 migration 비용을 치른다. 올바른 transport를 고르면 session continuity와 DNS-rebinding protection을 갖춘 remote-hostable MCP server를 얻는다.

**Type:** Learn
**Languages:** Python (stdlib, Streamable HTTP endpoint skeleton)
**Prerequisites:** Phase 13 · 07, 08 (MCP server and client)
**Time:** ~45 minutes

## 학습 목표

- deployment 형태(local vs remote, single-process vs fleet)에 따라 stdio와 Streamable HTTP 중 하나를 선택한다.
- Streamable HTTP single-endpoint pattern을 구현한다. request에는 POST, session stream에는 GET을 사용한다.
- DNS-rebinding을 막기 위해 `Origin` validation과 session-id semantics를 강제한다.
- 2026년 중반 제거 deadline 전에 legacy HTTP+SSE server를 Streamable HTTP로 migrate한다.

## 문제

첫 MCP remote transport(2024-11)는 HTTP+SSE였다. endpoint가 두 개였고, 하나는 client의 POST용, 하나는 server-to-client stream을 위한 Server-Sent-Events channel이었다. 동작은 했다. 하지만 어설펐다. session마다 endpoint가 두 개였고, 일부 CDN 앞에서는 cache가 깨졌으며, 일부 WAF가 공격적으로 종료하는 long-lived SSE connection에 강하게 의존했다.

2025-03-26 사양은 이를 Streamable HTTP로 대체했다. endpoint는 하나이고, client request에는 POST, session stream 수립에는 GET을 쓰며, 둘 다 `Mcp-Session-Id` header를 공유한다. 그 이후 새로 만들거나 migrate한 모든 server는 Streamable HTTP를 사용한다. 오래된 SSE mode는 deprecated되고 있다. Atlassian Rovo는 2026년 6월 30일에, Keboola는 2026년 4월 1일에 제거했고, 남은 대부분의 enterprise server도 2026년 말까지 제거할 예정이다.

그리고 stdio는 local server에 여전히 중요하다. Claude Desktop, VS Code, 그리고 모든 IDE 형태 client는 stdio로 server를 spawn한다. 올바른 mental model은 "이 machine"에는 stdio, "network 너머"에는 Streamable HTTP다. 서로 넘나들지 않는다.

## 개념

### stdio

- child-process transport다. client가 server를 spawn하고 stdin/stdout으로 통신한다.
- 줄마다 JSON object 하나다. newline-delimited 방식이다.
- session id가 없다. process identity가 session이다.
- auth가 필요 없다. child가 parent의 trust boundary를 상속한다.
- remote server에는 절대 사용하지 않는다. tunnel을 만들려면 SSH나 socat이 필요해지는데, 그 시점에는 Streamable HTTP를 써야 한다.

### Streamable HTTP

single endpoint `/mcp`(또는 임의 path) 하나를 사용한다. 세 HTTP method를 지원한다.

- **POST /mcp.** client가 JSON-RPC message를 보낸다. server는 단일 JSON response 또는 하나 이상의 response를 담은 SSE stream으로 응답한다. batched response와 해당 request 관련 notification에 유용하다.
- **GET /mcp.** client가 long-lived SSE channel을 연다. server는 이를 server-to-client request(sampling, notifications, elicitation)에 사용한다.
- **DELETE /mcp.** client가 명시적으로 session을 종료한다.

session은 server가 첫 response에 설정하고 client가 이후 모든 request에 echo하는 `Mcp-Session-Id` header로 식별된다. session id는 반드시 cryptographically random(128+ bits)이어야 한다. 안전을 위해 client가 고른 id는 거부한다.

### Single endpoint vs two

old spec의 two-endpoint mode는 2026년에도 아직 호출 가능하다. 사양은 이를 "legacy compatible"이라고 선언한다. 하지만 모든 새 server는 single-endpoint여야 한다. 공식 SDK는 single-endpoint를 emit한다. legacy mode는 migrate되지 않은 remote와 통신할 때만 사용하라.

### `Origin` validation과 DNS-rebinding

브라우저는 오늘날 MCP client가 아니지만, 공격자는 브라우저가 사용자의 local MCP server가 듣고 있는 `localhost:1234/mcp`로 POST하도록 유도하는 webpage를 만들 수 있다. server가 `Origin`을 확인하지 않으면 browser의 same-origin policy는 보호해 주지 못한다. `Origin: http://evil.com`은 유효한 cross-origin이기 때문이다.

2025-11-25 사양은 allowlist에 없는 `Origin`의 request를 server가 거부하도록 요구한다. allowlist에는 보통 MCP client host(`https://claude.ai`, `vscode-webview://*`)와 local UI를 위한 localhost variant가 들어간다.

### Session id lifecycle

1. client는 `Mcp-Session-Id` 없이 첫 request를 보낸다.
2. server는 random id를 할당하고 response header에 `Mcp-Session-Id`를 설정한다.
3. client는 이후 모든 request와 stream을 위한 `GET /mcp`에서 해당 header를 echo한다.
4. server는 session을 revoke할 수 있다. client는 이후 request에서 404를 보고 다시 initialize해야 한다.
5. client는 clean shutdown을 위해 session을 명시적으로 DELETE할 수 있다.

### Keepalive와 reconnect

SSE connection은 끊어진다. client는 같은 `Mcp-Session-Id`로 다시 GET해 재수립한다. server는 끊긴 동안 놓친 event를 합리적인 window까지 queue해야 하며, client가 echo하는 `last-event-id` header를 통해 replay해야 한다.

Phase 13 · 13은 full-session reconnect가 일어나도 long-running work가 살아남게 하는 Tasks를 다룬다.

### Backwards compatibility probe

old server와 new server를 모두 지원하려는 client는 다음을 수행한다.

1. `/mcp`로 POST한다.
2. response가 JSON 또는 SSE와 함께 `200 OK`이면 Streamable HTTP다.
3. response가 `Content-Type: text/event-stream`과 함께 `200 OK`이고 secondary endpoint를 가리키는 `Location` header가 있으면 legacy HTTP+SSE다. `Location`을 따라간다.

### Cloudflare, ngrok, hosting

2026년의 production remote MCP server는 Cloudflare Workers(MCP Agents SDK 포함), Vercel Functions, 또는 containerized Node/Python에서 실행된다. 핵심은 hosting이 SSE GET을 위한 long-lived HTTP connection을 지원해야 한다는 점이다. Vercel free tier는 10초 cap이 있어 부적합하다. Cloudflare Workers는 indefinite stream을 지원한다.

### Gateway composition

여러 MCP server 앞에 gateway를 둘 때(Phase 13 · 17), gateway는 session id를 rewrite하고 upstream을 multiplex하는 단일 Streamable HTTP endpoint다. tools는 gateway layer에서 merge된다. client는 단일 logical server를 본다.

### Transport failure modes

- **stdio SIGPIPE.** write 중 child process가 죽으면 SIGPIPE가 발생한다. server는 깨끗하게 종료해야 한다. client는 EOF를 감지하고 session을 dead로 표시해야 한다.
- **HTTP 502 / 504.** upstream failure 시 Cloudflare, nginx, 기타 proxy가 이를 emit한다. Streamable HTTP client는 짧은 backoff 뒤 한 번 retry해야 한다.
- **SSE connection drop.** TCP RST, proxy timeout, client network change가 stream을 닫는다. client는 `Mcp-Session-Id`와 선택적 `last-event-id`로 reconnect해 resume한다.
- **Session revocation.** server가 session id를 invalidate한다. client는 다음 request에서 404를 본다. client는 다시 handshake해야 한다.
- **Clock skew.** client의 Resource-TTL 계산이 server와 어긋난다. client는 server timestamp를 authoritative하게 취급해야 한다.

### Streamable HTTP를 우회해야 할 때

일부 enterprise는 자체 network 안에서 gRPC 또는 message-queue transport 뒤에 MCP server를 배치한다. 이는 non-standard다. MCP 사양은 이를 공식적으로 정의하지 않는다. gateway는 내부적으로 gRPC를 사용하면서 MCP client에는 Streamable HTTP surface를 노출할 수 있다. external surface는 spec-compliant하게 유지하고, translation은 gateway가 책임진다.

## 사용하기

`code/main.py`는 `http.server`(stdlib)를 사용해 최소 Streamable HTTP endpoint를 구현한다. `/mcp`에서 POST, GET, DELETE를 처리하고, 첫 response에 `Mcp-Session-Id`를 설정하며, `Origin`을 validate하고 allowlist에 없는 origin의 request를 거부한다. handler는 Lesson 07 notes server의 dispatch logic을 재사용한다.

볼 것:

- POST handler는 JSON-RPC body를 읽고 dispatch한 뒤 JSON response를 쓴다. 여기서는 single-response variant이며, SSE variant도 구조적으로 비슷하다.
- `Origin` check는 기본 `http://evil.example` probe를 거부하지만 `http://localhost`는 허용한다.
- session id는 random 128-bit hex string이다. server는 session별 state를 memory에 보관한다.

## 산출물

이 lesson은 `outputs/skill-mcp-transport-migrator.md`를 만든다. HTTP+SSE(legacy) MCP server가 주어지면, 이 skill은 session-id continuity, Origin check, backwards-compatible probe support를 갖춘 Streamable HTTP migration plan을 만든다.

## 연습 문제

1. `code/main.py`를 실행하라. `curl`에서 `initialize`를 POST하고 `Mcp-Session-Id` response header를 관찰하라. 해당 header를 echo하는 두 번째 request를 POST하고 session continuity를 검증하라.

2. SSE stream을 여는 GET handler를 추가하라. 5초마다 `notifications/progress` event 하나를 보내라. 같은 session id로 다시 GET해 reconnect하고 server가 이를 허용하는지 확인하라.

3. `last-event-id` replay logic을 구현하라. reconnect 시 해당 id 이후 생성된 모든 event를 replay하라.

4. `Origin` validation을 확장해 wildcard pattern(`https://*.example.com`)을 지원하고, `https://app.example.com`은 허용하지만 `https://evil.example.com.attacker.net`은 거부함을 확인하라.

5. official registry의 legacy HTTP+SSE server 하나를 가져와 migration을 스케치하라. endpoint handling, session id generation, header semantics에서 무엇이 바뀌는지 설명하라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|-----------|
| stdio transport | "Local child process" | stdin/stdout 위의 newline-delimited JSON-RPC |
| Streamable HTTP | "remote transport" | single-endpoint POST + GET + optional SSE, 2025-03-26 spec |
| HTTP+SSE | "Legacy" | 2026년 중반 제거되는 two-endpoint model |
| `Mcp-Session-Id` | "Session header" | server가 할당하고 이후 모든 request에 echo되는 random id |
| `Origin` allowlist | "DNS-rebinding defense" | 승인되지 않은 Origin의 request를 거부한다 |
| Single endpoint | "One URL" | `/mcp`가 모든 session operation의 POST / GET / DELETE를 처리한다 |
| `last-event-id` | "SSE replay" | event 누락 없이 끊긴 stream을 resume하는 데 쓰는 header |
| Backwards-compat probe | "old vs new detection" | transport를 자동 선택하는 client response-shape check |
| Long-lived HTTP | "SSE streaming" | server가 하나의 TCP connection에서 몇 분 또는 몇 시간 동안 event를 push한다 |
| Session revocation | "Force re-init" | server가 session id를 invalidate하고 client는 다시 handshake해야 한다 |

## 더 읽을거리

- [MCP — Basic transports spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports) — stdio와 Streamable HTTP에 대한 canonical reference
- [MCP — Basic transports spec 2025-03-26](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports) — Streamable HTTP를 도입한 revision
- [Cloudflare — MCP transport](https://developers.cloudflare.com/agents/model-context-protocol/transport/) — Workers-hosted Streamable HTTP pattern
- [AWS — MCP transport mechanisms](https://builder.aws.com/content/35A0IphCeLvYzly9Sw40G1dVNzc/mcp-transport-mechanisms-stdio-vs-streamable-http) — deployment shape별 비교
- [Atlassian — HTTP+SSE deprecation notice](https://community.atlassian.com/forums/Atlassian-Remote-MCP-Server/HTTP-SSE-Deprecation-Notice/ba-p/3205484) — 구체적인 migration deadline 예시
