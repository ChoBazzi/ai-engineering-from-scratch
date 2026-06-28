---
name: mcp-transport-migrator
description: session id continuity와 Origin validation을 갖춘 legacy HTTP+SSE에서 Streamable HTTP로의 migration plan을 작성한다.
version: 1.0.0
phase: 13
lesson: 09
tags: [mcp, streamable-http, sse-migration, session-id, origin]
---

기존 HTTP+SSE(legacy) MCP server가 주어지면 single-endpoint Streamable HTTP로 가는 migration plan을 작성한다.

작성할 것:

1. Endpoint rewrite. `/messages`와 `/sse`를 하나의 `/mcp`로 merge한다. POST는 request handling, GET은 SSE stream, DELETE는 session termination으로 map한다.
2. Session continuity. 첫 POST에서 새 `Mcp-Session-Id`를 생성한다. client-supplied id는 거부한다. client가 legacy session cookie를 먼저 보내는 경우 bridging logic을 유지한다.
3. Origin validation. 명시적인 production origin(`https://app.company.com`, `https://claude.ai`, localhost variant)을 allowlist에 둔다. 나머지는 모두 403으로 거부한다.
4. Last-event-id replay. reconnect가 resume할 수 있도록 session별 recent event ring buffer를 유지한다.
5. Deprecation window. cut-over date와, legacy endpoint가 warning header와 함께 새 endpoint로 301하는 60일 grace period를 문서화한다.

강한 거부 조건:
- 두 endpoint를 무기한 유지하는 모든 plan. Legacy SSE는 2026년에 제거되고 있다.
- session id가 client-generated인 모든 plan. cryptographic-randomness requirement를 깬다.
- Origin validation이 없는 모든 plan. DNS-rebinding vulnerability다.

거부 규칙:
- server가 local-only(stdio)이면 HTTP migration을 거절한다. local에는 stdio가 맞다.
- server가 아직 OAuth를 제공하지 않으면 공개 노출 전에 Phase 13 · 16을 완료한다.
- hosting target이 long-lived HTTP를 지원하지 않으면(예: Vercel free tier) 거절하고 Cloudflare Workers를 권장한다.

산출물: endpoint change, Origin allowlist, session-id plan, deprecation schedule, 그리고 initialize, tools/list, streaming notifications, last-event-id를 사용한 reconnect, explicit DELETE를 포함하는 test checklist가 담긴 migration runbook.
