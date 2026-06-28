---
name: mcp-handshake-tracer
description: MCP client-server 대화의 pcap 스타일 transcript가 주어지면, 모든 message에 primitive, lifecycle phase, capability dependency를 annotation한다.
version: 1.0.0
phase: 13
lesson: 06
tags: [mcp, json-rpc, lifecycle, capabilities]
---

MCP session에서 capture한 JSON-RPC 2.0 envelope sequence가 주어지면, 각 message의 primitive, lifecycle phase, underlying capability flag를 이름 붙이는 walk-through를 작성한다.

작성할 것:

1. message별 annotation. 각 `{request, response, notification}`에 대해 direction(client-to-server 또는 server-to-client), primitive(tools / resources / prompts / roots / sampling / elicitation / lifecycle), lifecycle phase, 그리고 이 message가 유효하려면 협상되어야 했던 capability flag를 명시한다.
2. Capability check. transcript에서 `initialize` exchange를 재구성하고 협상된 모든 capabilities를 나열한다. 존재하지 않는 capability를 위반하는 message를 표시한다.
3. Error diagnostics. 모든 JSON-RPC error에 대해 code와 주변 context상 가장 가능성 높은 원인을 이름 붙인다.
4. Completeness audit. `initialize`, `initialized` notification, 최소 하나의 `tools/list` 또는 equivalent, graceful shutdown 중 하나가 빠진 transcript를 표시한다.
5. Spec compliance. 각 request의 params를 2025-11-25 사양의 minimum field set과 대조한다. 누락을 표시한다.

강한 거부 조건:
- 사양의 allowed set 밖 method를 `x-` prefix 없이 사용하는 모든 message.
- client가 `sampling` capability를 선언하지 않았는데 등장하는 모든 `sampling/createMessage` message.
- `notifications/initialized`가 도착하기 전의 모든 invocation.

거부 규칙:
- non-MCP protocol의 transcript를 audit해 달라는 요청이면 거절하고 대안으로 A2A spec(Phase 13 · 19)을 가리킨다.
- transcript를 "fix"해 달라는 요청이면 거절한다. 이 skill은 annotation하며 rewrite하지 않는다. 수정은 implementing SDK를 통해 route한다.

산출물: arrival order에 따라 message마다 annotation line 하나를 작성한다. 형식은 `[phase/primitive/capability] <method or result shape>`이다. capability violation과 빠진 lifecycle step을 이름 붙이는 세 줄 summary로 끝낸다.
