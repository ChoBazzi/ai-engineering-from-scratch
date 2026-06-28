---
name: mcp-client-harness
description: MCP server의 declarative list(name, command, args)가 주어지면 handshake, namespace merge, routing을 갖춘 multi-server client를 scaffold한다.
version: 1.0.0
phase: 13
lesson: 08
tags: [mcp, client, multi-server, routing, namespace]
---

실행할 MCP server configuration이 주어지면, 각 server를 spawn하고 handshake하며, tool list를 하나의 namespace로 merge하고, 각 call을 소유 server로 route하는 client harness를 작성한다.

작성할 것:

1. Server configuration parser. `name -> {command, args, env}`로 map한다. command가 path에 존재하는지 validate한다.
2. Spawn plan. stdin/stdout/stderr pipe, `bufsize=1`, text mode로 subprocess.Popen을 사용한다. server마다 background reader thread 하나를 둔다.
3. Handshake pipeline. 각 session에 대해 `initialize`를 보내고, response를 기다리고, capabilities를 저장하고, `notifications/initialized`를 보낸다.
4. Namespace merge. collision policy를 선택한다. `prefix-on-collision`(default), `reject-on-collision`, 또는 `silent-overwrite`(forbidden). startup 때 merged tool list를 출력한다.
5. Routing function. `client.call(canonical_name, arguments)`는 owning session을 찾고 `tools/call` message를 쓴다. pending-request table의 future를 통해 matching-id response를 기다린다.

강한 거부 조건:
- 각 server를 자체 process로 spawn하지 않는 harness. in-process multiplexing은 isolation model을 무너뜨린다.
- `silent-overwrite`를 default collision policy로 쓰는 harness. security risk다.
- stdout read에서 main thread를 block하는 harness. notification이 멈춘다.

거부 규칙:
- server command가 untrusted(pinned allowlist에 없음)라면 spawn을 거절하고 security check를 위해 Phase 13 · 15로 route한다.
- 사용자가 이유 없이 10개가 넘는 server를 configuration하면 경고하고 gateway(Phase 13 · 17)를 제안한다.
- 여기서 OAuth 처리를 요청하면 거절하고 Phase 13 · 16으로 route한다.

산출물: Session, merge logic, routing, 각 configured server를 exercise하는 main loop가 포함된 완전한 client-harness Python file(약 150줄). collision policy와 merged tool 수를 이름 붙이는 한 줄 summary로 끝낸다.
