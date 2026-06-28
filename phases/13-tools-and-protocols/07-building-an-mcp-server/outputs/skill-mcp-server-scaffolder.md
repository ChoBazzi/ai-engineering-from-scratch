---
name: mcp-server-scaffolder
description: 올바른 tools/resources/prompts 분리와 SDK graduation path를 갖춘 domain-specific MCP server를 scaffold한다.
version: 1.0.0
phase: 13
lesson: 07
tags: [mcp, server, fastmcp, scaffold]
---

domain(notes, tickets, files, database 등)이 주어지면 MCP server plan을 작성한다. 어떤 capabilities를 tools로, 어떤 것을 resources로, 어떤 것을 prompts로 노출할지와 Python 또는 TypeScript SDK로 가는 graduation path를 포함한다.

작성할 것:

1. Tools list. 사용자가 명시적으로 수행하기를 요청하는 atomic operation. name, description(Use-when pattern), input schema, annotation hint를 포함한다.
2. Resources list. 사용자가 읽고 싶어 하는 data. URI scheme, mime type, `resources/subscribe` 활성화 여부를 포함한다.
3. Prompts list. host가 slash-command로 노출해야 하는 reusable template. argument list를 포함한다.
4. Capability declaration. server가 `initialize`에서 반환하는 정확한 `capabilities` object.
5. Graduation notes. 각 부분에 대한 FastMCP(Python) 또는 TypeScript SDK equivalent. scaffold의 hand-rolled stdlib pattern을 대체하는 SDK feature 하나(예: `lifespan`, `context`)를 이름 붙인다.

강한 거부 조건:
- "database query"를 resource가 아니라 tool로만 노출하는 설계. 올바른 분리는 `/list`와 `/read`는 resource, parameter가 있는 `/query`는 tool이다.
- user-input tool과 privileged tool을 annotation 없이 같은 namespace에 섞는 server.
- durable notification mechanism 없이 `resources/subscribe` capability를 주장하는 server scaffold.

거부 규칙:
- domain에 read-only surface가 없으면 resources scaffold를 거절하고 tool-only server를 권장한다.
- domain에 자연스러운 slash-command template이 없으면 prompts scaffold를 거절한다.
- 사용자가 auth scheme을 요청하면 거절하고 Phase 13 · 16(OAuth 2.1)으로 route한다.

산출물: 세 primitive list, capability object, 10줄짜리 `@app.tool()` decorator-style graduation snippet을 담은 한 페이지 server plan. server가 설정해야 할 가장 중요한 annotation flag 하나로 끝낸다.
