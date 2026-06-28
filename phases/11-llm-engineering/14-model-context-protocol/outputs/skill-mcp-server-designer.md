---
name: mcp-server-designer
description: tools, resources, safety default를 갖춘 MCP server를 설계하고 scaffold합니다.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

도메인(internal API, database, file source)과 server를 mount할 host가 주어지면 다음을 출력하세요.

1. Primitive map. 어떤 capability가 `tools`(action), `resources`(읽기 전용 data), `prompts`(사용자 호출 template)가 되는지 정합니다. Primitive마다 한 줄씩 작성하세요.
2. Auth plan. Stdio(신뢰된 local), API key를 쓰는 streamable HTTP, 또는 PKCE를 쓰는 OAuth 2.1 중 하나를 선택하고 근거를 설명하세요.
3. Schema draft. 모든 tool parameter에 대한 JSON Schema를 작성하되, `description` field는 API docs가 아니라 model tool-selection에 맞춰 조정하세요.
4. Destructive-action list. State를 변경하는 모든 tool을 나열하고 `destructiveHint: true`와 human approval을 요구하세요.
5. Test plan. Tool마다 schema-only contract test 하나, MCP client를 통한 round-trip test 하나, red-team prompt-injection case 하나를 작성하세요.

Approval path 없이 disk에 쓰거나 external API를 호출하는 server는 출시를 거부하세요. 한 server에 tool을 20개 넘게 노출하는 것도 거부하고, 대신 domain-scoped server로 나누세요.
