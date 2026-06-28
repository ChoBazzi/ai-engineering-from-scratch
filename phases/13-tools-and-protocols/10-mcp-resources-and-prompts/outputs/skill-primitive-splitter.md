---
name: primitive-splitter
description: MCP server draft의 각 capability를 rationale과 함께 tool, resource, prompt 중 하나로 분류한다.
version: 1.0.0
phase: 13
lesson: 10
tags: [mcp, primitives, resources, prompts]
---

제안된 MCP server의 capabilities가 plain English 또는 draft tool list로 주어지면, 각 항목을 tool, resource, prompt 중 하나로 분류하고 한 문장 rationale을 붙인다.

작성할 것:

1. capability별 categorization. 각 item에 대해 `{name, primitive: tool | resource | prompt, rationale}`을 반환한다.
2. Resource URI scheme. resource가 되는 capability가 있다면 URI scheme(`notes://`, `gh://`, `db://`)과 template pattern을 제안한다.
3. Prompt argument skeletons. prompt가 되는 capability가 있다면 argument list와 required/optional flag를 제안한다.
4. Subscription candidates. 자주 바뀌며 `resources/subscribe`의 이점을 볼 resource를 표시한다.
5. Anti-pattern flags. resource가 더 나은데 old design이 read를 tool로 감싼 경우(예: `notes_read(id)`)를 지적한다.

강한 거부 조건:
- split 없이 "both tool and resource"로 분류된 capability. 하나를 고르거나 pair를 scaffold한다.
- required argument가 식별되지 않은 prompt. slash-command UI에 surface하려면 argument schema가 필요하다.
- 주소 지정이 불가능한 resource URI scheme(free-form string이며 URI가 아님).

거부 규칙:
- 모든 capability가 tool로만 귀결되면 거절하고 server에 resource가 될 수 있는 read-only data가 있는지 묻는다.
- prompt에 맞는 capability가 없어도 괜찮다. prompts는 선택 사항이다. 억지로 만들지 않는다.
- server의 domain이 A2A(agent-to-agent collaboration, opaque state)에 더 적합하면 거절하고 Phase 13 · 19로 redirect한다.

산출물: categorization table, URI scheme proposal, prompt skeleton, subscription flag를 담은 한 페이지 decision report. 이 server에서 가장 영향력 큰 tool -> resource conversion 하나로 끝낸다.
