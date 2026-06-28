# MCP Resources와 Prompts — Tools 너머의 Context 노출

> MCP에서 attention의 90퍼센트는 tools가 가져간다. 하지만 나머지 두 server 프리미티브는 다른 문제를 해결한다. Resources는 읽기 위한 data를 노출하고, prompts는 재사용 가능한 template을 slash-command로 노출한다. 많은 server는 read를 tool로 감싸는 대신 resources를 사용해야 하고, workflow를 client prompt에 hard-code하는 대신 prompts를 사용해야 한다. 이 lesson은 decision rule을 정리하고 `resources/*`와 `prompts/*` message를 따라간다.

**Type:** Build
**Languages:** Python (stdlib, resource + prompt handler)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 학습 목표

- 주어진 domain에서 capability를 tool, resource, prompt 중 무엇으로 노출할지 결정한다.
- `resources/list`, `resources/read`, `resources/subscribe`를 구현하고 `notifications/resources/updated`를 처리한다.
- argument template과 함께 `prompts/list`와 `prompts/get`을 구현한다.
- host가 prompt를 slash-command로 노출하는 경우와 context를 자동 inject하는 경우를 구분한다.

## 문제

notes app을 위한 순진한 MCP server는 모든 것을 tool로 노출한다. `notes_read`, `notes_list`, `notes_search`가 그 예다. 이는 모든 data access를 model-driven tool call로 감싼다. 결과는 다음과 같다.

- context가 도움이 될 수 있는 모든 query마다 model이 `notes_read`를 호출할지 결정해야 한다.
- read-only content를 subscribe하거나 host의 side panel로 stream할 수 없다.
- client UI(Claude Desktop의 resource attachment panel, Cursor의 "Include file" picker)가 data를 surface할 수 없다.

올바른 분리는 다음과 같다. data는 resource로 노출하고, mutating 또는 computed action은 tool로 노출하며, 재사용 가능한 multi-step workflow는 prompt로 노출한다. 각 프리미티브에는 고유한 UX affordance와 access pattern이 있다.

## 개념

### Tools vs resources vs prompts — decision rule

| 사용자가 원하는 일 | Primitive |
|------------|-----------|
| 사용자가 데이터를 검색, 필터링, 변환하려 함 | tool |
| 사용자가 host가 이 데이터를 context로 포함하기를 원함 | resource |
| 사용자가 다시 실행할 수 있는 templated workflow를 원함 | prompt |

기준은 이렇다. 관련 query마다 model이 호출해서 이득을 본다면 tool이다. 사용자가 conversation에 attach해서 이득을 본다면 resource다. 사용자가 재사용하고 싶은 단위가 전체 multi-step workflow라면 prompt다.

### Resources

`resources/list`는 `{resources: [{uri, name, mimeType, description?}]}`를 반환한다. `resources/read`는 `{uri}`를 받고 `{contents: [{uri, mimeType, text | blob}]}`를 반환한다.

URI는 주소 지정 가능한 무엇이든 될 수 있다.

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14` (custom scheme)
- `memory://session-2026-04-22/recent` (server-specific)

`contents[]`는 text와 binary를 모두 지원한다. binary는 base64-encoded string인 `blob`과 `mimeType`을 사용한다.

### Resource subscriptions

capabilities에 `{resources: {subscribe: true}}`를 선언한다. client는 `resources/subscribe {uri}`를 호출한다. resource가 바뀌면 server가 `notifications/resources/updated {uri}`를 보낸다. client는 다시 읽는다.

use case는 disk의 file을 resource로 가진 notes server다. file watcher가 update notification을 trigger하고, host 밖에서 file이 edit되면 Claude Desktop이 해당 file을 context로 다시 가져온다.

### Resource templates(2025-11-25 addition)

`resourceTemplates`는 parameterized URI pattern을 노출하게 해 준다. 예를 들어 completion target인 `id`를 가진 `notes://{id}`가 있다. client는 resource picker에서 id를 autocomplete할 수 있다.

### Prompts

`prompts/list`는 `{prompts: [{name, description, arguments?}]}`를 반환한다. `prompts/get`은 `{name, arguments}`를 받고 `{description, messages: [{role, content}]}`를 반환한다.

prompt는 host가 model에 전달할 message list로 채워지는 template이다. 예를 들어 `code_review` prompt는 `file_path` argument를 받고 세 message sequence를 반환한다. system message, file body를 담은 user message, reasoning template이 있는 assistant kickoff가 그것이다.

### Hosts와 prompts

Claude Desktop, VS Code, Cursor는 chat UI에서 prompt를 slash-command로 노출한다. 사용자는 `/code_review`를 입력하고 form에서 argument를 고른다. server의 prompt는 "user shortcut"과 "model에 보내는 full prompt" 사이의 계약이다.

모든 client가 아직 prompt를 지원하는 것은 아니다. capability negotiation을 확인하라. server가 prompt capability를 선언했지만 client가 prompt를 지원하지 않으면 slash command가 보이지 않을 뿐이다.

### "list changed" notification

resources와 prompts는 set이 mutate될 때 `notifications/list_changed`를 emit한다. 새 note 20개를 방금 import한 notes server는 `notifications/resources/list_changed`를 emit한다. client는 추가분을 반영하기 위해 `resources/list`를 다시 호출한다.

### Content type conventions

text에는 `mimeType: "text/plain"`, `text/markdown`, `application/json`을 쓴다.
binary에는 `image/png`, `application/pdf`와 `blob` field를 쓴다.
MCP Apps(Lesson 14)에는 `ui://` URI 안에서 `text/html;profile=mcp-app`을 쓴다.

### Dynamic resources

resource URI가 static file에 대응해야 하는 것은 아니다. `notes://recent`는 읽을 때마다 최신 note 다섯 개를 반환할 수 있다. `db://query/users/active`는 parameterized query를 실행할 수 있다. server는 content를 동적으로 계산해도 된다.

규칙은 이렇다. client가 URI로 cache할 수 있다면 URI는 stable해야 한다. computation이 one-shot이라면 client cache가 stale되지 않도록 URI에 timestamp나 nonce를 포함해야 한다.

### Subscriptions vs polling

subscription-capable client는 `notifications/resources/updated`를 통해 server push를 받는다. subscription 이전 client나 이를 지원하지 않는 host는 다시 읽는 방식으로 poll한다. 둘 다 spec-compliant하다. server의 capability declaration이 무엇을 지원하는지 client에게 알려 준다.

subscription의 비용은 server의 session별 state다. 누가 무엇을 subscribe했는지 보관해야 한다. subscribed set은 bounded하게 유지하고, 연결이 끊긴 client는 timeout 처리해야 한다.

### Prompts vs system prompts

MCP의 prompts는 system prompts가 아니다. host의 system prompt(자체 운영 지침)와 MCP prompts(사용자가 invoke하는 server-supplied template)는 나란히 존재한다. 잘 동작하는 client는 server prompt가 자신의 system prompt를 override하게 두지 않는다. 대신 layer로 쌓는다.

## 사용하기

`code/main.py`는 Lesson 07의 notes server를 다음으로 확장한다.

- `resources/subscribe` support가 있는 note별 resource(`notes://note-1` 등).
- 세 message template으로 render되는 `review_note` prompt.
- note가 수정될 때 `notifications/resources/updated`를 emit하는 file-watcher simulation.
- 항상 최신 note 다섯 개를 반환하는 `notes://recent` dynamic resource.

demo를 실행해 전체 flow를 확인하라.

## 산출물

이 lesson은 `outputs/skill-primitive-splitter.md`를 만든다. 제안된 MCP server가 주어지면, 이 skill은 각 capability를 rationale과 함께 tool / resource / prompt로 분류한다.

## 연습 문제

1. `code/main.py`를 실행하라. initial resource list를 관찰한 다음 note edit을 trigger하고 `notifications/resources/updated` event가 발생하는지 검증하라.

2. `resources/list_changed` emitter를 추가하라. 새 note가 생성되면 client가 다시 discover하도록 notification을 보내라.

3. GitHub MCP server를 위한 prompt 세 개를 설계하라. `summarize_pr`, `triage_issue`, `release_notes`를 만들고 각각 argument schema를 포함하라. prompt body는 추가 수정 없이 runnable해야 한다.

4. Lesson 07 server의 기존 tool 하나를 골라 tool로 남겨야 하는지, resource와 tool pair로 나눠야 하는지 분류하라. 한 문장으로 정당화하라.

5. 사양의 `server/resources`와 `server/prompts` section을 읽어라. `resources/read`에서 드물게 채워지지만 spec-supported인 field 하나를 찾아라. 힌트: resource content의 `_meta`를 보라.

## 핵심 용어

| 용어 | 사람들이 흔히 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Resource | "Exposed data" | host가 읽을 수 있는 URI-addressable content |
| Resource URI | "data를 가리키는 pointer" | scheme prefix가 붙은 identifier(`file://`, `notes://` 등) |
| `resources/subscribe` | "변경 감시" | 특정 URI에 대한 client-opt-in server-push update |
| `notifications/resources/updated` | "Resource changed" | subscribed resource에 새 content가 있음을 client에 알리는 signal |
| Resource template | "Parameterized URI" | host picker를 위한 completion hint가 있는 URI pattern |
| Prompt | "Slash-command template" | argument slot을 가진 이름 있는 multi-message template |
| Prompt arguments | "Template inputs" | render 전에 host가 수집하는 typed parameter |
| `prompts/get` | "Render template" | server가 채워진 message list를 반환한다 |
| Content block | "Typed chunk" | `{type: text \| image \| resource \| ui_resource}` |
| Slash-command UX | "User shortcut" | host가 prompt를 `/`로 시작하는 command로 surface한다 |

## 더 읽을거리

- [MCP — Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) — resource URI, subscription, template
- [MCP — Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) — prompt template와 slash-command integration
- [MCP — Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) — 전체 `resources/*` message reference
- [MCP — Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) — 전체 `prompts/*` message reference
- [MCP — Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) — official docs를 확장한 community guide
