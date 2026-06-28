# MCP Apps — `ui://`를 통한 인터랙티브 UI 리소스

> 텍스트만 있는 도구 출력은 에이전트가 보여줄 수 있는 것을 제한합니다. MCP Apps(SEP-1724, 2026년 1월 26일 공식화)는 도구가 sandboxed interactive HTML을 반환해 Claude Desktop, ChatGPT, Cursor, Goose, VS Code 안에 inline으로 렌더링하게 합니다. 대시보드, 양식, 지도, 3D scene까지 하나의 extension으로 처리합니다. 이 lesson은 `ui://` resource scheme, `text/html;profile=mcp-app` MIME, iframe-sandbox postMessage protocol, 그리고 서버가 HTML을 렌더링하게 할 때 생기는 보안 표면을 다룹니다.

**Type:** Build
**Languages:** Python (stdlib, UI resource emitter), HTML (sample app)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 minutes

## 학습 목표

- 도구 호출에서 `ui://` 리소스를 반환하고 올바른 MIME과 metadata를 설정합니다.
- `_meta.ui.resourceUri`, `_meta.ui.csp`, `_meta.ui.permissions`로 도구의 연관 UI를 선언합니다.
- UI-to-host 통신을 위한 iframe sandbox postMessage JSON-RPC를 구현합니다.
- UI에서 시작되는 공격을 막는 CSP와 permissions-policy 기본값을 적용합니다.

## 문제

2025년식 `visualize_timeline` 도구는 "여기 시간순으로 정리된 노트 14개가 있습니다: ..."를 반환할 수 있습니다. 그것은 문단일 뿐입니다. 사용자가 실제로 원하는 것은 interactive timeline입니다. MCP Apps 이전의 선택지는 클라이언트별 widget API(Claude artifacts, OpenAI Custom GPT HTML)나 UI 없음뿐이었습니다.

MCP Apps(SEP-1724, 2026년 1월 26일 출시)는 계약을 표준화합니다. 도구 결과에는 URI가 `ui://...`이고 MIME이 `text/html;profile=mcp-app`인 `resource`가 들어갑니다. 호스트는 제한된 CSP와 명시적으로 허용되지 않은 네트워크 접근 금지를 적용한 sandboxed iframe 안에서 이를 렌더링합니다. iframe 내부 UI는 작은 postMessage JSON-RPC 방언으로 호스트에 메시지를 보냅니다.

호환되는 모든 클라이언트(Claude Desktop, ChatGPT, Goose, VS Code)는 같은 `ui://` 리소스를 같은 방식으로 렌더링합니다. 하나의 서버, 하나의 HTML bundle, 범용 UI입니다.

## 개념

### `ui://` resource scheme

도구는 다음을 반환합니다.

```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

그다음 호스트는 `ui://notes/timeline` URI에 대해 `resources/read`를 호출하고 다음을 받습니다.

```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### Iframe sandbox

호스트는 다음 설정의 sandboxed `<iframe>` 안에서 HTML을 렌더링합니다.

- `sandbox="allow-scripts allow-same-origin"`(또는 서버 선언에 따른 더 엄격한 값)
- 응답 헤더를 통해 적용되는 서버 선언 CSP.
- 호스트 origin의 쿠키나 localStorage 없음.
- CSP의 `connectSrc`로 제한된 네트워크 접근.

### postMessage protocol

iframe은 `window.postMessage`로 호스트와 통신합니다. 작은 JSON-RPC 2.0 방언입니다.

항상 `targetOrigin`을 상대의 정확한 origin으로 고정하고, 수신 측에서는 payload를 처리하기 전에 `event.origin`을 allowlist와 대조하세요. 이 채널의 어느 쪽에서도 `"*"`를 쓰지 마세요. 본문에는 도구 호출과 리소스 읽기가 들어갑니다.

```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

UI가 호출할 수 있는 host-side 메서드는 다음과 같습니다.

- `host.callTool(name, arguments)` — 서버 도구를 호출합니다.
- `host.readResource(uri)` — MCP 리소스를 읽습니다.
- `host.getPrompt(name, arguments)` — 프롬프트 템플릿을 가져옵니다.
- `host.close()` — UI를 닫습니다.

모든 호출은 여전히 MCP protocol을 거치며 서버의 권한을 상속합니다.

### Permissions

`_meta.ui.permissions` 목록은 추가 capability를 요청합니다.

- `camera` — 사용자 카메라 접근(문서 스캔 UI에 사용).
- `microphone` — 음성 입력.
- `geolocation` — 위치.
- `network:*` — `connectSrc`만으로 허용되는 것보다 넓은 네트워크 접근.

각 permission은 UI가 렌더링되기 전에 사용자가 보게 되는 prompt입니다.

### 보안 위험

iframe 안의 HTML도 여전히 HTML입니다. 새로운 공격 표면이 생깁니다.

- **UI를 통한 prompt-injection.** 악성 서버 UI가 system message처럼 보이는 텍스트를 보여줘 사용자를 속일 수 있습니다. 호스트 렌더링은 서버 UI와 호스트 UI를 시각적으로 구분해야 합니다.
- **`connectSrc`를 통한 exfiltration.** CSP가 `connect-src: *`를 허용하면 UI는 데이터를 어디든 보낼 수 있습니다. 기본값은 엄격해야 합니다.
- **Clickjacking.** UI가 호스트 chrome 위에 겹칩니다. 호스트는 z-index 조작을 막고 opacity 규칙을 강제해야 합니다.
- **Focus 탈취.** UI가 keyboard focus를 가져가 다음 메시지를 캡처합니다. 호스트가 이를 가로채야 합니다.

Phase 13 · 15는 MCP 보안의 일부로 이를 깊이 다룹니다. 이 lesson에서는 먼저 소개합니다.

### `ui/initialize` handshake

iframe이 로드된 뒤 postMessage로 `ui/initialize`를 보냅니다.

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

호스트는 capabilities와 session token으로 응답합니다. UI는 이후의 모든 host call에 session token을 사용합니다.

### AppRenderer / AppFrame SDK primitives

ext-apps SDK는 두 가지 편의 primitive를 노출합니다.

- `AppRenderer`(서버 측) — React / Vue / Solid component를 감싸 올바른 MIME과 metadata가 있는 `ui://` 리소스를 내보냅니다.
- `AppFrame`(클라이언트 측) — 리소스를 받고, iframe을 mount하며, postMessage를 중재합니다.

이를 사용해도 되고 HTML과 JSON-RPC를 직접 작성해도 됩니다.

### 생태계 상태

MCP Apps는 2026년 1월 26일 출시되었습니다. 2026년 4월 기준 클라이언트 지원 상태는 다음과 같습니다.

- **Claude Desktop.** 2026년 1월부터 완전 지원.
- **ChatGPT.** Apps SDK를 통해 완전 지원(같은 underlying MCP Apps protocol).
- **Cursor.** Beta. settings에서 활성화.
- **VS Code.** Insider builds only.
- **Goose.** 완전 지원.
- **Zed, Windsurf.** Roadmapped.

프로덕션 서버 사례: dashboards, map visualizations, data tables, chart builders, sandbox IDE previews.

## 사용하기

`code/main.py`는 notes 서버를 `visualize_timeline` 도구로 확장합니다. 이 도구는 `ui://notes/timeline` 리소스를 반환하고, 해당 URI의 `resources/read` handler는 SVG timeline이 들어간 작지만 완전한 HTML bundle을 반환합니다. HTML은 stdlib로 template 처리됩니다. 빌드 시스템은 없습니다. stdlib만으로는 브라우저를 구동할 수 없으므로 postMessage는 JS 주석으로 스케치되어 있습니다.

살펴볼 점:

- 도구 응답의 `_meta.ui`가 resourceUri, CSP, permissions를 담습니다.
- HTML은 네트워크 접근 없이 렌더링됩니다. 모든 데이터가 inline입니다.
- JS는 `window.parent.postMessage`를 통해 `host.callTool`을 호출합니다. 이 stdlib 데모에서는 문서화되어 있지만 inert 상태입니다.

## 산출물

이 lesson은 `outputs/skill-mcp-apps-spec.md`를 만듭니다. interactive UI가 있으면 좋은 도구가 주어지면, 이 skill은 전체 MCP Apps 계약을 만듭니다. `ui://` URI, CSP, permissions, postMessage entrypoints, 보안 checklist가 포함됩니다.

## 연습

1. `code/main.py`를 실행하고 내보낸 HTML을 검사하세요. HTML을 브라우저에서 직접 열어 SVG가 렌더링되는지 확인하세요. 그런 다음 UI가 `host.callTool("notes_update", ...)`를 호출하는 데 사용할 postMessage 계약을 스케치하세요.

2. CSP를 강화하세요. `'unsafe-inline'`을 제거하고 nonce 기반 script policy를 사용하세요. HTML 생성 코드에서 무엇이 바뀌나요?

3. 노트를 제자리에서 편집하는 양식이 있는 두 번째 UI 리소스 `ui://notes/editor`를 추가하세요. 사용자가 제출하면 iframe은 `host.callTool("notes_update", ...)`를 호출합니다.

4. UI의 공격 표면을 audit하세요. 악성 서버가 어디에 콘텐츠를 주입할 수 있나요? iframe sandbox는 무엇을 막고 무엇은 막지 못하나요?

5. SEP-1724 스펙을 읽고 이 toy 구현이 사용하지 않는 MCP Apps SDK capability 하나를 찾으세요. 힌트: component-level state sync.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| MCP Apps | "Interactive UI resources" | 2026-01-26에 출시된 SEP-1724 extension |
| `ui://` | "App URI scheme" | UI bundle을 위한 resource scheme |
| `text/html;profile=mcp-app` | "그 MIME" | MCP App HTML의 Content-type |
| Iframe sandbox | "렌더 컨테이너" | CSP와 permissions가 적용된 UI의 브라우저 sandboxing |
| postMessage JSON-RPC | "UI-to-host wire" | host call을 위한 작은 JSON-RPC-over-postMessage 방언 |
| `_meta.ui` | "도구-UI binding" | 도구 결과를 UI 리소스에 연결하는 metadata |
| CSP | "Content-Security-Policy" | scripts, network, styles의 허용 source를 선언 |
| AppRenderer | "서버 SDK primitive" | framework component를 `ui://` 리소스로 변환 |
| AppFrame | "클라이언트 SDK primitive" | postMessage를 중재하는 iframe mount helper |
| `ui/initialize` | "Handshake" | UI가 host로 보내는 첫 postMessage |

## 더 읽을거리

- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — reference implementation과 SDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — 공식 spec 문서
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — 고수준 문서
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — 2026년 1월 launch post
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — JSDoc 스타일 SDK reference
