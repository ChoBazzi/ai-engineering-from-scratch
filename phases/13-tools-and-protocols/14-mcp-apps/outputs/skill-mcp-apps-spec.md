---
name: mcp-apps-spec
description: interactive UI resource가 필요한 도구를 위한 전체 MCP Apps contract를 만듭니다.
version: 1.0.0
phase: 13
lesson: 14
tags: [mcp, apps, ui-resources, csp, iframe-sandbox]
---

interactive UI(timeline, form, dashboard, map, chart)가 있으면 좋은 도구가 주어지면, MCP Apps contract를 만드세요.

다음을 산출하세요.

1. `ui://` URI. UI resource의 canonical name 하나(예: `ui://notes/timeline`).
2. Tool result shape. `text` preamble과 `ui_resource` block이 있는 `content[]`, 채워진 `_meta.ui`.
3. CSP. `default-src`, `script-src`, `connect-src`, `img-src`, `style-src`의 최소 allowlist. 꼭 필요하지 않다면 `'unsafe-inline'`을 피하세요.
4. Permissions list. 필요하면 camera / mic / geolocation / network, 필요 없으면 비워 둡니다.
5. postMessage entry points. UI가 호출할 `host.*` call과 반환값.
6. Security checklist. host와 구분하기, clickjacking 없음, strict connect-src, 사용자 콘텐츠를 렌더링한다면 HTML sanitization.

강한 거부 조건:
- `default-src *`가 있는 CSP. 완전히 열린 보안 위험입니다.
- UI가 실제로 사용하지 않는 `permissions` 요청. 최소 권한을 지키세요.
- 외부 script를 load하는 모든 ui:// resource. bundle하거나 거부하세요.
- sanitization 없이 user-controlled HTML을 렌더링하는 모든 UI. XSS 벡터입니다.

거부 규칙:
- UI가 단순 static result라면 App scaffolding을 거부하고 text content를 반환하세요.
- 도구가 native host widgets(progress bars, confirmation dialogs)로 충분하다면 그쪽을 권장하세요.
- 호스트가 아직 MCP Apps를 지원하지 않는다면(2026-04 기준 VS Code stable, Zed, Windsurf), fallback-to-text 경로를 표시하세요.

출력: `ui://` URI, tool result JSON, CSP, permissions, postMessage entry points, security checklist가 담긴 한 페이지 contract. 이 UI를 렌더링할 최소 host에 대한 한 문장으로 끝내세요.
