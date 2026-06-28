# MCP Security II — OAuth 2.1, Resource Indicator, Incremental Scope

> 원격 MCP 서버에는 authentication뿐 아니라 authorization이 필요합니다. 2025-11-25 스펙은 OAuth 2.1 + PKCE + resource indicators(RFC 8707) + protected-resource metadata(RFC 9728)에 맞춰 정렬됩니다. SEP-835는 403 WWW-Authenticate에서 step-up authorization으로 incremental scope consent를 추가합니다. 이 lesson은 모든 hop을 볼 수 있도록 step-up flow를 state machine으로 구현합니다.

**Type:** Build
**Languages:** Python (stdlib, OAuth state machine simulator)
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security I)
**Time:** ~75 minutes

## 학습 목표

- resource server와 authorization server의 책임을 구분합니다.
- PKCE로 보호되는 OAuth 2.1 authorization code flow를 따라갑니다.
- `resource`(RFC 8707)와 protected-resource metadata(RFC 9728)를 사용해 confused-deputy attack을 방지합니다.
- step-up authorization을 구현합니다. 서버가 더 높은 scope를 요구하는 WWW-Authenticate와 함께 403을 응답하면, client가 사용자 동의를 다시 요청하고 재시도합니다.

## 문제

초기 MCP(2025년 이전)는 원격 서버에 ad-hoc API key를 붙이거나 auth가 아예 없는 상태로 제공되기도 했습니다. 2025-11-25 스펙은 완전한 OAuth 2.1 profile로 그 간극을 닫습니다.

현실에서 필요한 세 가지:

- **일반 원격 서버.** 사용자가 Notion / GitHub / Gmail에 접근하는 원격 MCP 서버를 설치합니다. OAuth 2.1 with PKCE가 올바른 형태입니다.
- **Scope escalation.** `notes:read`를 grant받은 notes 서버가 특정 작업에서 나중에 `notes:write`를 필요로 할 수 있습니다. 전체 flow를 다시 수행하는 대신 step-up(SEP-835)이 추가 scope를 요청합니다.
- **Confused deputy prevention.** Client가 Server A에 audience-scoped된 token을 들고 있습니다. Server A가 악의적이라 이 token을 Server B에 제시하려고 합니다. Resource indicators(RFC 8707)는 token을 의도된 audience에 고정합니다.

OAuth 2.1 자체가 새로운 것은 아닙니다. 새로운 것은 MCP의 profile입니다. 특정 required flow(authorization code + PKCE만 허용, implicit 없음, 기본적으로 client credentials 없음), 모든 token request에 mandatory resource indicator, client가 어디로 가야 하는지 알 수 있도록 게시되는 protected-resource metadata입니다.

## 개념

### Roles

- **Client.** MCP client(Claude Desktop, Cursor 등).
- **Resource server.** MCP server(notes, GitHub, Postgres 등).
- **Authorization server.** token을 발급합니다. resource server와 같은 서비스일 수도 있고 별도 IdP(Auth0, Keycloak, Cognito)일 수도 있습니다.

MCP profile에서 resource server와 authorization server는 같은 host일 수 있지만 URL로 구분하는 것이 좋습니다.

### Authorization code + PKCE

flow:

1. Client가 `code_verifier`(random)와 `code_challenge`(SHA256)를 생성합니다.
2. Client가 사용자를 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`으로 redirect합니다.
3. 사용자가 동의합니다. Authorization server가 `redirect_uri?code=...`로 redirect합니다.
4. Client가 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`로 POST합니다.
5. Authorization server가 저장된 challenge와 verifier의 hash를 검증하고 access token을 발급합니다.
6. Client가 token을 사용합니다. resource server에 대한 모든 요청에 `Authorization: Bearer ...`를 붙입니다.

PKCE는 authorization-code interception attack을 막습니다. Resource indicators는 token이 다른 곳에서 유효하지 않게 합니다.

### Protected-resource metadata (RFC 9728)

Resource server는 `.well-known/oauth-protected-resource` 문서를 게시합니다.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

Client는 resource server에서 authorization server를 discovery합니다. 구성 부담이 줄어듭니다. client는 resource URL만 알면 됩니다.

### Resource indicators (RFC 8707)

Token request의 `resource` parameter는 token의 의도된 audience를 고정합니다. 발급된 token에는 `aud: "https://notes.example.com"`이 포함됩니다. 이 token을 받은 다른 MCP 서버는 `aud`를 확인하고 거절합니다.

### Scope model

Scope는 공백으로 구분된 문자열입니다. 일반적인 MCP convention:

- `notes:read`, `notes:write`, `notes:delete`
- admin capability에는 `admin:*`(아껴서 사용)
- identity에는 `profile:read`

Scope 선택은 least-privilege여야 합니다. 지금 필요한 것을 요청하고, 더 필요할 때 step up합니다.

### Step-up authorization (SEP-835)

사용자가 `notes:read`를 grant합니다. 나중에 agent에게 note 삭제를 요청합니다. 서버가 응답합니다.

```text
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

Client는 `insufficient_scope` error를 보고 추가 scope에 대한 consent dialog를 사용자에게 표시하고, 이를 위한 mini OAuth flow를 수행한 뒤 새 token으로 request를 재시도합니다.

### Token audience validation

모든 request에서 server는 `token.aud == self.resource_url`을 확인합니다. 불일치하면 401입니다. 이것이 cross-server token reuse를 막습니다.

### Short-lived tokens and rotation

Access token은 short-lived(기본 1시간)여야 합니다. Refresh token은 refresh마다 rotate됩니다. Client는 background에서 silent refresh를 처리합니다.

### No token passthrough

Sampling server(Phase 13 · 11)는 client token을 다른 서비스로 pass through하면 안 됩니다. sampling request가 boundary입니다.

### Confused deputy prevention

Token은 `aud`에 bind됩니다. Client는 `client_id`에 bind됩니다. 모든 request는 둘 모두에 대해 validate됩니다. 스펙은 MCP 이전 원격 도구 ecosystem에서 흔했던 오래된 "pass-the-token" 패턴을 명시적으로 금지합니다.

### Client ID discovery

각 MCP client는 고정 URL에 metadata를 게시합니다. Authorization server는 client의 metadata document를 fetch해 redirect URI와 contact info를 discovery할 수 있습니다. 이로써 수동 client registration이 제거됩니다.

### Gateways and OAuth

Phase 13 · 17은 enterprise gateway가 OAuth를 처리하는 방식을 보여줍니다. gateway가 upstream server credential을 보관하고, client에게는 gateway-issued token을 발급하며, upstream token은 gateway를 떠나지 않습니다. 신뢰 모델이 뒤집힙니다. 사용자는 gateway에 한 번 authenticate하고, gateway가 N개의 server authorization을 처리합니다.

## 사용하기

`code/main.py`는 전체 OAuth 2.1 step-up flow를 state machine으로 simulate합니다. 구현 내용:

- PKCE code-verifier / challenge 생성.
- resource indicator가 있는 authorization code flow.
- Protected-resource metadata endpoint.
- audience check가 있는 token validation.
- `insufficient_scope`에서 step-up.

이 lesson에는 HTTP server가 없습니다. 모든 hop을 trace할 수 있도록 state machine이 memory에서 실행됩니다. Phase 13 · 17의 gateway lesson은 이를 실제 transport에 연결합니다.

## 산출물

이 lesson은 `outputs/skill-oauth-scope-planner.md`를 만듭니다. 도구가 있는 원격 MCP 서버가 주어지면 이 skill은 scope 집합, pinning rule, step-up policy를 설계합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 두 scope step-up flow를 trace하세요. step-up에서 어떤 hop이 반복되는지 기록하세요.

2. refresh-token rotation을 추가하세요. 모든 refresh가 새 refresh token을 발급하고 이전 token을 invalidate하게 만드세요. 도난당한 refresh token이 rotation 뒤에 사용되는 상황을 simulate하고 실패하는지 확인하세요.

3. protected-resource metadata endpoint를 stdlib http.server를 사용한 실제 HTTP response로 구현하세요. Lesson 09의 /mcp endpoint를 mirror하세요.

4. GitHub MCP 서버의 scope hierarchy를 설계하세요. read repo, write PR, approve PR, merge PR, admin. 각 level 사이에 step-up을 사용하세요.

5. RFC 8707과 RFC 9728을 읽으세요. MCP가 RFC 예시와 다르게 사용하는 9728의 필드 하나를 식별하세요. (힌트: `scopes_supported`와 관련 있습니다.)

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| OAuth 2.1 | "Modern OAuth" | PKCE를 의무화하고 implicit flow를 금지하는 통합 RFC |
| PKCE | "Proof-of-possession" | authorization-code interception을 막는 code verifier + challenge |
| Resource indicator | "Token audience" | token을 한 server에 고정하는 RFC 8707 `resource` parameter |
| Protected-resource metadata | "Discovery doc" | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | "Incremental consent" | 필요할 때 scope를 추가하는 SEP-835 flow |
| `insufficient_scope` | "403 with WWW-Authenticate" | 더 큰 scope에 대해 다시 consent하라는 server signal |
| Confused deputy | "Token reuse across services" | trusted holder가 token을 부적절하게 forward하는 attack |
| Short-lived token | "Access token TTL" | 빠르게 expire되는 bearer. refresh token이 갱신함 |
| Scope hierarchy | "Least privilege stack" | level 사이에 step-up이 있는 단계형 scope 집합 |
| Client ID metadata | "Client discovery doc" | client가 자신의 OAuth metadata를 게시하는 URL |

## 더 읽을거리

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — canonical MCP OAuth profile
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 변경 사항 walk-through
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience-pinning RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — discovery-document RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 실용적인 step-up-flow walk-through
