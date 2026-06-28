# Production MCP Auth — Enrollment, JWKS Refresh, Audience-Pinned Token

> Lesson 16은 OAuth 2.1 state machine을 memory 안에 세웠습니다. 2026년에는 실제 조직에 ship하는 모든 MCP server가 production auth 뒤에 있습니다. 무한한 client population으로 scale되는 client enrollment(Client ID Metadata Documents 우선, backwards-compatible fallback으로 dynamic client registration), authorization-server metadata discovery(RFC 8414 *또는* OpenID Connect Discovery), 새벽 3시 token validation을 깨지 않는 JWKS cache refresh, cross-resource replay를 거부하는 audience-pinned token이 필요합니다. 이 lesson은 authorization server, resource server(MCP server), client라는 세 role로 전체 surface를 모델링해 discovery부터 validated tool call까지 모든 hop을 추적할 수 있게 합니다.
>
> **Spec note (2025-11-25):** 2025년 11월 MCP authorization spec은 Dynamic Client Registration을 `SHOULD`에서 `MAY`로 낮추고 **Client ID Metadata Documents (CIMD)**를 권장 기본 enrollment mechanism으로 삼았습니다. 이 lesson은 스펙의 priority order대로 둘 다 가르치며, code는 한 process 안에서 완전히 self-contained인 walk-through를 위해 DCR을 유지합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 minutes

## 학습 목표

- RFC 8414 metadata를 통해 authorization server를 discovery하고 contract를 verify합니다.
- RFC 7591 dynamic client registration을 구현해 MCP client가 admin intervention 없이 enroll하게 합니다.
- key roll-over 중에도 signature verification이 살아남도록 JWKS key를 schedule에 맞춰 cache하고 refresh합니다.
- RFC 8707 resource indicator를 사용해 token을 단일 MCP resource에 pin하고 confused-deputy reuse를 거부합니다.
- 세 role(authorization server, resource server, client)을 깔끔하게 분리해 각자가 자기에게 속한 check만 enforce하게 합니다.
- IdP capability matrix를 읽고 IdP가 MCP auth profile을 만족하지 못하면 deploy를 거절합니다.

## 문제

Lesson 16 simulator는 memory에서 OAuth 2.1을 실행합니다. Production에는 memory-only simulator가 보지 못하는 세 가지 operational gap이 있습니다.

첫 번째 gap은 enrollment입니다. 실제 조직은 수백 개 MCP server와 수천 개 MCP client를 운영합니다. operator가 모든 Cursor user를 OAuth client로 손수 등록하지 않습니다. 2025-11-25 spec은 이를 해결하는 client priority order를 제공합니다. pre-registered `client_id`가 있으면 사용하고, 아니면 **Client ID Metadata Document**를 사용합니다(client가 자신이 control하는 HTTPS URL로 자신을 식별하고 authorization server가 metadata를 *pull*합니다). 그다음 fallback으로 **RFC 7591 dynamic client registration**을 사용합니다(client가 `POST /register`를 *push*하고 즉시 `client_id`를 받습니다). 그것도 아니면 user에게 prompt합니다. CIMD는 per-server registration을 완전히 없애면서 DNS-rooted trust model을 유지하므로 권장 기본값입니다. DCR은 backwards compatibility를 위해 유지됩니다. 둘 다 authorization server metadata에서 entry point를 discovery합니다. CIMD는 `client_id_metadata_document_supported`, DCR은 `registration_endpoint`입니다.

두 번째 gap은 key rotation입니다. JWT validation은 authorization server의 signing key에 의존하며, 이는 JSON Web Key Set(JWKS)으로 게시됩니다. authorization server는 이를 schedule에 맞춰 rotate합니다(보통 hourly, incident response 중에는 더 빠를 수 있음). boot 때 JWKS를 한 번만 fetch하는 MCP server는 rotation window까지는 잘 validate하다가 이후 restart 전까지 모든 request가 실패합니다. Production은 JWKS를 cached value로 wiring하고, 이전 key가 expire되기 전에 cache를 overwrite하는 refresh job을 둡니다. 또한 cache보다 새로운 key로 서명된 token이 도착하는 경우를 위해 cache miss에서 fallback fetch를 수행합니다.

세 번째 gap은 audience binding입니다. Lesson 16은 RFC 8707 resource indicator를 소개했습니다. Production에서 이 indicator는 모든 request의 hard claim check가 됩니다. MCP server는 `token.aud`를 자신의 canonical resource URL과 비교하고 mismatch를 HTTP 401로 거절합니다. 이것은 upstream MCP server(또는 한 server용 token을 가진 악의적 client)가 같은 trust mesh의 다른 server에 token을 replay하는 것을 막는 유일한 defense입니다.

이 lesson은 각 gap을 surface의 concrete piece에 매핑합니다. metadata document는 HTTP endpoint입니다. JWKS cache refresh는 scheduled job과 key-value cache입니다. JWT validation은 resource server가 tool을 dispatch하기 전에 실행하는 routine입니다. 세 role을 분리하세요. authorization server는 key를 issue하고 rotate하고, resource server는 cache하고 validate하고, client는 discover하고 enroll합니다.

## 개념

### RFC 8414 — OAuth Authorization Server Metadata

`/.well-known/oauth-authorization-server`의 document는 client가 필요한 모든 것을 설명합니다.

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

MCP resource URL을 받은 client는 discovery를 chain합니다. RFC 9728의 `oauth-protected-resource`(resource server의 document)가 issuer를 이름 붙이고, `oauth-authorization-server`(이 RFC)가 모든 endpoint를 이름 붙입니다. client는 authorization URL을 hard-code하지 않습니다.

MCP용 IdP를 신뢰하기 전에 verify해야 하는 contract:

- `code_challenge_methods_supported`가 `S256`을 포함합니다(RFC 7636에 따른 PKCE). 스펙은 명시적입니다. 이 field가 **absent**이면 authorization server는 PKCE를 지원하지 않으며 client는 진행을 **MUST** refuse해야 합니다.
- `grant_types_supported`가 `authorization_code`를 포함하고 `password`와 `implicit`을 거절합니다.
- 최소 하나의 enrollment path가 advertise됩니다. `client_id_metadata_document_supported: true`(CIMD, preferred) **또는** `registration_endpoint`(RFC 7591 DCR, fallback). 둘 중 하나면 contract를 만족합니다. 더 이상 DCR을 hard-require하지 않습니다.
- OAuth 2.1에서는 `response_types_supported`가 정확히 `["code"]`입니다.

`S256`이 빠져 있으면 MCP server는 이 IdP에 대해 deploy를 거절합니다. PKCE에는 degraded mode가 없습니다. enrollment path가 *둘 다* advertise되지 않았고 pre-registered `client_id`도 없다면 enroll할 수 없습니다. 이 경우 deployment manifest가 잘못된 것이지 code가 잘못된 것이 아닙니다.

### RFC 9728 (recap) — Protected Resource Metadata

Lesson 16에서 RFC 9728을 다뤘습니다. production에서의 차이는 이 document가 client가 *이* MCP server가 신뢰하는 authorization server를 찾는 유일한 위치라는 점입니다. 단일 MCP server가 여러 IdP의 token을 받을 수 있습니다(staff용 하나, partner용 하나). RFC 9728은 그 집합을 선언하고, RFC 8414는 각 IdP가 무엇을 지원하는지 문서화합니다.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents (the recommended default)

CIMD는 registration을 *push*에서 *pull*로 뒤집습니다. authorization server에게 `client_id`를 mint해 달라고 요청하는 대신, client는 자신이 control하는 HTTPS URL을 `client_id` **로** 사용합니다. URL은 JSON metadata document로 resolve되고, authorization server는 OAuth flow 중 필요할 때 이를 fetch합니다. Trust는 DNS에 root를 둡니다. server operator가 `app.example.com`을 신뢰한다면 `https://app.example.com/client.json`에서 served되는 client를 신뢰합니다. registration round-trip이 없고, 고갈될 `client_id` namespace가 없으며, server별 state를 동기화할 필요가 없습니다.

client가 host하는 metadata document:

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

document의 `client_id` 값은 served되는 URL과 같아야 **MUST** 합니다. authorization server가 이를 verify하고 mismatch를 reject합니다. authorization server는 RFC 8414 metadata에서 `client_id_metadata_document_supported: true`로 지원을 advertise합니다.

스펙이 분명히 말하는 security fact 두 가지:

- **SSRF.** authorization server가 attacker-supplied URL을 fetch합니다. server-side request forgery를 방어해야 합니다(internal/admin endpoint로 fetch 금지).
- **localhost impersonation.** CIMD만으로는 local attacker가 legitimate client's metadata URL을 claim하고 임의의 `localhost` redirect를 bind하는 것을 막을 수 없습니다. authorization server는 consent 중 redirect URI hostname을 명확히 표시해야 **MUST** 하며, `localhost`-only redirect에 warn해야 **SHOULD** 합니다.

CIMD는 server-side state가 필요 없으므로 DCR처럼 registrar를 세울 필요가 없습니다. client side는 read-only입니다. static HTTPS endpoint에서 metadata document를 serve하고 authorization server가 pull하게 두세요.

### RFC 7591 — Dynamic Client Registration (fallback / backwards compatibility)

DCR은 이제 `MAY`이며, 2025-11-25 이전 deployment 및 아직 CIMD를 지원하지 않는 IdP와의 backwards compatibility를 위해 유지됩니다. DCR이 없고 CIMD나 pre-registration도 없으면 모든 MCP client(Cursor, Claude Desktop, custom agent)는 IdP admin과 out-of-band exchange가 필요합니다. DCR을 쓰면 client는 다음을 post합니다.

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

server는 `client_id`와 나중 update에 쓸 `registration_access_token`을 응답합니다.

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none`은 사용자 device에서 실행되는 MCP client의 올바른 기본값입니다. 이들은 `client_id`만 받고 exfiltrate될 `client_secret`은 받지 않습니다. PKCE가 public client에 필요한 proof-of-possession을 제공합니다.

production pitfall 세 가지:

- registration endpoint는 source IP로 rate-limit해야 합니다. 그렇지 않으면 hostile actor가 수백만 개 fake registration을 script로 만들고 `client_id` namespace를 고갈시킵니다. registrar가 request를 처리하기 전에 rate-limit check를 실행하세요.
- `software_statement`(client를 보증하는 signed JWT)가 일부 enterprise IdP에서 필요합니다. lesson의 mock은 이를 건너뜁니다. production은 localhost redirect URI가 아닌 모든 unsigned registration을 reject하는 verification step을 연결합니다.
- `registration_access_token`은 plaintext가 아니라 hash로 저장해야 합니다. 이 token이 도난당하면 attacker가 client redirect URI를 rewrite할 수 있습니다.

### RFC 8707 (recap) — Resource Indicators

Lesson 16에서 shape를 세웠습니다. production rule은 다음입니다. 모든 token request는 `resource=<canonical-mcp-url>`을 포함하고, MCP server는 모든 call에서 `token.aud`가 자신의 resource URL과 일치하는지 verify합니다. canonical URI는 server의 *가장 구체적인* identifier입니다. lowercase scheme과 host를 사용하고 fragment가 없으며 convention상 trailing slash가 없습니다. path component는 rule로 strip되지 않습니다. individual MCP server를 식별하는 데 필요하면 spec은 path를 유지합니다. `https://mcp.example.com`, `https://mcp.example.com/mcp`, `https://mcp.example.com:8443`, `https://mcp.example.com/server/mcp`는 모두 valid canonical URI입니다. server마다 하나를 고르고 `aud`를 정확히 거기에 pin하세요. (이 lesson의 mock은 brevity를 위해 `https://notes.example.com` 같은 bare-host audience를 사용합니다. 하나의 origin에 여러 MCP server를 co-host하는 deployment는 path로 구분합니다.)

### RFC 7636 (recap) — PKCE

PKCE는 OAuth 2.1에서 mandatory입니다. lesson의 authorization-code flow는 항상 `code_challenge`와 `code_verifier`를 운반합니다. server는 verifier가 없거나 verifier가 저장된 challenge로 hash되지 않는 token request를 reject합니다.

### MCP Spec 2025-11-25 Auth Profile

MCP spec(2025-11-25)은 MCP server의 authorization layer가 해야 할 일을 정확히 규정합니다.

- RFC 9728 protected-resource metadata를 구현하고, 401의 `WWW-Authenticate: Bearer resource_metadata="..."` header **또는** well-known URI `/.well-known/oauth-protected-resource`를 통해 위치를 제공합니다(SEP-985는 header를 optional로 만들고 well-known fallback을 둠). metadata `authorization_servers` field는 최소 하나의 server를 이름 붙여야 **MUST** 합니다.
- **모든** request에서 `Authorization: Bearer ...`로만 token을 받습니다. query string에는 절대 넣지 않고, session start에서만 validate하지 않습니다.
- request마다 `aud`, `iss`, `exp`, required scope를 validate합니다. server는 token이 자신을 위해 specifically issued되었는지(audience) validate해야 **MUST** 합니다. missing 또는 mismatched `aud`는 reject되며 wildcard로 취급되지 않습니다.
- 401/403에서는 `error=...`, `resource_metadata="<PRM-URL>"` parameter(metadata document의 URL이지 bare resource가 아님), `insufficient_scope`(403)에서 `scope="..."`를 담은 `WWW-Authenticate: Bearer`를 반환합니다. 참고: parameter는 discovery pointer인 `resource_metadata`입니다. challenge에는 `resource` parameter가 없습니다.
- Authorization-server discovery는 RFC 8414 OAuth metadata **또는** OpenID Connect Discovery 1.0을 받습니다. client는 두 well-known suffix를 priority order대로 시도해야 합니다.
- **mix-up attack**은 server가 아니라 client가 방어합니다. client는 redirect 전 expected `issuer`를 record하고, code를 redeem하기 전에 `iss` authorization-response parameter(RFC 9207)를 validate합니다. PKCE만으로는 mix-up을 막지 못합니다. client가 steer된 token endpoint에 자기 `code_verifier`를 넘기기 때문입니다.

OAuth 2.1 draft는 substrate입니다. RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD가 surface입니다. MCP spec은 profile입니다.

### IdP capability matrix

모든 IdP가 full MCP profile을 지원하지는 않습니다. 아래 matrix는 2025-11-25 spec 기준의 factual capability statement를 문서화합니다. 이는 recommendation이 아니라 *deployment gate*입니다.

CIMD는 2025-11-25 spec에서 ship되었고 underlying OAuth draft는 2025년 10월에야 adopt되었으므로 vendor support는 아직 도착 중입니다. 아래의 "CIMD"는 permanent statement가 아니라 "현재 위치이며 tenant에서 verify하라"는 뜻으로 다루세요.

| IdP 범주 | AS metadata(8414/OIDC) | CIMD | RFC 7591 DCR | RFC 8707 resource | RFC 7636 S256 PKCE | 참고 |
|---|---|---|---|---|---|---|
| Self-hosted (Keycloak) | yes | emerging | yes | yes (since 24.x) | yes | 이 lesson의 MCP profile reference IdP. full DCR path end-to-end, CIMD는 새 spec을 추적 중. |
| Enterprise SSO (Microsoft Entra ID) | yes | emerging | yes (premium tiers) | yes | yes | DCR availability는 tenant tier마다 다릅니다. deploy 전 target tenant에서 verify하세요. |
| Enterprise SSO (Okta) | yes | emerging | yes (Okta CIC / Auth0) | yes | yes | DCR은 Auth0(현재 Okta CIC)에서 사용 가능. classic Okta org는 admin pre-registration 필요. |
| Social login IdPs (generic) | varies | no | rarely | rarely | yes | 대부분 social IdP는 client를 static partner로 취급합니다. identity source로만 쓰고 MCP-aware authorization server를 위에 layer하세요. |
| Custom / homegrown | depends | depends | depends | depends | depends | 직접 ship한다면 full profile을 ship하고 CIMD를 선호하세요. PKCE나 audience binding을 건너뛰면 MCP auth contract가 깨집니다. |

deployment manifest의 refusal rule: 선택한 IdP가 `code_challenge_methods_supported`에 `S256`을 list하지 않으면 MCP server는 start를 거절합니다. PKCE에는 degraded mode가 없습니다. Enrollment는 더 soft한 gate입니다. working path 하나가 필요합니다(pre-registered `client_id`, `client_id_metadata_document_supported: true`, 또는 `registration_endpoint`). CIMD나 pre-registration이 대신할 수 있으므로 DCR 부재만으로는 더 이상 refusal trigger가 아닙니다.

### JWKS refresh pattern (rotate at the AS, refresh at the resource server)

두 verb를 분리하세요. 혼동하면 실제 production bug가 됩니다.

- **Rotate**는 *authorization server*가 하는 일입니다. 새 signing key를 mint하고 JWKS에 게시한 뒤 나중에 old key를 retire합니다. resource server는 이 작업에 관여하지 않고 할 수도 없습니다. IdP의 private key를 들고 있지 않기 때문입니다.
- **Refresh**는 *resource server*가 하는 일입니다. published JWKS를 다시 `GET`해서 cache에 넣습니다. resource server가 JWKS에 대해 수행하는 유일한 action입니다.

production failure mode는 stale cache입니다. scheduled refresh job과 key-value cache로 해결하세요. resource server는 fixed interval로 `<issuer>/.well-known/jwks.json`을 fetch하고 `cache[issuer] = {keys, fetched_at}`를 overwrite하는 job(cron, timer, runtime이 제공하는 무엇이든)을 실행합니다. validator는 그 cache를 읽습니다. token의 `kid`가 cache에 없으면 fall-back으로 **한 번** synchronous refresh를 trigger한 뒤 다시 check합니다. 이렇게 하면 scheduled refresh와, 새 key로 서명된 token이 다음 scheduled refresh 전에 도착하는 key-overlap window를 동시에 처리합니다.

fall-back은 **반드시 re-fetch여야 하며 rotate가 아니어야 합니다**. cache-miss path를 rotate-and-mint에 연결하면 두 가지가 깨집니다. (1) fresh key를 mint해도 token과 맞는 `kid`가 아니므로 lookup은 어차피 실패합니다. (2) random `kid` 값을 가진 token을 뿌리는 attacker가 무제한 key creation을 강제합니다. self-inflicted DoS입니다. re-fetch는 idempotent이므로 bogus `kid`는 최대 한 번의 wasted fetch만 비용으로 듭니다.

cache shape:

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

steady state에서는 key가 두 개입니다. Authorization server는 previous key(`k_2026_03`)를 retire하기 전에 next key(`k_2026_04`)를 introduce하는 방식으로 rotate하므로 old key로 발급된 token은 expire될 때까지 valid합니다. cache는 union을 보관하고 validator는 `kid`로 선택합니다.

### The validation routine

MCP server는 tool dispatch 전 validation을 실행합니다. `code/main.py`가 쓰는 shape:

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate`는 JWT를 decode하고, JWKS cache에서 signing key를 resolve하며(miss 시 한 번 refresh), signature를 verify한 뒤, `iss`를 allow-list와, `aud`를 이 server의 canonical resource와, `exp`와 required scope를 check합니다. 첫 failure에서 `WWW-Authenticate` challenge를 반환합니다. 이것을 resource server의 단일 routine으로 유지하면 모든 entry point(모든 tool call, 모든 transport)가 같은 check를 거칩니다. validation 없이 tool에 도달하는 path가 없습니다.

### Audience-replay walkthrough (access-token privilege restriction)

Server A(`notes.example.com`)와 Server B(`tasks.example.com`)가 같은 authorization server에 register되어 있습니다. Server A가 compromise되었습니다. attacker가 사용자의 notes token을 가져와 Server B에 replay합니다.

Server B의 validator:

1. JWT를 decode하고 `kid`로 JWKS를 fetch해 signature를 verify합니다.
2. `iss`를 protected-resource metadata의 `authorization_servers`와 check합니다. (Pass — 같은 IdP.)
3. `aud == "https://tasks.example.com"`를 check합니다. (Fail — token의 `aud`는 `https://notes.example.com`.)
4. 401과 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`를 반환합니다.

audience claim은 protocol layer에서 이 attack에 대한 유일한 defense입니다. performance 때문에 건너뛰는 것이 가장 흔한 production mistake입니다. validator는 session start에서만이 아니라 모든 request에서 실행되어야 합니다. 스펙은 이를 **access-token privilege restriction**이라고 부릅니다. MCP server는 audience에 자신을 이름 붙이지 않은 token을 반드시 reject해야 **MUST** 합니다.

> **Naming note.** spec은 *confused deputy*라는 용어를 관련 있지만 구별되는 문제에 예약합니다. MCP server가 static client ID를 사용해 third-party API에 대한 OAuth **proxy**로 동작하면서 per-client user consent 없이 token을 forward하는 문제입니다. Audience binding은 위 replay를 고칩니다. confused-deputy fix는 per-client consent **plus** inbound token을 upstream API로 pass through하지 않는 것입니다(MCP server는 자신의 별도 upstream token을 받아야 **MUST** 합니다).

### Mix-up attacks (a client-side defense the server cannot provide)

client는 생애 동안 많은 authorization server와 대화합니다. 악의적 AS는 client가 honest AS의 authorization code를 attacker의 token endpoint에서 redeem하게 만들 수 있습니다. audience binding은 여기서 도움이 되지 않습니다. attack은 token이 존재하기 전에 일어납니다. defense는 client에 있습니다(RFC 9207).

1. redirect 전 client는 validated AS metadata에서 expected `issuer`를 record합니다.
2. authorization response에서 client는 반환된 `iss` parameter를 record된 issuer와 비교합니다(normalization 없이 simple string comparison). code를 어디에도 보내기 전에 수행합니다.
3. mismatch(또는 AS가 `authorization_response_iss_parameter_supported`를 advertise했는데 `iss`가 absent) → reject하고 `error` field도 표시하지 않습니다.

PKCE만으로는 mix-up을 막지 못합니다. client가 steer된 token endpoint에 자기 `code_verifier`를 넘기기 때문입니다. 그래서 spec은 issuer를 PKCE verifier 및 `state`와 함께 request별로 record합니다.

### Failure modes

- **Stale JWKS.** AS가 key를 rotate한 뒤 validator가 valid token을 reject합니다. fix는 위의 cron-refresh + cache-miss-refetch pattern입니다. refresh job 없이 JWKS를 cache하지 마세요.
- **Rotate-as-fall-back.** cache-miss path를 re-fetch가 아니라 rotate-and-mint에 연결하는 것은 실제 bug입니다. missing `kid`를 만들지 못하고 attacker-controlled `kid` 값을 key-creation DoS로 바꿉니다. fall-back은 idempotent `refresh-jwks`여야 합니다.
- **Missing `aud` claim.** 일부 IdP는 token request에 `resource`가 없으면 기본적으로 `aud`를 omit합니다. validator는 missing `aud` token을 reject해야 하며 absence를 wildcard로 취급하면 안 됩니다.
- **Mix-up via missing `iss` check.** redirect 전에 record한 issuer와 RFC 9207 `iss` authorization-response parameter를 validate하지 않는 client는 honest AS의 code를 attacker의 token endpoint에서 redeem하도록 steer될 수 있습니다. 이는 client-side failure입니다. resource server는 보상할 수 없습니다.
- **Scope upgrade race.** 같은 user에 대한 두 concurrent step-up flow가 모두 성공해 서로 다른 scope의 access token 두 개를 만들 수 있습니다. validator는 request에 제시된 token을 사용해야지 "user의 current scope"를 lookup하면 안 됩니다. 그렇게 하면 TOCTOU window가 생깁니다.
- **Registration token theft.** leaked `registration_access_token`은 attacker가 redirect URI를 rewrite하게 합니다. 이를 at rest에서 hash하고, 모든 update에서 cleartext 제시를 요구하며, 의심 시 rotate하세요.
- **`iss` not pinned.** 어떤 `iss`든 받아들이는 validator는 attacker가 자기 authorization server를 세우고 target audience용 client를 등록해 token을 issue하게 합니다. protected-resource metadata의 `authorization_servers` list가 allow-list입니다. enforce하세요.

## 사용하기

`code/main.py`는 stdlib Python과 세 role(`AuthorizationServer`, `ResourceServer`, `Client`)로 full production flow를 따라갑니다. flow:

1. Authorization server가 `/.well-known/oauth-authorization-server`에 RFC 8414 metadata를 게시합니다.
2. MCP client가 metadata endpoint를 호출하고 enrollment option(`client_id_metadata_document_supported` for CIMD, `registration_endpoint` for DCR)과 `S256` PKCE support를 check합니다.
3. walk-through는 DCR fallback path를 택합니다. client가 `/register`(RFC 7591)에 post하고 `client_id`를 받습니다. (CIMD client라면 대신 자신의 HTTPS `client_id` URL을 제시하고 이 step을 건너뜁니다.)
4. MCP client가 `resource` indicator(RFC 8707)가 있는 PKCE-protected authorization code flow(RFC 7636)를 실행합니다.
5. MCP client가 `Authorization: Bearer ...`로 MCP server의 tool을 호출합니다.
6. MCP server가 JWKS cache에서 signing key를 resolve하며 `validate`를 실행합니다.
7. IdP가 key를 rotate합니다. scheduled refresh가 JWKS를 cache로 다시 pull합니다.
8. 다음 call은 restart 없이 refreshed key로 validate되고, 이전 token도 overlap window 동안 계속 validate됩니다.
9. 다른 MCP resource에 대한 audience-replay attempt는 `audience mismatch`와 `resource_metadata` pointer가 있는 401을 받습니다.

여기서 JWT는 HS256과 shared secret을 사용합니다(lesson이 stdlib only로 실행되도록). Production은 위 JWKS pattern과 함께 RS256 또는 EdDSA를 사용합니다. validation logic은 otherwise identical입니다. IdP와 resource server가 한 process 안에 있으므로 `refresh_jwks`는 authorization server의 key list를 직접 읽습니다. wire에서는 `jwks_uri`에 대한 HTTP `GET`입니다.

## 산출물

이 lesson은 `outputs/skill-mcp-auth.md`를 만듭니다. MCP server config와 IdP capability set이 주어지면 이 skill은 세워야 할 auth surface를 emit합니다. protected-resource metadata, 사용할 enrollment path(CIMD, pre-registration, DCR fallback), JWKS refresh schedule, scope mapping, IdP가 full RFC profile을 지원하지 않을 때 적용할 refusal rule을 포함합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. flow를 trace하세요. step 6에서 IdP가 key를 rotate하고, scheduled `refresh_jwks`가 published set을 다시 pull하며, old token(overlap window)과 fresh token이 restart 없이 모두 validate되는 방식을 기록하세요.

2. protected-resource metadata의 `authorization_servers` list에 새 IdP를 추가하세요. 새 IdP가 서명한 token을 issue하고 validator가 accept하는지 확인하세요. list에 없는 IdP가 서명한 token을 issue하고 validator가 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`로 reject하는지 확인하세요.

3. registrar가 request를 accept하기 전에 실행되는 rate-limit check를 `register_client`에 추가하세요. source IP를 key로 하는 작은 dict에 token-bucket을 보관하세요.

4. RFC 7591을 읽고 lesson의 `/register` handler가 validate하지 않는 field 두 개를 식별하세요. validation을 추가하세요. (힌트: `software_statement`와 `redirect_uris` URI scheme.)

5. Client ID Metadata Document path를 추가하세요. `client_id`가 자기 URL과 같은 `client.json`을 serve하고, authorization server가 fetch하고 verify하게 하세요(`client_id` ≠ URL이면 reject). CIMD client가 `register_client` call 없이 enroll되는지 확인하세요.

6. DoS fix를 증명하세요. random `kid`가 있는 token을 validator에 보내고 `refresh_jwks`가 최대 한 번 실행되며 authorization server의 key count가 늘지 않는지 확인하세요. 그런 다음 일부러 fall-back을 rotate-and-mint로 다시 wire하고 bogus token마다 key count가 증가하는 것을 관찰하세요. 이후 re-fetch를 복원하세요.

7. mix-up section의 client-side RFC 9207 `iss` check를 구현하세요. authorization request 전에 expected issuer를 record한 다음, `iss`가 일치하지 않는 authorization response를 reject하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| ASM | "OAuth metadata document" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| CIMD | "Client metadata URL" | Client ID Metadata Document. `client_id`로 쓰이는 HTTPS URL이며 AS가 JSON을 pull함. 2025-11-25 이후 권장 기본값 |
| DCR | "Self-service client registration" | RFC 7591 `POST /register` flow. 2025-11-25에서 `MAY` fallback으로 낮아짐 |
| JWKS | "Public keys for JWT validation" | `jwks_uri`에서 fetch하고 `kid`로 index하는 JSON Web Key Set |
| Rotate vs refresh | "Updating the keys" | *Rotate* = AS가 signing key를 mint/retire. *refresh* = resource server가 published set을 re-fetch. resource server는 refresh만 함 |
| Resource indicator | "Audience parameter" | token을 한 server에 pin하는 RFC 8707 `resource` parameter |
| `aud` claim | "Audience" | validator가 canonical resource URL과 비교하는 JWT claim |
| Audience replay | "Token replay" | Server A용 token을 Server B에 제시. audience validation으로 방어(spec: access-token privilege restriction) |
| Confused deputy | "Proxy token misuse" | static client ID가 있는 MCP proxy가 per-client consent 없이 token을 forward하는 문제. audience replay와 구별됨 |
| Mix-up attack | "Wrong token endpoint" | client가 honest AS의 code를 attacker endpoint에서 redeem하도록 steer됨. RFC 9207 `iss`로 client-side 방어 |
| `iss` allow-list | "Trusted authorization servers" | protected-resource metadata의 `authorization_servers`에 이름 붙은 집합 |
| `resource_metadata` | "Where to find the PRM doc" | 401/403에서 RFC 9728 metadata URL을 이름 붙이는 `WWW-Authenticate` parameter |
| Public client | "Native or browser client" | `client_secret`이 없는 OAuth client. PKCE가 보완 |
| `WWW-Authenticate` | "401/403 response header" | client recovery를 drive하는 `Bearer error=...` directive를 운반 |

## 더 읽을거리

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization) — 이 lesson이 구현하는 MCP auth profile
- [MCP blog — One Year of MCP: November 2025 Spec Release](https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/) — 2025-11-25 변경 사항(CIMD, XAA, DCR demotion)
- [Aaron Parecki — Client Registration in the November 2025 MCP Authorization Spec](https://aaronparecki.com/2025/11/25/1/mcp-authorization-spec-update) — CIMD-over-DCR rationale
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR(fallback path)
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public-client proof-of-possession
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience pinning
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — mix-up attack을 방어하는 `iss` parameter
- [OAuth 2.1 draft](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — 통합 OAuth substrate
