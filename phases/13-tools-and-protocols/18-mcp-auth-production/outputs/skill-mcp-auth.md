---
name: mcp-auth-wiring
description: production MCP authorization(RFC 8414, CIMD, 7591, 8707, 7636 PKCE, 9728, 9207)을 구축합니다. protected-resource metadata, enrollment, JWKS refresh, request별 token validation을 포함합니다.
version: 1.1.0
phase: 13
lesson: 18
tags: [mcp, oauth, cimd, dcr, jwks, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728, rfc9207]
---

MCP 서버 구성과 IdP capability 집합이 주어지면 production MCP authorization layer를 이루는 auth surface와 refusal rule을 산출하세요.

입력:

- `mcp_resource_url` — canonical resource URL(가장 구체적인 식별자. 같은 origin에 함께 호스팅되는 서버를 구분할 때만 path를 유지), `aud` 및 protected-resource metadata의 `resource` 값으로 사용됩니다.
- `idp_metadata_url` — IdP의 `/.well-known/oauth-authorization-server` 또는 OpenID Connect Discovery URL입니다.
- `idp_capabilities` — `code_challenge_methods_supported`, `grant_types_supported`, `client_id_metadata_document_supported`(CIMD), `registration_endpoint`(DCR), `response_types_supported`, `authorization_response_iss_parameter_supported`(RFC 9207)의 관측값입니다.
- `tools` — 각 도구가 요구하는 scope가 포함된 MCP 도구 목록입니다.

다음을 산출하세요.

1. **Refusal gate.** hard condition 중 하나라도 실패하면 wiring을 거절하고 중단하세요.
   - `code_challenge_methods_supported`에 `S256`이 없습니다(PKCE에는 degraded mode가 없습니다).
   - `grant_types_supported`에 `authorization_code`가 없습니다.
   - `response_types_supported`가 정확히 `["code"]`가 아닙니다.
   - enrollment path가 없습니다. pre-registered `client_id`, `client_id_metadata_document_supported: true`(CIMD), `registration_endpoint`(DCR) 중 어느 것도 사용할 수 없습니다. 하나만 있으면 충분합니다. DCR 부재만으로는 더 이상 거절 사유가 아닙니다(2025-11-25에서 DCR은 `MAY`로 낮아졌고 CIMD가 권장 기본값입니다).

2. MCP 서버가 `/.well-known/oauth-protected-resource`에 게시할 **Protected-resource metadata document**(RFC 9728). `resource`, `authorization_servers`(issuer allow-list), `scopes_supported`, `bearer_methods_supported: ["header"]`를 포함합니다.

3. **HTTP endpoints.**
   - `GET /.well-known/oauth-protected-resource` — (2)의 문서를 반환합니다.
   - `POST /mcp`(MCP transport) — 어떤 도구도 dispatch하기 전에 token validation을 실행합니다.
   - (DCR path 전용) `POST /register` — registrar이며, 앞단에 rate-limit check를 둡니다.

4. **Background job + routines.**
   - `jwks_uri`를 다시 fetch해 `{keys, fetched_at}` cache에 넣는 scheduled JWKS refresh. Idempotent해야 하며 key를 mint하지 않습니다. AS가 rotate하고 resource server는 refresh만 합니다. 기본값은 `0 */6 * * *`이고, rotation이 잦은 IdP에는 `*/15 * * * *`로 강화하세요.
   - `validate` routine — `iss` allow-list, cached JWKS 대비 signature, `aud == mcp_resource_url`, `exp`, required scope를 확인합니다.
   - step-up issuance path — 사용자가 처음 grant하지 않은 scope 뒤에 걸린 작업이 도구 목록에 있을 때만 필요합니다.

5. **Cache plan.** 허용된 issuer마다 `issuer`를 key로 하는 entry 하나를 두고 `{keys, fetched_at}`를 보관합니다. read pattern을 문서화하세요. validator는 cache를 읽고 `kid` miss가 나면 단 한 번의 synchronous refresh로 fall back합니다(rotate가 아니라 re-fetch입니다. re-fetch는 idempotent하며 key-creation DoS로 바뀔 수 없습니다).

6. **Scope mapping.** 모든 도구를 필요한 scope에 매핑하세요. 다음 table을 출력하세요.
   `| tool | required_scope | rationale |`. Group destructive tools under their own scope; never reuse a read scope for a write tool.

7. **Runtime 거부 규칙**(validator가 반드시 encode해야 함):
   - `aud != mcp_resource_url`이면 reject → 401 `Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="<prm_url>"`.
   - `iss not in authorization_servers`이면 reject.
   - 단 한 번의 re-fetch fall-back 뒤에도 `kid`가 cached JWKS에 없으면 reject.
   - required scope가 없으면 reject → 403 `Bearer error="insufficient_scope", scope="<required>", resource_metadata="<prm_url>"`.
   - `code_verifier` 또는 `resource` parameter가 없는 token request는 reject.

강한 거부 조건(다음은 절대 wire하지 말고 요청을 거절한 뒤 이유를 문서화하세요):

- `client_secret`를 plaintext로 저장. Public client는 `token_endpoint_auth_method: none`을 사용하고 confidential client는 `private_key_jwt`를 사용합니다. 저장 상태나 registration response log에 plaintext shared secret을 남기지 마세요.
- validator에서 `aud` check를 건너뜀. Audience binding(access-token privilege restriction)은 RFC 8707 + RFC 9728의 핵심 이유입니다.
- JWKS cache-miss fall-back을 re-fetch가 아니라 rotate-and-mint에 연결. 누락된 `kid`를 절대 만들어내지 못하며, attacker-controlled `kid` 값이 무제한 key creation을 강제하게 됩니다. fall-back은 idempotent refresh여야 합니다.
- PKCE 없는 authorization code request 허용. OAuth 2.1은 이를 금지합니다. 저장된 authorization-code record에 `code_challenge`가 없는 `/token` exchange는 validator가 reject해야 합니다.
- refresh job 없이 JWKS caching. scheduled refresh가 함께 배포되거나, auth surface를 배포하지 않아야 합니다.
- allow-list 없이 `iss` claim 신뢰. 어떤 `iss`의 token이든 받아들이는 validator는 공격자가 자기 IdP를 세워 token을 위조하게 해 줍니다.
- inbound MCP token을 upstream API로 forwarding(token passthrough). MCP 서버가 upstream API를 호출한다면 반드시 별도의 token을 받아야 합니다. passthrough는 confused-deputy 문제를 만듭니다.
- `registration_access_token`을 plaintext로 저장. Hash-at-rest를 적용하고 모든 update마다 cleartext 제시를 요구하세요.

산출물: protected-resource document, 선택한 enrollment path(CIMD / pre-registration / DCR), HTTP endpoints, JWKS refresh job, cache plan, scope mapping table, encode된 runtime refusal rules를 담은 한 페이지 계획. 마지막에는 선택한 IdP에서 나타날 가능성이 가장 큰 단일 deployment-blocking gap을 적으세요. 일반적으로 CIMD 지원 여부이며, enterprise SSO에서는 DCR availability로 fall back합니다.
