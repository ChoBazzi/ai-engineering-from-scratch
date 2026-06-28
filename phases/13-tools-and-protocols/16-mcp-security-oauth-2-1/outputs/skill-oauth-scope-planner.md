---
name: oauth-scope-planner
description: 원격 MCP 서버를 위한 OAuth 2.1 scope 집합, pinning 규칙, step-up 정책을 설계합니다.
version: 1.0.0
phase: 13
lesson: 16
tags: [oauth, pkce, resource-indicators, step-up, sep-835]
---

도구 목록이 있는 원격 MCP 서버가 주어지면 authorization 모델을 설계하세요.

다음을 산출하세요.

1. Scope 계층. 단계형 scope 집합(예: `read` -> `write` -> `delete` -> `admin`). 작업 클래스마다 scope 하나를 두고, scope 집합을 불필요하게 폭증시키지 마세요.
2. Scope-to-tool 매핑. 각 도구에 필요한 scope를 주석으로 표시하세요. scope가 둘 이상 필요한 도구는 별도로 표시하세요.
3. Step-up 정책. 초기 동의가 아니라 step-up이 필요한 작업을 정하세요. 일반적으로 파괴적 작업에는 step-up이 필요합니다.
4. Resource indicator 값. `resource` 매개변수에 쓰는 canonical URL입니다. URL이 `.well-known/oauth-protected-resource`의 resource 필드와 일치하는지 확인하세요.
5. Protected-resource metadata. `authorization_servers`, `scopes_supported`, `resource`를 포함하는 `.well-known/oauth-protected-resource` JSON 초안을 작성하세요.

강한 거부 조건:
- admin scope가 필요하지만 명시적 확인 대화상자 없이 호출되는 도구. step-up이 필요합니다.
- 작업 클래스를 둘 이상 포괄하는 scope. 권한이 서서히 커지는 privilege creep입니다.
- audience validation을 건너뛰는 서버. confused-deputy 취약점입니다.

거부 규칙:
- 서버가 로컬(stdio)이면 OAuth를 거절하고 stdio는 parent trust를 상속한다고 설명하세요.
- 서버가 legacy OAuth 2.0 implicit flow에 의존하면 거절하고 2.1 + PKCE로 마이그레이션하도록 요구하세요.
- 사용자가 passwordless "API key only" auth를 요청하면 원격 서버에서는 거절하세요. 사용자가 authorize하는 접근에는 resource indicators가 포함된 OAuth 2.1 authorization code + PKCE가 필요합니다. Client credentials는 user delegation이 없는 machine-to-machine 시나리오에만 적합합니다.

산출물: scope 계층, scope-to-tool 매핑, step-up 정책, resource indicator, protected-resource metadata JSON을 담은 한 페이지 authorization 계획. 마지막에는 사용자가 처음 마주쳤을 때 가장 놀랄 가능성이 큰 step-up 작업을 적으세요.
