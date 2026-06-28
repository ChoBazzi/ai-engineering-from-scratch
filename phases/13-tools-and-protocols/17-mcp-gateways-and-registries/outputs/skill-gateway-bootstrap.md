---
name: gateway-bootstrap
description: 사용자, backend, compliance 제약이 주어졌을 때 gateway 구성 명세를 작성합니다.
version: 1.0.0
phase: 13
lesson: 17
tags: [mcp, gateway, rbac, audit, policy]
---

엔터프라이즈 MCP 계획(사용자, backend, compliance 제약)이 주어지면 gateway 구성 명세를 작성하세요.

다음을 산출하세요.

1. Backend 목록. 각 항목에는 registry(Official / Glama / custom), canonical name(reverse-DNS), pinned description hash를 포함하세요.
2. 사용자 목록. 각 사용자의 role과 허용된 tool 집합을 포함하세요.
3. RBAC matrix. 사용자 x backend-tool마다 한 행을 두고 allow/deny를 표시하세요.
4. Rate limits. 사용자별 burst 및 sustained limit, 비용이 큰 도구의 도구별 limit를 포함하세요.
5. Audit 계획. 로그 목적지(file, OpenTelemetry, SIEM), retention, 캡처할 필드를 포함하세요.

강한 거부 조건:
- 명시적 관리자 승인 없이 Official Registry에 없는 backend.
- 모든 사용자에게 모든 도구를 허용하는 RBAC 규칙. 권한 폭증입니다.
- immutable storage가 없는 audit 계획. compliance 실패입니다.

거부 규칙:
- 개발자 수가 100명을 넘는데 role이 정의되지 않았다면 bootstrap을 거절하고 최소 세 가지 role을 요구하세요.
- 계획에 OAuth 2.1 identity provider가 지정되지 않았다면 거절하고 먼저 Keycloak 또는 Auth0 도입을 권장하세요.
- backend 중 하나라도 stdio를 사용하면 HTTP gateway를 통한 proxy를 거절하세요. stdio 서버는 개발자별로 로컬에서 실행됩니다.

산출물: backend 목록, 사용자 목록, RBAC matrix, rate limits, audit 계획을 담은 한 페이지 구성 문서. 마지막에는 팀이 가장 먼저 구현해야 할 단일 policy 규칙을 적으세요.
