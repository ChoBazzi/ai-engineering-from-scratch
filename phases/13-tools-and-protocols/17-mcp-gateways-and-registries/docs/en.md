# MCP Gateways와 Registries — 엔터프라이즈 Control Plane

> Enterprise는 모든 개발자가 임의의 MCP 서버를 설치하도록 둘 수 없습니다. gateway는 auth, RBAC, audit, rate limiting, caching, tool-poisoning detection을 중앙화한 뒤 병합된 tool surface를 단일 MCP endpoint로 노출합니다. Official MCP Registry(Anthropic + GitHub + PulseMCP + Microsoft, namespace-verified)가 canonical upstream입니다. 이 lesson은 gateway가 어디에 들어가는지 설명하고, 최소 구현을 따라가며, 2026년 vendor landscape를 훑습니다.

**Type:** Learn
**Languages:** Python (stdlib, minimal gateway)
**Prerequisites:** Phase 13 · 15 (tool poisoning), Phase 13 · 16 (OAuth 2.1)
**Time:** ~45 minutes

## 학습 목표

- MCP gateway가 어디에 위치하는지 설명합니다(MCP client와 여러 backend MCP server 사이).
- 다섯 가지 gateway 책임(auth, RBAC, audit, rate limit, policy)을 구현합니다.
- gateway layer에서 pinned-tool-hash manifest를 enforce합니다.
- Official MCP Registry와 metaregistry(Glama, MCPMarket, MCP.so, Smithery, LobeHub)를 구분합니다.

## 문제

Fortune 500 기업에 승인된 MCP 서버 30개, 개발자 5000명, compliance 및 audit 요구사항, 중앙화 policy를 원하는 security team이 있습니다. 모든 개발자가 IDE에 임의 서버를 설치하도록 두는 것은 선택지가 아닙니다.

gateway pattern:

1. Gateway는 개발자가 연결하는 단일 Streamable HTTP endpoint로 실행됩니다.
2. Gateway는 각 backend MCP server의 credential을 보관합니다.
3. 모든 개발자 request는 gateway 자체 OAuth를 통해 authenticate되고 scoped됩니다.
4. Gateway는 policy를 적용하며 call을 backend server로 route합니다.
5. 모든 call은 audit을 위해 logged됩니다.

Cloudflare MCP Portals, Kong AI Gateway, IBM ContextForge, MintMCP, TrueFoundry, Envoy AI Gateway는 모두 2025-2026년에 gateway 또는 gateway feature를 출시했습니다.

그동안 Official MCP Registry가 canonical upstream으로 출시되었습니다. gateway가 pull할 수 있는 curated, namespace-verified, reverse-DNS-named server입니다. Metaregistry(Glama, MCPMarket, MCP.so, Smithery, LobeHub)는 여러 source의 server를 aggregate합니다.

## 개념

### Five gateway responsibilities

1. **Auth.** 개발자를 식별하는 OAuth 2.1. user role로 매핑합니다.
2. **RBAC.** user별 policy: 어떤 server, 어떤 tool, 어떤 scope.
3. **Audit.** 모든 call을 누가, 무엇을, 언제, 어떤 결과로 호출했는지 기록합니다.
4. **Rate limit.** abuse를 막기 위한 user별 / tool별 / server별 cap.
5. **Policy.** poisoned description 거절, Rule of Two enforce, PII redact.

### Gateway as a single endpoint

개발자에게 gateway는 하나의 MCP server처럼 보입니다. 내부적으로는 N개의 backend로 route합니다. Session id(Phase 13 · 09)는 boundary에서 rewrite됩니다.

### Credential vaulting

개발자는 backend token을 보지 못합니다. gateway가 이를 보관하거나, 보관하는 identity provider로 proxy합니다. gateway에서 `notes:read`를 가진 개발자는 gateway 자체 backend credential로 notes MCP server에 transitively 접근할 수 있지만, transitive access를 bind하는 policy 아래에서만 가능합니다.

### Tool-hash pinning at the gateway

Gateway는 승인된 tool description의 manifest(SHA256 hash)를 보관합니다. discovery 시 각 backend의 `tools/list`를 fetch하고 hash를 manifest와 비교한 뒤, description이 mutate된 tool은 제거합니다. Phase 13 · 15의 rug-pull 방어를 중앙에 적용한 것입니다.

### Policy-as-code

고급 gateway는 OPA/Rego, Kyverno, Styra로 policy를 표현합니다. "user `alice`는 org `acme`의 repo에서만 `github.open_pr`를 호출할 수 있다" 같은 rule을 선언적으로 encode합니다. 단순 gateway는 hand-coded Python을 씁니다. 두 형태 모두 유효합니다.

### Session-aware routing

사용자 session에 server가 섞여 있을 때 gateway는 multiplex합니다. 개발자의 단일 MCP session은 server마다 하나씩 N개의 backend session을 갖습니다. 어떤 backend의 notification이든 gateway를 통해 개발자의 session으로 route됩니다.

### Namespace merging

Gateway는 모든 backend의 tool namespace를 병합하며, 보통 collision 시 prefix를 붙입니다. `github.open_pr`, `notes.search`. 이렇게 하면 routing이 모호하지 않습니다.

### Registries

- **Official MCP Registry (`registry.modelcontextprotocol.io`).** Anthropic, GitHub, PulseMCP, Microsoft stewardship 아래 출시되었습니다. Namespace-verified(reverse-DNS: `io.github.user/server`)이며 기본 품질이 pre-filter됩니다.
- **Glama.** 여러 source를 aggregate하는 search-centric metaregistry.
- **MCPMarket.** vendor listing이 있는 commercial-leaning directory.
- **MCP.so.** community directory. open submission.
- **Smithery.** package-manager-style installation flow.
- **LobeHub.** LobeChat app에 UI-integrated registry를 제공합니다.

Enterprise gateway는 기본적으로 Official Registry에서 pull하고, admin-curated metaregistry 추가를 허용하며, pinned되지 않은 것은 모두 거절합니다.

### Reverse-DNS naming

Official Registry는 public server에 reverse-DNS name을 요구합니다. 예: `io.github.alice/notes`. Namespace는 squatting을 방지하고 trust delegation을 더 명확하게 만듭니다.

### Vendor survey, April 2026

| 벤더 | 강점 |
|--------|----------|
| Cloudflare MCP Portals | edge-hosted, OAuth 통합, 무료 tier |
| Kong AI Gateway | K8s-native, 세밀한 policy, OpenTelemetry로 log 전송 |
| IBM ContextForge | enterprise IAM, compliance, audit export |
| TrueFoundry | DevOps 지향, metric 우선 |
| MintMCP | developer-platform 지향 |
| Envoy AI Gateway | open-source, 맞춤형 filter |

Phase 17(production infrastructure)은 gateway operation을 더 깊게 다룹니다.

## 사용하기

`code/main.py`는 약 150줄의 minimal gateway를 제공합니다. fake Bearer token으로 user를 authenticate하고, user별 RBAC policy를 보관하며, request를 두 backend MCP server로 route하고, 모든 call을 audit log에 기록하고, rate limit을 enforce하며, backend tool description hash가 pinned manifest와 일치하지 않으면 거절합니다.

볼 부분:

- `user_id`를 key로 하고 허용된 `server_tool` entry를 가진 `RBAC` dict.
- `AUDIT_LOG`는 append-only event list입니다.
- Rate limit은 user별 token bucket을 사용합니다.
- Pinned manifest는 `server::tool -> hash` dict입니다.

## 산출물

이 lesson은 `outputs/skill-gateway-bootstrap.md`를 만듭니다. enterprise MCP 계획(사용자, backend, compliance)이 주어지면 이 skill은 gateway configuration spec을 생성합니다.

## 연습 문제

1. `code/main.py`를 실행하세요. 허용된 user로 call한 다음, disallowed user로 call하고, rate-limit-exceeded burst도 실행하세요. 세 flow를 모두 확인하세요.

2. client에 반환하기 전에 result에서 PII를 redact하는 policy를 추가하세요. SSN 형태의 문자열에는 단순 regex pass를 사용하고, gap(email, phone number)을 기록하세요.

3. audit log를 확장해 OpenTelemetry GenAI span을 emit하세요. 정확한 attribute는 Phase 13 · 20에서 다룹니다.

4. backend 다섯 개(notes, github, postgres, jira, slack)가 있는 50명 개발자 팀의 RBAC policy를 설계하세요. 각 backend에서 누가 read-only를 갖나요? 누가 write를 갖나요?

5. Cloudflare enterprise MCP post를 처음부터 끝까지 읽으세요. 이 stdlib gateway가 제공하지 않는 Cloudflare feature 하나를 식별하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Gateway | "MCP proxy" | client와 backend 사이에 있는 중앙화 server |
| Credential vaulting | "Backend tokens stay server-side" | 개발자는 upstream token을 보지 못함 |
| Session-aware routing | "Multi-backend session" | gateway가 개발자 session마다 N개 backend session을 multiplex |
| Tool-hash pinning | "Approved manifest" | 승인된 모든 tool description의 SHA256. 중앙에서 rug-pull을 차단 |
| RBAC | "Per-user policy" | tool과 server에 대한 role-based access control |
| Policy-as-code | "Declarative rules" | gateway에서 enforce되는 OPA/Rego, Kyverno, Styra policy |
| Audit log | "Who, what, when" | compliance를 위한 append-only event log |
| Rate limit | "Per-user token bucket" | abuse를 막기 위한 per-minute cap |
| Official MCP Registry | "Canonical upstream" | `registry.modelcontextprotocol.io`, namespace-verified |
| Reverse-DNS naming | "Registry namespace" | `io.github.user/server` convention |

## 더 읽을거리

- [Official MCP Registry](https://registry.modelcontextprotocol.io/) — canonical upstream, namespace-verified
- [Cloudflare — Enterprise MCP](https://blog.cloudflare.com/enterprise-mcp/) — OAuth와 policy가 있는 gateway pattern
- [agentic-community — MCP gateway registry](https://github.com/agentic-community/mcp-gateway-registry) — open-source reference gateway
- [TrueFoundry — What is an MCP gateway?](https://www.truefoundry.com/blog/what-is-mcp-gateway) — feature comparison article
- [IBM — MCP context forge](https://github.com/IBM/mcp-context-forge) — IBM의 enterprise gateway
