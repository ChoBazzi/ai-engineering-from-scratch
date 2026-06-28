---
name: ecosystem-blueprint
description: product need가 주어지면 primitive, security posture, telemetry, packaging을 포함한 전체 Phase 13 ecosystem architecture를 작성합니다.
version: 1.0.0
phase: 13
lesson: 22
tags: [mcp, capstone, ecosystem, architecture, a2a, otel]
---

product need(research, summarization, automation, 기타 agent-driven workflow)가 주어지면 전체 architecture를 작성하세요.

다음을 산출하세요.

1. MCP primitives. 어떤 tools, resources, prompts, tasks가 필요한지 정하세요. `ui://` app이 있나요? async task가 있나요?
2. Security posture. OAuth 2.1 scope set, gateway RBAC matrix, pinned hash manifest, Rule of Two audit.
3. A2A collaboration. sub-agent call이 있는지 식별하고 Agent Card를 정의하세요.
4. Telemetry. OTel GenAI span hierarchy. Exporter 및 backend 선택.
5. Packaging. AGENTS.md, SKILL.md, deployment surface(Docker Compose, K8s).
6. Phase 13 lesson과의 매핑. 각 design choice가 어느 lesson으로 거슬러 올라가는지 적으세요.

강한 거부 조건:
- single turn에서 untrusted input, sensitive data, consequential action을 결합하는 architecture(Rule of Two 위반).
- MCP 및 A2A hop을 가로지르는 trace propagation이 없는 architecture.
- LLM layer에 fallback provider가 최소 하나도 없는 architecture.

거부 규칙:
- product need가 direct LLM call로 더 잘 해결된다면 full ecosystem scaffold를 거절하세요.
- 팀에 gateway를 운영할 SRE가 없다면 managed gateway(Cloudflare MCP Portals, Portkey)를 권장하세요.
- architecture에 payment가 포함된다면 AP2를 drift risk가 있는 A2A extension으로 표시하고 별도 signoff를 권장하세요.

산출물: primitives, security posture, A2A hops, telemetry plan, packaging, lesson map을 담은 한 페이지 blueprint. 마지막에는 이 배포에서 가장 어려운 단일 operational risk를 식별하는 한 문장을 적으세요.
