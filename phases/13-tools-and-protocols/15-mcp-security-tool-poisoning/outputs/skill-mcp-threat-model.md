---
name: mcp-threat-model
description: MCP deployment에 대해 적용 가능한 attack classes, 현재 방어, Rule-of-Two 위반을 이름 붙이는 threat model을 만듭니다.
version: 1.0.0
phase: 13
lesson: 15
tags: [mcp, security, tool-poisoning, threat-model, rule-of-two]
---

MCP deployment(서버 목록, 도구 목록, 권한 목록)가 주어지면 threat model을 만드세요.

다음을 산출하세요.

1. Attack applicability. 일곱 공격 클래스(tool poisoning, rug pull, shadowing, MPMA, parasitic toolchain, sampling attacks, supply-chain masquerade) 각각에 대해 적용 가능성을 high / medium / low로 평가하고 한 문장 근거를 쓰세요.
2. Defense inventory. 이미 적용된 방어(hash pinning, static detector, gateway, signed registry, MELON, Rule-of-Two enforcement)를 나열하세요.
3. Rule of Two audit. 모든 도구를 untrusted / sensitive / consequential로 분류하고, 한 turn에서 셋 모두가 결합되는 조합을 flag하세요.
4. Missing defenses. threat profile을 고려할 때 아직 적용되지 않은 가장 leverage가 큰 방어를 이름 붙이세요.
5. Runbook. 팀이 다음 주에 보안 태세를 개선하기 위해 취해야 할 세 가지 action.

강한 거부 조건:
- "우리는 이 서버를 신뢰하므로 공격 클래스 X는 적용되지 않는다"고 말하는 모든 threat model. 서버 하나는 침해된다고 가정하세요.
- silent-overwrite namespace resolution을 사용하는 모든 deployment.
- sampling이 활성화되어 있지만 per-session rate limiter가 없는 모든 deployment.

거부 규칙:
- deployment에 승인된 도구 description 문서가 없다면 거부하고 hash pinning을 먼저 의무화하세요.
- deployment가 public unsigned MCP registries를 사용한다면 supply-chain risk를 표시하고 verified registry로 migration을 권장하세요.
- 어떤 도구든 untrusted input, sensitive data, consequential action을 결합한다면 승인을 거부하고 split을 요구하세요.

출력: attack applicability table, defense inventory, Rule-of-Two flag list, 세 가지 action runbook이 담긴 한 페이지 threat model. 이 deployment에 가장 가치가 높은 단일 security addition으로 끝내세요.
