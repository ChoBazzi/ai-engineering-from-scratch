---
name: parallel-inference-router
description: reasoning workload를 voting, tree-of-thought, multi-agent, Hogwild!, speculative decoding 전략 사이에서 라우팅합니다.
version: 1.0.0
phase: 10
lesson: 22
tags: [parallel-inference, hogwild, speculative-decoding, tree-of-thought, multi-agent, reasoning]
---

reasoning workload profile(task당 token budget, task parallelism characteristics, model family, deployment target, latency budget)이 주어지면 parallel-inference strategy 또는 조합을 추천하세요.

다음을 산출하세요:

1. Task classification. Long reasoning(5k+ tokens), medium chain-of-thought(1k-5k), short chat(1k 미만), classification. 첫 결정을 이끕니다.
2. Parallelism axis. Within-sequence(speculative decoding) vs across-sequence(voting, Hogwild!, multi-agent). 대부분 workload는 within-sequence axis를 먼저 적용할 때 이득을 봅니다.
3. Strategy recommendation. 다음 중 고르세요: speculative decoding only(100 token 이상 workload의 safe default), speculative + Hogwild!(parallelizable structure가 있는 long reasoning), tree-of-thought(명시적 branch-and-prune 문제), multi-agent(role-specialization 문제), voting ensemble(high-stakes classification).
4. Parameter settings. speculative decoding: draft family(EAGLE-3 default)와 `N`(Phase 10 · 15 skill). Hogwild!: worker count N(2-4, 그 이상은 드묾), coordination prompt template, single-node deployment confirmation.
5. Combined speedup estimate. speculative decoding과 Hogwild!를 결합한다면 multiplicative speedup을 보고하세요(일반 범위: 3x spec * 1.5-2x Hogwild! = 4.5-6x).

강한 거부 조건:
- 2000 token 미만 workload에 Hogwild!. Coordination overhead가 지배합니다.
- non-reasoning model에서 Hogwild!(emergent coordination 없음).
- 자연스러운 role decomposition이 없는 문제에 multi-agent framework.
- 명시적 branch-and-prune logic 없는 tree-of-thought(그렇지 않으면 strategy가 linear CoT로 축소됨).
- 노드 간 Hogwild! 실행(cross-node cache synchronization이 너무 느림).

거부 규칙:
- workload가 experimental research라면 Hogwild!를 production bet가 아니라 experiment로 추천하세요. 2026년 4월 기준 speedup은 task-dependent이고 real-world deployment는 드뭅니다.
- 사용자가 guaranteed speedup을 요구하면 거부하고, strong-guarantee property(output distribution preserved)는 speculative decoding에만 있다고 설명하세요. Hogwild!는 empirical입니다.
- 사용자의 VRAM이 제한적이면 Hogwild! N>2를 거부하세요. cache는 공유되지만 각 worker는 자기 activation memory가 필요합니다.

출력: task classification, parallelism axis, strategy, parameters, combined speedup estimate를 나열한 한 페이지 recommendation. 마지막에는 Hogwild!가 첫 100 production request에서 값을 못 할 때 speculative decoding alone으로 되돌릴 구체적 latency 또는 accuracy metric을 명명하는 "rollback trigger" 문단을 붙이세요.
