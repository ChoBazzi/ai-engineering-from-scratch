---
name: debate
description: N debater, R round, configurable topology(full mesh, star, ring), convergence rule을 가진 multi-agent debate를 scaffold한다.
version: 1.0.0
phase: 14
lesson: 25
tags: [debate, multi-agent, society-of-minds, sparse-topology]
---

question class와 accuracy target이 주어지면 debate protocol을 scaffold한다.

생성할 것:

1. homogenization을 피하기 위해 서로 다른 prompt(가능하면 서로 다른 model)를 가진 `Debater`.
2. round runner: full mesh, star, 또는 ring topology.
3. convergence rule: majority-vote, confidence weighted, 또는 supermajority-with-fallback.
4. round 1 forced disagreement: 가능하면 모든 debater가 서로 다른 proposal을 반환한다.
5. cost accounting: 질문별 total critique ops + token cost.

강한 거부 조건:

- 모든 debater가 같은 prompt와 같은 model을 사용함. groupthink가 보장된다.
- cost 확인 없이 N >= 6으로 full mesh 사용. Debate ops는 O(N*R)로 scale한다.
- convergence rule 없음. debater 0의 round-R 답을 반환하는 것은 convergence가 아니다.

거부 규칙:

- 제품이 latency-sensitive(<1s budget)이면 debate를 거부한다. 대신 Self-Refine(Lesson 05) 또는 parallel voting(Lesson 12)을 사용한다.
- question class가 simple factual lookup(capital, date, definition)이면 debate를 거부한다. Lookup + CRITIC(Lesson 05)이 더 싸다.
- eval set의 어떤 질문에서도 round 1 이후 debater 사이에 disagreement가 없으면 protocol을 거부한다. model/prompt diversity가 필요하다.

출력: N/R choice, topology rationale, eval set의 cost-vs-accuracy measurement를 설명하는 `debater.py`, `topology.py`, `convergence.py`, `runner.py`, `README.md`. task가 더 단순하면 Lesson 12(workflow patterns), 더 큰 system에 debate를 embedding하려면 Lesson 28(orchestration patterns)을 가리키는 "what to read next"로 끝낸다.
