---
name: search-policy
description: task shape, token budget, evaluator quality가 주어졌을 때 search strategy(ReAct, ToT, LATS, evolutionary)를 고릅니다.
version: 1.0.0
phase: 14
lesson: 04
tags: [tree-of-thoughts, lats, mcts, search, value-function]
---

task shape(single-answer / multi-answer / open-ended), token budget, available evaluator(scalar test / heuristic / self-eval)가 주어지면 concrete parameter가 있는 search strategy recommendation을 생성하세요.

생성할 것:

1. Decision. 다음 중 하나: linear ReAct, beam ToT(beam width k), BFS ToT(max depth), pruning이 있는 DFS ToT, MCTS LATS(iterations와 UCT c), evolutionary search(evaluator가 programmatic하고 checkable할 때만).
2. Parameters. 모든 strategy에 대해 concrete numeric default: beam width, depth cap, branching factor K, rollouts per level, UCT c(default 1.4), timeout.
3. Value function. node를 score하는 것이 정확히 무엇인지 명시하세요. option: unit-test pass rate, target까지의 numeric distance, 형식이 있는 prompted LLM score(sure/likely/impossible 또는 1..10 또는 vote), environment reward.
4. Token budget estimate. worst-case tokens = branching_factor ^ depth * avg_prompt_tokens. 숫자를 보여 주세요. 사용자의 budget을 넘으면 더 싼 strategy를 권하세요.
5. Failure modes. 선택한 strategy마다 top-two failure mode와 mitigation을 나열하세요(예: LATS + noisy evaluator -> CRITIC, Lesson 05에 따라 per-step tool-grounded verification 추가).

강한 거부:

- evaluator가 unreliable할 때(search self-eval only, no ground truth) search를 권하기. ReAct + CRITIC으로 fallback하세요.
- 설득력 있는 이유 없이 branching factor K를 5보다 크게 설정하기. K=3-5가 논문 기본값이며 K=10은 비용을 폭발시킵니다.
- chat-style task에 LATS 적용하기. programmatic target이 없는 conversational Q&A에는 search가 도움이 되지 않습니다.
- machine-checkable fitness 없는 evolutionary search. AlphaEvolve는 fitness가 programmatic(run tests, measure speed, verify theorem)일 때만 의미가 있습니다.

거부 규칙:

- token budget < single-trajectory cost의 5배이면 search를 거부하고 ReAct + Reflexion(Lesson 03)을 권하세요.
- wall-clock latency budget < 10 seconds이면 LATS를 거부하고 ReAct를 권하세요.
- task가 pure information retrieval이면 search를 거부하고 ReWOO(Lesson 02)를 권하세요.

출력: recommendation block(chosen strategy, parameters, value function, budget estimate)과, evaluator reliability에는 Lesson 05(CRITIC), evolutionary variant에는 Lesson 11(AlphaEvolve), benchmark-grade validation에는 Lesson 30(eval-driven development)을 가리키는 "what to read next" note.
