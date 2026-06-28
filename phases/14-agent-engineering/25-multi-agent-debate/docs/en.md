# Multi-Agent Debate와 Collaboration

> Du et al.(ICML 2024, "Society of Minds")은 N개의 model instance가 독립적으로 답을 제안한 뒤 R round 동안 서로를 반복적으로 critique해 수렴하게 한다. factuality, rule-following, reasoning을 개선한다. Sparse topology는 token cost에서 full mesh를 이긴다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 minutes

## Learning Objectives

- debate protocol을 설명한다: N proposer, R round, 공유 답안으로 수렴.
- debate가 factuality, rule-following, reasoning을 개선하는 이유를 설명한다.
- sparse topology를 설명한다. 모든 debater가 모든 상대를 볼 필요는 없다.
- full-mesh와 sparse variant를 가진 scripted LLM 기반 stdlib debate를 구현하고 token cost와 accuracy를 측정한다.

## The Problem

Self-Refine(Lesson 05)은 한 모델이 자기 자신을 critique하는 방식이라 groupthink 위험이 있다. CRITIC(Lesson 05)은 critique를 external tool에 grounding하지만 항상 가능하지는 않다. Debate는 세 번째 mode를 도입한다. 여러 instance, cross-critique, disagreement를 통한 convergence다.

## The Concept

### Society of Minds (Du et al., ICML 2024)

- N개의 model instance가 같은 질문에 대해 독립적으로 답을 제안한다.
- R round 동안 각 model은 다른 model의 proposal을 읽고 critique한다.
- model은 critique를 바탕으로 답을 update한다.
- R round 후 convergent answer를 반환한다.

원래 실험은 비용 때문에 N=3, R=2를 사용했다. 어려운 문제(MMLU, GSM8K, Chess Move Validity, biography generation)에서는 agent와 round가 많을수록 accuracy가 개선된다.

cross-model 조합은 single-model debate를 이긴다. ChatGPT + Bard together > either alone.

### Sparse topology

"Improving Multi-Agent Debate with Sparse Communication Topology"(arXiv:2406.11776, 2024-2025)는 full-mesh debate가 항상 최적은 아님을 보였다. Sparse topology(star, ring, hub-and-spoke)는 더 낮은 token cost로 accuracy를 맞출 수 있다. 각 debater는 peer의 subset만 본다.

의미:

- Full mesh N=5, R=3 = 5 × 3 = 15 proposals, 각자가 peer 4명을 읽음 = 60 critique ops.
- Star N=5, R=3(허브 1개 + spoke 4개) = 15 proposals, spoke는 hub만 읽음 = 12 critique ops.

### When debate helps

- **Factuality.** N개의 독립 proposal과 cross-check가 hallucination을 줄인다.
- **Rule-following.** Chess move validity — 한 model이 rule을 놓치면 다른 model이 잡는다.
- **Open-ended reasoning.** 여러 framing이 올바른 답으로 좁혀 간다.

### When debate hurts

- **Latency-sensitive UX.** N × R serial round는 감당하기 어려운 latency다.
- **Cost-sensitive scale.** 질문마다 N × R token이 든다.
- **Simple factual lookups.** 한 번의 lookup이 다섯 번의 debate보다 싸다.

### 2026 practical instantiations

- **Anthropic orchestrator-workers**(Lesson 12) — synthesis step이 있는 debate의 한 variant.
- **LangGraph supervisor**(Lesson 13) — central router + specialist agent가 debate를 node로 구현할 수 있다.
- **OpenAI Agents SDK**(Lesson 16) — agent가 iterative critique를 위해 서로 handoff한다.
- **Multi-agent evals** — debate + evaluator-optimizer를 짝지어 eval signal을 만든다.

### Where this pattern goes wrong

- **Convergence collapse.** 모든 agent가 첫 번째 오답으로 수렴한다. 필수 disagreement round로 완화한다.
- **Hub failure.** star topology에서 나쁜 hub는 모두를 오염시킨다. hub를 rotate하거나 여러 hub를 사용한다.
- **Prompt homogenization.** 모든 agent가 같은 prompt를 사용해 같은 답을 낸다. 다양한 prompt 및/또는 model을 사용한다.

## Build It

`code/main.py`는 stdlib debate를 구현한다.

- `Debater` class(debater별 opinion drift가 있는 scripted LLM).
- `FullMeshDebate`와 `SparseDebate` runner.
- 세 질문: factual 하나, rule-based 하나, reasoning 하나.
- metric: convergent answer, rounds to convergence, total critique ops.

실행:

```bash
python3 code/main.py
```

출력: protocol별 accuracy와 cost. sparse는 더 낮은 cost로 3개 중 2개 질문에서 full mesh와 맞먹는다.

## Use It

- **Anthropic orchestrator-workers** — 단순한 2-3 worker debate에 사용한다.
- **LangGraph** — checkpointing이 있는 stateful multi-round debate에 사용한다.
- **Custom** — research 또는 특수한 correctness guarantee에 사용한다.

## Ship It

`outputs/skill-debate.md`는 configurable topology, N, R, convergence rule을 가진 multi-agent debate를 scaffold한다.

## Exercises

1. "forced disagreement" rule을 구현한다. round 1에서 모든 debater는 가능하면 서로 다른 proposal을 내야 한다. convergence speed에 미치는 영향을 측정한다.
2. confidence-weighted aggregation을 추가한다. debater는 (answer, confidence)를 반환하고 aggregator는 confidence로 가중한다. 도움이 되는가?
3. "agent" 하나를 의견이 다른 scripted LLM으로 교체한다. heterogeneity가 accuracy를 개선하는가?
4. 자신의 세 질문에서 full mesh vs sparse의 token cost를 측정한다. cost vs accuracy를 plot한다.
5. Society of Minds 논문을 읽는다. toy를 N=5, R=3으로 port한다. 무엇이 깨지는가? 무엇이 나아지는가?

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Debate | "Multi-agent critique" | N proposer, R round cross-critique, convergence |
| Full mesh | "Everyone reads everyone" | 모든 debater가 매 round마다 모든 peer를 읽음 |
| Sparse topology | "Limited peer view" | debater가 peer의 subset만 읽음 |
| Hub-and-spoke | "Star topology" | 중앙 debater 하나, N-1 spoke는 hub만 읽음 |
| Convergence | "Agreement" | debater가 공유 답안으로 수렴 |
| Society of Minds | "Du et al. debate paper" | ICML 2024 multi-agent debate method |

## Further Reading

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — canonical multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — sparse topology result
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — debate variant로서의 orchestrator-workers
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — single-model self-critique counterpart
