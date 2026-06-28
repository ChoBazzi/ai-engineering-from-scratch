# Tree of Thoughts and LATS: Deliberate Search

> 단일 chain-of-thought trajectory에는 backtrack할 여지가 없습니다. ToT(Yao et al., 2023)는 reasoning을 각 node의 self-evaluation이 있는 tree로 바꿉니다. LATS(Zhou et al., 2024)는 ToT를 ReAct 및 Reflexion과 함께 Monte Carlo Tree Search 아래 통합합니다. Game of 24는 4%(CoT)에서 74%(ToT)로 올라가고, LATS는 HumanEval에서 92.7% pass@1을 달성합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 minutes

## Learning Objectives

- reasoning을 search로 frame합니다. node는 "thought", edge는 "expansion", value는 "얼마나 promising한가"입니다.
- self-evaluation scoring이 있는 stdlib ToT-style BFS tree search를 구현합니다.
- select / expand / simulate / backpropagate를 갖춘 toy LATS MCTS loop로 확장합니다.
- search가 token multiplier를 감수할 가치가 있는 경우(Game of 24, code generation)와 단일 trajectory로 충분한 경우(simple Q&A)를 결정합니다.

## The Problem

Chain-of-thought는 선형 보행입니다. 첫 step이 틀리면 이후 모든 step은 나쁜 전제 위에서 작동합니다. Game of 24(네 digit에 + - * /를 써서 24 만들기)에서 GPT-4 CoT는 4% accuracy에 그칩니다. 모델은 초기에 wrong subexpression을 고르고 회복하지 못합니다.

reasoning에 필요한 것은 여러 candidate를 제안하고, 평가하고, promising한 것을 고르고, dead end가 나타나면 backtrack하는 능력입니다. 그것이 search입니다. Tree of Thoughts와 LATS는 두 가지 표준 formulation입니다.

## The Concept

### Tree of Thoughts (Yao et al., NeurIPS 2023)

각 node는 coherent intermediate step("thought")입니다. 각 node는 K개의 child thought로 expand될 수 있습니다. LLM은 scoring prompt로 각 node를 self-evaluate합니다. search는 tree를 탐색합니다. BFS, DFS, beam이 가능합니다.

```text
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Self-evaluation이 핵심입니다. 논문은 세 가지 variant를 보여 줍니다. `sure / likely / impossible` classification, `1..10` numeric score, candidate 간 vote입니다. 세 방식 모두 Game of 24에서 CoT를 크게 이깁니다(GPT-4 기준 4% -> 74%).

### LATS (Zhou et al., ICML 2024)

LATS는 ToT, ReAct, Reflexion을 MCTS 아래 통합합니다. LLM은 세 역할을 합니다.

- **Policy**: candidate next action을 제안(ReAct-style).
- **Value function**: partial trajectory를 score(ToT-style self-eval).
- **Self-reflector**: 실패 시 natural-language reflection을 쓰고(Reflexion-style), future rollout을 reseed하는 데 사용.

Environment feedback(observation)은 value function에 섞여 search가 model opinion뿐 아니라 실제 tool result로 informed되게 합니다. 논문 당시 결과는 GPT-4로 HumanEval pass@1 92.7%(SOTA), GPT-3.5로 WebShop average 75.9(gradient-based fine-tuning에 근접)입니다.

### MCTS, 최소 버전

iteration마다 네 phase가 있습니다.

1. **Select** - UCT(upper confidence bound for trees)로 root에서 leaf까지 이동.
2. **Expand** - policy로 K개 child 생성.
3. **Simulate** - child에서 policy로 rollout하고 value function(또는 environment reward)으로 leaf score.
4. **Backpropagate** - path 위로 visit count와 value estimate update.

UCT formula: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. 첫 항은 exploitation, 두 번째 항은 exploration입니다. `c`는 task별로 tune합니다.

### 비용의 현실

search는 token을 폭발시킵니다. Game of 24의 ToT는 CoT 대비 100-1000배 token을 씁니다. LATS도 비슷합니다. 공짜가 아닙니다. search는 다음 경우에 남겨 두세요.

- 단일 trajectory가 명백히 불충분한 task(Game of 24, complex code).
- wall-clock보다 correctness가 중요한 task.
- 값싸고 reliable한 value function이 있는 task(code의 unit test, math의 explicit target).

task에 단일 정답이 있고 evaluator가 noisy하다면 search는 보통 더 나쁘게 만듭니다. "좋은 score를 받은" wrong answer를 찾아내기 때문입니다.

### 2026년 위치

대부분 production agent는 LATS를 실행하지 않습니다. ReAct와 tool-grounded verification(CRITIC, Lesson 05)을 실행합니다. search는 specialized niche에 나타납니다.

- test를 value function으로 실행하는 coding agent(HumanEval-style).
- 여러 query path를 탐색하는 deep-research agent.
- LangGraph subgraph 안의 planning-heavy workflow.

AlphaEvolve(Lesson 11)는 2025년의 극단입니다. code에 대한 evolutionary search, machine-checkable fitness, frontier gain(56년 만의 첫 4x4 matmul improvement)입니다.

## Build It

`code/main.py`는 다음을 구현합니다.

- stylized "pick arithmetic ops" task 위의 tiny ToT BFS.
- 같은 task 위의 toy LATS MCTS loop(Select / Expand / Simulate / Backpropagate), UCT selection 포함.
- symbolic score와 self-eval score를 조합하는 value function.

실행:

```bash
python3 code/main.py
```

trace는 ToT가 BFS로 node마다 세 candidate를 expand하는 모습과, LATS가 MCTS로 best rollout에 수렴하는 모습을 비교해 보여 줍니다. 두 방식의 token count도 출력됩니다.

## Use It

LangGraph는 ToT-style exploration을 subgraph pattern으로 제공합니다. LangChain 팀의 LATS blog(May 2024)가 reference tutorial입니다. LlamaIndex는 `TreeOfThoughts` agent를 제공합니다. 2026년 대부분 production agent에서 이 pattern은 `if task_complexity > threshold: use_search()` gate 뒤에 있습니다. Lesson 05의 evaluator-optimizer pattern을 보세요.

## Ship It

`outputs/skill-search-policy.md`는 task shape, budget, evaluator fidelity가 주어졌을 때 linear ReAct, ToT, LATS, evolutionary search 중 하나를 선택합니다.

## Exercises

1. toy LATS를 UCT c=0.1과 c=2.0으로 실행하세요. trace에서 무엇이 달라지나요?
2. value function을 더 noisy한 scorer로 바꾸세요(random jitter 추가). MCTS가 여전히 best leaf를 찾나요? 견딜 수 있는 최소 signal-to-noise는 얼마인가요?
3. beam-search ToT를 구현하세요(each level에서 top-k 유지). BFS와 비교하세요. tight token budget에서는 어느 쪽이 나은가요?
4. LATS Section 5.1을 읽으세요. HumanEval trajectory count를 재현하세요. 보고된 pass@1에 도달하려면 rollout이 몇 개 필요한가요?
5. LATS paper의 "when LATS helps less" 논의를 읽으세요. task shape를 search strategy로 mapping하는 한 단락짜리 decision rule을 쓰세요.

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Tree of Thoughts | "Branching CoT" | Yao et al. - self-evaluation이 있는 thought node의 tree |
| LATS | "MCTS for LLMs" | Zhou et al. - ToT + ReAct + Reflexion을 MCTS 아래 통합 |
| UCT | "Upper confidence bound" | exploitation(Q)과 exploration(ln N / n)을 balance하는 select formula |
| Value function | "How good is this state" | Prompted LLM score 또는 environment reward이며 backprop에 feed |
| Policy | "Action proposer" | candidate next thought/action을 내는 ReAct-style generator |
| Rollout | "Simulated trajectory" | policy로 node에서 leaf까지 이동하고 value로 score |
| Backpropagate | "Update ancestors" | leaf reward를 path 위로 밀어 visit count와 Q update |
| Search cost | "Token explosion" | Game of 24에서 CoT 대비 100-1000배. 도입 전 budget 산정 필요 |

## Further Reading

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) - 표준 논문
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) - Reflexion feedback이 있는 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - search를 위한 subgraph pattern
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) - programmatic evaluator가 있는 evolutionary search
