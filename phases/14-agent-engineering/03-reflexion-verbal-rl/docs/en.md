# Reflexion: Verbal Reinforcement Learning

> Gradient-based RL은 failure mode 하나를 고치려면 수천 번의 trial과 GPU cluster가 필요합니다. Reflexion(Shinn et al., NeurIPS 2023)은 이를 자연어로 처리합니다. 실패한 trial 뒤 agent가 reflection을 쓰고, episodic memory에 저장하고, 다음 trial을 그 memory에 condition합니다. 이는 Letta의 sleep-time compute, Claude Code의 CLAUDE.md learnings, pro-workflow의 learn-rule 뒤에 있는 pattern입니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## Learning Objectives

- Reflexion의 세 component(Actor, Evaluator, Self-Reflector)와 episodic memory의 역할을 이름 붙입니다.
- binary evaluator, reflection buffer, fresh re-attempt를 갖춘 stdlib Reflexion loop를 구현합니다.
- 주어진 task에 대해 scalar, heuristic, self-evaluated feedback source 중 무엇을 쓸지 고릅니다.
- verbal reinforcement가 gradient-based RL이라면 수천 trial이 필요했을 error를 잡는 이유를 설명합니다.

## The Problem

agent가 task에 실패했습니다. 표준 RL에서는 trial을 수천 번 더 실행하고, gradient를 계산하고, weight를 update합니다. 비싸고 느리며, 대부분 production agent에는 failure마다 training budget이 없습니다.

Reflexion(Shinn et al., arXiv:2303.11366)은 다른 질문을 던집니다. agent가 왜 실패했는지 생각하고 그 생각을 prompt에 넣어 다시 시도하면 어떨까요? weight update 없음. gradient 없음. trial 사이에 저장되는 자연어만 있습니다.

결과적으로 ALFWorld에서 ReAct와 다른 non-fine-tuned baseline을 이깁니다. HotpotQA에서도 ReAct보다 향상됩니다. code generation(HumanEval/MBPP)에서는 당시 state of the art를 세웠습니다. 단 한 번의 gradient step도 없이 말입니다.

## The Concept

### 세 component

```text
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory - binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

여기에 data structure 하나가 더 있습니다.

```text
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

trial 하나는 Actor를 실행합니다. Evaluator가 score를 매깁니다. score가 낮으면 Self-Reflector가 reflection을 생성합니다("질문이 X를 묻는다고 오해했지만 실제로는 Y를 묻고 있었기 때문에 wrong tool을 골랐다"). reflection은 episodic memory로 들어갑니다. 다음 trial은 새로 시작하지만 reflection을 봅니다.

### 세 evaluator type

1. **Scalar** - 외부 binary signal. ALFWorld는 succeed 또는 fail입니다. HumanEval test는 pass 또는 fail입니다. 가장 단순하고 signal이 가장 강합니다.
2. **Heuristic** - 미리 정의한 failure signature. "agent가 같은 action을 두 번 연속 냈다면 stuck으로 mark." "trajectory가 50 step을 넘으면 inefficient로 mark."
3. **Self-evaluated** - LLM이 자기 trajectory를 score합니다. ground truth가 없을 때 필요합니다. signal은 약하므로 tool-grounded verification(Lesson 05 - CRITIC)과 잘 맞습니다.

2026년 기본값은 혼합입니다. 가능하면 scalar, 아니면 self-eval, safety rail로 heuristic을 사용합니다.

### 왜 generalize되는가

Reflexion은 새로운 algorithm이라기보다 이름 붙은 pattern입니다. 거의 모든 production "self-healing" agent가 어떤 변형을 실행합니다.

- Letta의 sleep-time compute(Lesson 08): 별도 agent가 past conversation을 reflect하고 memory block에 씁니다.
- Claude Code의 `CLAUDE.md` / "save memory" pattern: reflection이 learning으로 capture되어 future session 앞에 붙습니다.
- pro-workflow의 `/learn-rule` command: correction이 explicit rule로 capture됩니다.
- LangGraph의 reflection node: output을 score하고 필요하면 refine으로 route하는 node.

모두 같은 insight에서 나옵니다. 자연어는 "실패에서 배운 것"을 run 사이에 전달하기에 충분히 풍부한 medium입니다.

### 작동하는 경우와 작동하지 않는 경우

Reflexion이 작동하는 경우:

- 명확한 failure signal이 있음(test failure, tool error, wrong answer).
- task class가 재현 가능함(같은 유형의 question이 다시 나올 수 있음).
- reflection이 trajectory를 개선할 여지가 있음(action budget이 충분함).

Reflexion이 도움이 되지 않는 경우:

- agent가 첫 시도에 이미 성공함.
- failure가 외부 요인임(network down, tool broken) - "network가 down이었다"는 reflection은 future run에 도움이 되지 않음.
- reflection이 미신이 됨 - 일회성 flaky run에 대한 narrative를 저장함.

2026년 pitfall: memory rot. reflection이 쌓입니다. 일부는 obsolete하거나 wrong입니다. episodic buffer가 커질수록 re-run이 느려집니다. mitigation은 periodic compaction(Lesson 06), reflection TTL, 또는 별도 sleep-time cleanup agent(Letta)입니다.

```figure
react-trace
```

## Build It

`code/main.py`는 target sum이 되는 3-element list를 만드는 toy puzzle로 Reflexion을 구현합니다. Actor는 candidate list를 내고, Evaluator는 sum을 check하며, Self-Reflector는 무엇이 잘못됐는지 한 줄로 씁니다. reflection은 다음 trial을 위해 episodic memory에 들어갑니다.

구성 요소:

- `Actor` - reflection을 보면 개선되는 scripted policy.
- `Evaluator.binary()` - target sum에 대한 pass/fail.
- `SelfReflector` - failure에 대한 one-line diagnosis 생성.
- `EpisodicMemory` - TTL semantics를 가진 bounded list.

실행:

```bash
python3 code/main.py
```

trace는 세 trial을 보여 줍니다. Trial 1은 실패하고 reflection이 저장됩니다. Trial 2는 reflection을 보고 개선하지만 여전히 실패합니다. Trial 3은 성공합니다. baseline run(no reflection)과 비교하세요. baseline은 trial 1의 answer에 stuck됩니다.

## Use It

LangGraph는 reflection을 node pattern으로 제공합니다. Claude Code의 `/memory` command와 pro-workflow의 `/learn-rule`은 episodic buffer를 markdown file로 externalize합니다. Letta의 sleep-time compute는 downtime에 Self-Reflector를 실행해 primary agent를 latency-bound로 유지합니다. OpenAI Agents SDK는 Reflexion을 직접 제공하지 않습니다. score로 trajectory를 reject하는 custom Guardrail과 run 사이에 살아남는 memory `Session`으로 직접 만듭니다.

## Ship It

`outputs/skill-reflexion-buffer.md`는 reflection capture, TTL, deduplication이 있는 episodic buffer를 만들고 유지합니다. task class와 failure가 주어지면 다음 trial에 실제로 도움이 되는 reflection을 냅니다(generic한 "be more careful"이 아님).

## Exercises

1. binary evaluator를 target에서 얼마나 먼지 반환하는 scalar evaluator로 바꾸세요. 더 빨리 수렴하나요?
2. reflection에 10 trial TTL을 추가하세요. 그 시점 이후 older reflection은 해가 되나요, 도움이 되나요?
3. heuristic evaluator를 구현하세요. 같은 action이 반복되면 trial을 stuck으로 mark합니다. 이것은 Self-Reflector와 어떻게 상호작용하나요?
4. reflection을 무시하는 adversarial Actor로 Reflexion을 실행하세요. Actor가 reflection을 알아차리게 하는 최소 prompt engineering은 무엇인가요?
5. Reflexion paper의 AlfWorld에 관한 Section 4를 읽으세요. success-rate 130% 개선을 개념적으로 재현하세요. vanilla ReAct 대비 핵심 delta는 무엇인가요?

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Reflexion | "Self-correction" | Shinn et al. 2023 - Actor, Evaluator, Self-Reflector와 episodic memory |
| Verbal reinforcement | "Learning without gradients" | 다음 trial prompt 앞에 붙는 자연어 reflection |
| Episodic memory | "Per-task reflections" | 하나의 task class에 대한 prior reflection의 bounded buffer |
| Scalar evaluator | "Binary success signal" | ground truth에서 온 pass/fail 또는 numeric score |
| Heuristic evaluator | "Pattern-based detector" | stuck-loop, too-many-steps 같은 predefined failure signature |
| Self-evaluator | "LLM-as-judge on own trace" | ground truth가 없을 때 쓰는 낮은 signal의 fallback - tool-grounded verification과 함께 사용 |
| Memory rot | "Stale reflections" | obsolete entry로 episodic buffer가 채워짐. compaction/TTL로 해결 |
| Sleep-time reflection | "Async self-reflection" | primary agent를 빠르게 유지하기 위해 hot path 밖에서 Self-Reflector 실행 |

## Further Reading

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) - 표준 논문
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) - production의 async reflection
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) - context의 일부로 episodic buffer 관리
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - reflection node pattern
