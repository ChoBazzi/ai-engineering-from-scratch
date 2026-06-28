# Self-Refine and CRITIC: Iterative Output Improvement

> Self-Refine(Madaan et al., 2023)는 하나의 LLM을 generate, feedback, refine 세 역할로 loop 안에서 사용합니다. 평균 gain은 7개 task에서 absolute +20입니다. CRITIC(Gou et al., 2023)은 verification을 external tool로 route하여 feedback step을 단단하게 만듭니다. 2026년 이 pattern은 모든 framework에서 "evaluator-optimizer"(Anthropic) 또는 guardrail loop(OpenAI Agents SDK)로 배포됩니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 minutes

## Learning Objectives

- Self-Refine의 세 prompt(generate, feedback, refine)를 말하고, refine prompt에서 history가 중요한 이유를 설명합니다.
- CRITIC의 핵심 insight를 설명합니다. LLM은 external grounding 없이는 self-verification에 unreliable합니다.
- history와 optional external verifier를 갖춘 stdlib Self-Refine loop를 구현합니다.
- 이 pattern을 Anthropic의 "evaluator-optimizer" workflow 및 OpenAI Agents SDK의 output guardrail에 mapping합니다.

## The Problem

agent가 거의 맞는 answer를 생성했습니다. code 한 줄에 syntax error가 있을 수 있습니다. summary가 너무 길 수 있습니다. plan이 edge case를 놓쳤을 수 있습니다. 원하는 것은 agent가 자신의 output을 critique한 뒤 고치는 것입니다.

Self-Refine은 이것이 하나의 model, training data 없음, RL 없음으로 작동함을 보여 줍니다. 하지만 함정이 있습니다. LLM은 hard fact에 대한 self-verification을 잘하지 못합니다. CRITIC은 해결책에 이름을 붙입니다. verify step을 external tool(search, code interpreter, calculator, test runner)로 route합니다.

두 논문은 함께 2026년 iterative improvement의 기본값을 정의합니다. generate, verify(가능하면 externally), refine, verifier가 통과하면 stop.

## The Concept

### Self-Refine (Madaan et al., NeurIPS 2023)

하나의 LLM, 세 역할:

```text
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

핵심 detail: `refine`은 전체 history, 즉 이전 output과 critique를 모두 봅니다. 그래서 같은 실수를 반복하지 않습니다. 논문은 이를 ablation합니다. history를 빼면 quality가 급격히 떨어집니다.

headline: GPT-4를 포함해 7개 task(math, code, acronym, dialog) 평균 absolute +20 improvement. training 없음, external tool 없음, single model입니다.

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Self-Refine의 약점은 feedback step이 LLM이 자기 자신을 score하는 것이라는 점입니다. factual claim에서는 unreliable합니다(hallucination은 그것을 만든 model에게도 종종 그럴듯해 보입니다). CRITIC은 `feedback(task, output)`을 `verify(task, output, tools)`로 대체하며, `tools`에는 다음이 포함됩니다.

- factual claim을 위한 search engine.
- code correctness를 위한 code interpreter.
- arithmetic을 위한 calculator.
- domain-specific verifier(unit tests, type checkers, linters).

verifier는 tool result에 grounded한 structured critique를 생성합니다. refiner는 이 critique에 condition합니다.

headline: critique가 grounded되어 있기 때문에 CRITIC은 factual task에서 Self-Refine을 능가합니다. external verifier가 없는 task(creative writing, formatting)에서는 CRITIC이 Self-Refine으로 축소됩니다.

### stop condition

흔한 형태는 두 가지입니다.

1. **Verifier passes.** external test가 success를 반환합니다. 가능하면 선호합니다(unit tests, type checker, guardrail assertion).
2. **No feedback issued.** model이 "the output is fine"이라고 말합니다. 더 싸지만 unreliable합니다. max-iteration cap과 함께 쓰세요.

2026년 기본값은 결합입니다. "verifier passes OR model says fine AND iterations >= 2 OR iterations >= max_iterations"일 때 stop합니다.

### Evaluator-Optimizer (Anthropic, 2024)

Anthropic의 2024년 12월 글은 이를 다섯 workflow pattern 중 하나로 이름 붙입니다. 두 역할이 있습니다.

- Evaluator: output을 score하고 critique를 생성.
- Optimizer: critique를 받아 output을 수정.

evaluator가 pass할 때까지 loop합니다. Anthropic framing에서 이는 Self-Refine/CRITIC입니다. Anthropic이 추가하는 중요한 engineering detail은 evaluator prompt와 optimizer prompt가 상당히 달라야 한다는 점입니다. 그래야 model이 rubber-stamp하지 않습니다.

### OpenAI Agents SDK output guardrails

OpenAI Agents SDK는 이 pattern을 "output guardrails"로 제공합니다. guardrail은 agent의 final output에 실행되는 validator입니다. guardrail이 trip하면(`OutputGuardrailTripwireTriggered` 발생), output은 reject되고 agent가 retry할 수 있습니다. Guardrail은 tool을 호출할 수도 있고(CRITIC-style) pure function일 수도 있습니다(Self-Refine-style).

### 2026년 pitfall

- **Rubber-stamp loops.** 같은 model이 같은 prompt style로 generation과 critique를 수행하면 "looks good to me"에 수렴합니다. 구조적으로 다른 prompt를 쓰거나, critique에는 더 작고 싼 model을 쓰세요.
- **Over-refinement.** refine pass마다 latency와 token이 늘어납니다. 1-3 pass로 budget을 잡고, 그 이후에는 human review로 escalate하세요.
- **CRITIC on trivial tasks.** external verifier가 없다면 CRITIC은 Self-Refine으로 퇴화합니다. stub verifier를 위해 latency를 지불하지 마세요.

## Build It

`code/main.py`는 topic이 주어졌을 때 짧은 bullet list를 만드는 toy task로 Self-Refine과 CRITIC을 구현합니다. verifier는 format(3 bullets, 각 bullet 60자 이하)을 check합니다. CRITIC은 known hallucination에 penalty를 주는 external "fact verifier"를 추가합니다.

구성 요소:

- `generate` - scripted producer.
- `feedback` - LLM-style self-critique.
- `verify_external` - CRITIC-style grounded verifier.
- `refine` - history를 받아 output rewrite.
- Stop condition - verifier passes 또는 max 4 iterations.

실행:

```bash
python3 code/main.py
```

Self-Refine run과 CRITIC run을 비교하세요. CRITIC은 external verifier가 self-critic에게 없는 grounding을 갖고 있기 때문에 Self-Refine이 놓친 factual error를 잡습니다.

## Use It

Anthropic의 evaluator-optimizer는 Claude 친화적 언어로 부르는 같은 pattern입니다. OpenAI Agents SDK의 output guardrails는 CRITIC 형태입니다(guardrail이 tool을 호출할 수 있음). LangGraph는 Self-Refine처럼 읽히는 reflection node를 제공합니다. Google Gemini 2.5 Computer Use는 per-step safety evaluator를 추가하는데, 이는 CRITIC variant입니다. 모든 action은 commit 전에 verify됩니다.

## Ship It

`outputs/skill-refine-loop.md`는 task shape, verifier availability, iteration budget이 주어졌을 때 evaluator-optimizer loop를 구성합니다. generator, evaluator/verifier, optimizer prompt와 stop policy를 냅니다.

## Exercises

1. max_iterations=1로 toy를 실행하세요. 그래도 CRITIC이 도움이 되나요?
2. external verifier를 noisy한 verifier로 바꾸세요(random 30% false positives). loop가 어떻게 행동하나요? 이것이 대부분 guardrail stack의 2026년 현실입니다.
3. "generator-critic on different models" variant를 구현하세요. 큰 model이 generate하고 작은 model이 critique합니다. same-model보다 낫나요?
4. CRITIC Section 3(arXiv:2305.11738 v4)을 읽으세요. 세 verification-tool category를 이름 붙이고 각각의 예를 드세요.
5. OpenAI Agents SDK의 `output_guardrails`를 CRITIC의 verifier role에 mapping하세요. SDK가 잘못하는 것은 무엇이고 잘하는 것은 무엇인가요?

## Key Terms

| Term | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Self-Refine | "LLM that fixes itself" | 하나의 model에서 history를 포함한 generate -> feedback -> refine loop |
| CRITIC | "Tool-grounded verification" | feedback을 external verifier(search, code, calc, tests)로 대체 |
| Evaluator-Optimizer | "Anthropic workflow pattern" | evaluator가 score하고 optimizer가 revise하는 두 역할의 convergence loop |
| Output guardrail | "Post-hoc check" | agent output 후 실행되는 OpenAI Agents SDK validator |
| Verify step | "Critique phase" | grounded인지 self-rated인지가 갈리는 핵심 결정 |
| Refine history | "What the model already tried" | refine prompt 앞에 붙는 prior outputs + critiques. 빼면 quality가 붕괴 |
| Rubber-stamp loop | "Self-agreement failure" | same-prompt critique가 "looks good"을 반환. 구조적으로 다른 prompt로 해결 |
| Stop condition | "Convergence test" | Verifier passes OR no feedback AND iteration cap. single-condition은 금지 |

## Further Reading

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) - 표준 논문
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738) - tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - evaluator-optimizer workflow pattern
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - CRITIC 형태 verifier로서 output guardrail
