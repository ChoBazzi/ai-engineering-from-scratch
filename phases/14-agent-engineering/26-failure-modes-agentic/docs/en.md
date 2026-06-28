# Failure Modes: 에이전트가 깨지는 이유

> MASFT(Berkeley, 2025)는 3개 category의 14가지 multi-agent failure mode를 catalog화한다. Microsoft의 Taxonomy는 기존 AI failure가 agentic setting에서 어떻게 증폭되는지 문서화한다. industry field data는 다섯 반복 mode로 수렴한다: hallucinated actions, scope creep, cascading errors, context loss, tool misuse.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 minutes

## Learning Objectives

- MASFT의 세 failure category와 각 category의 구체적 mode 네 가지 이상을 말한다.
- agentic failure가 기존 AI failure mode(bias, hallucination)를 증폭하는 이유를 설명한다.
- industry에서 반복되는 다섯 mode와 mitigation을 설명한다.
- agent trace에 failure-mode label을 tag하는 stdlib detector를 구현한다.

## The Problem

팀은 trace의 90%에서 작동하는 agent를 출시한다. 나머지 10% failure는 random noise가 아니다. 소수의 반복 category로 떨어진다. 이름을 붙일 수 있으면 monitor하고 고칠 수 있다.

## The Concept

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy. 14가지 failure mode를 3개 category로 cluster한다. inter-annotator Cohen's Kappa 0.88 — category가 안정적으로 구분 가능하다는 뜻이다.

중심 주장: failure는 더 나은 base model로 고칠 LLM limitation이 아니라 multi-agent system의 근본적인 design flaw다.

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 기존 AI failure(bias, hallucination, data leakage)는 agentic setting에서 증폭된다.
- autonomy에서 새로운 failure가 생긴다: unintended action at scale, tool misuse, mission drift.
- 이 whitepaper는 agentic product의 risk register다.

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- failure는 orchestration, internal state evolution, environment interaction에서 생긴다.
- 단순한 "bad code"나 "bad model output"이 아니다.

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

두 가지 주요 manifestation:

1. **Instruction-following Deviation** — agent가 system prompt를 따르지 않는다.
2. **Long-range Contextual Misuse** — agent가 이전 turn의 context를 잊거나 잘못 적용한다.

Sub-intention error: Omission(빠진 step), Redundancy(반복 step), Disorder(순서가 틀린 step).

### The five industry-recurring modes

Arize, Galileo, NimbleBrain의 2024-2026 field analysis는 다음으로 수렴한다.

1. **Hallucinated actions.** agent가 존재하지 않는 tool을 invoke하거나 argument를 fabricate한다.
2. **Scope creep.** agent가 사용자의 요청을 넘어 task를 확장한다(추가 PR 생성, 추가 email 발송).
3. **Cascading errors.** 하나의 잘못된 call이 downstream effect를 유발한다. phantom SKU hallucination이 네 개 API call을 촉발해 multi-system incident가 된다.
4. **Context loss.** long-horizon task가 early-turn constraint를 잊는다.
5. **Tool misuse.** 올바른 tool을 잘못된 argument로 호출하거나 아예 잘못된 tool을 호출한다.

Cascading이 치명적이다. agent는 "I failed"와 "the task is impossible"을 구분하지 못하고, loop를 닫기 위해 400 error에서 success message를 hallucinate하는 일이 흔하다.

### Mitigation: gates at every step

reasoning chain의 모든 step에 automated verification gate를 두고 environment state에 대한 factual grounding을 확인한다. 구체적으로:

- step별 safety classifier(Lesson 21).
- tool-call argument validation(Lesson 06).
- retrieved content를 known fact와 cross-check(Lesson 05, CRITIC).
- state를 다시 probe해 success hallucination을 감지(파일이 실제로 생성되었는가?).

### Where failure monitoring goes wrong

- **crash만 tag함.** 대부분의 agent failure는 valid-looking output을 만든다. content-level check가 필요하다.
- **baseline 없음.** drift detection에는 last-known-good이 필요하다. 없으면 "this is getting worse"라고 말할 수 없다.
- **over-alerting.** 모든 failure가 page를 만든다. cluster하고 rate-limit하라.

## Build It

`code/main.py`는 stdlib failure-mode tagger를 구현한다.

- 다섯 mode를 포괄하는 synthetic trace dataset.
- mode별 detector function(tool call, output, repeat action에 대한 signature pattern).
- 각 trace에 label을 붙이고 mode distribution을 report하는 tagger.

실행:

```bash
python3 code/main.py
```

출력: trace별 label + aggregate distribution. Phoenix의 trace clustering이 드러내는 내용을 저렴하게 재현한다.

## Use It

- **Phoenix** — production drift clustering에 사용(Lesson 24).
- **Langfuse** — session replay + annotation에 사용.
- **Custom** — observability platform이 감지하지 못하는 domain-specific signature에 사용.

## Ship It

`outputs/skill-failure-detector.md`는 domain에 맞춘 failure-mode detector를 생성하고 trace store에 연결한다.

## Exercises

1. "success hallucination" detector를 추가한다. agent가 success를 반환하지만 target state는 변하지 않는다.
2. 자신이 만든 product의 real trace 100개를 tag한다. 어떤 mode가 지배적인가? 고치는 비용은 얼마인가?
3. "cascade radius" metric을 구현한다. step N의 failure가 downstream step 몇 개에 영향을 주었는가?
4. MASFT의 14 failure mode를 읽는다. 자신의 product에 적용되는 세 가지를 고르고 detector를 작성한다.
5. detector 하나를 CI job에 연결한다. trace의 >=5%가 mode에 tag되면 build를 fail한다.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MASFT | "Multi-agent failure taxonomy" | Berkeley의 14-mode categorization |
| Cascading error | "Ripple failure" | 초기 실수 하나가 N step에 전파됨 |
| Context loss | "Forgot the constraint" | long-horizon turn이 early-turn fact를 놓침 |
| Tool misuse | "Wrong tool / wrong args" | 유효한 call이지만 invocation이 잘못됨 |
| Success hallucination | "Faked completion" | agent가 400에서 success를 주장하지만 state는 unchanged |
| Scope creep | "Overreach" | agent가 요청보다 더 많이 수행함 |
| Instruction-following deviation | "Disobedience" | system prompt 또는 user constraint를 무시함 |
| Sub-intention errors | "Plan bugs" | plan execution의 omission, redundancy, disorder |

## Further Reading

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 failure mode, 3 category
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — risk register
- [Arize Phoenix](https://docs.arize.com/phoenix) — 실제 drift clustering
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — 더 단순한 pattern이 mode를 완전히 피하는 경우
