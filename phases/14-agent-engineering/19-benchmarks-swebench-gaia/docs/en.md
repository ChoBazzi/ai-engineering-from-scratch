# Benchmark: SWE-bench, GAIA, AgentBench

> 2026년 agent evaluation을 지탱하는 세 가지 benchmark가 있다. SWE-bench는 code patching을 테스트한다. GAIA는 generalist tool use를 테스트한다. AgentBench는 multi-environment reasoning을 테스트한다. 각 benchmark의 구성, contamination 이야기, 그리고 무엇을 측정하지 않는지 알아야 한다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 06 (Tool Use)
**Time:** ~60 minutes

## 학습 목표

- SWE-bench의 test harness(FAIL_TO_PASS)를 말하고, 왜 unit test로 gate하는지 설명한다.
- SWE-bench Verified(OpenAI, 500 tasks)가 존재하는 이유와 무엇을 제거하는지 설명한다.
- GAIA의 설계를 설명한다. 사람에게는 단순하지만 AI에게는 어렵고, 세 가지 난이도 level이 있다.
- AgentBench의 여덟 environment와 open-source LLM의 주요 blocker를 말한다.
- SWE-bench+ contamination finding과 그 함의를 요약한다.

## 문제

Leaderboard는 한 benchmark에서 어떤 model이 이기는지 알려 준다. 그러나 다음은 알려 주지 않는다.

- benchmark가 오염되었는지(해결책이 training data에 있거나 test leakage가 있는지).
- benchmark가 당신이 신경 쓰는 것을 측정하는지(code vs browsing vs generalist).
- evaluator가 견고한지(AST matching, state check, human review).

숫자를 인용하기 전에 세 가지 anchor benchmark와 그 failure mode를 알아야 한다.

## 개념

### SWE-bench(Jimenez et al., ICLR 2024 oral)

- 인기 있는 Python repo 12개의 실제 GitHub issue 2,294개.
- Agent가 받는 것: fix 전 commit의 codebase + 자연어 issue description.
- Agent가 만드는 것: patch.
- Evaluator: patch 적용 후 repo의 test suite 실행. patch는 FAIL_TO_PASS test(이전에는 실패, 이제는 통과)를 뒤집어야 하고 PASS_TO_PASS test를 깨면 안 된다.

SWE-agent(Yang et al., 2024)는 agent-computer interface(파일 editor command, 모델이 이해하는 search syntax)를 강조해 release 시점에 12.5%를 달성했다.

### SWE-bench Verified

OpenAI, 2024년 8월. 사람이 선별한 500-task subset. 애매한 issue, 신뢰할 수 없는 test, fix가 불명확한 task를 제거한다. "agent가 실제 patch를 낼 수 있는가?"를 보는 주요 benchmark다.

### Contamination

- SWE-bench issue의 94% 이상은 대부분의 model cutoff보다 오래되었다.
- **SWE-bench+**는 성공 patch의 32.67%에서 issue text에 solution이 누출되어 있었고(model이 description 안에서 fix를 봄), 31.08%는 약한 test coverage 때문에 의심스럽다고 보았다.
- Verified는 더 깨끗하지만 contamination-free는 아니다.

실무적 함의: SWE-bench에서 50%를 기록하는 model이 SWE-bench+에서는 35%일 수 있다. SWE-bench 성능을 주장한다면 항상 둘 다 보고하라.

### GAIA(Mialon et al., 2023년 11월)

- 질문 466개. huggingface.co/gaia-benchmark의 private leaderboard에는 300개가 유지된다.
- 설계 철학: "사람에게는 개념적으로 단순하지만(92%) AI에게는 어렵다(GPT-4 with plugins: 15%)."
- reasoning, multi-modality, web, tool use를 테스트한다.
- 세 가지 difficulty level이 있다. Level 3는 modality를 넘나드는 긴 tool chain을 요구한다.

GAIA는 "generalist capability"를 측정할 때 돌리는 benchmark다. code-specific benchmark와 혼동하지 말라.

### AgentBench(Liu et al., ICLR 2024)

- code(Bash, DB, KG), game(Alfworld, LTP), web(WebShop, Mind2Web), open-ended generation에 걸친 8개 environment.
- multi-turn이며 split당 약 4k-13k turn.
- 주요 finding: long-term reasoning, decision-making, instruction following이 OSS LLM이 commercial model을 따라잡는 데 blocker다.

### 이들이 측정하지 않는 것

- 실제 운영 비용(tokens, wall-clock).
- 적대적 조건에서의 safety behavior.
- 당신의 domain에서의 성능(자체 eval을 사용하라, Lesson 30).
- tail failure(benchmark는 평균을 낸다. production operator는 최악의 1%를 본다).

### Benchmarking이 잘못되는 지점

- **단일 숫자 집착.** SWE-bench 50%는 P50/P75/P95 비용 + step distribution보다 덜 알려 준다.
- **오염된 주장.** Verified나 SWE-bench+를 언급하지 않고 SWE-bench를 보고하는 것은 오해를 부른다.
- **Benchmark-as-development-target.** benchmark에 최적화하면 production usefulness에서 멀어진다.

## 직접 만들기

`code/main.py`는 toy SWE-bench-like harness를 구현한다.

- synthetic bug-fix task(3개 task).
- patch를 제안하는 scripted "agent".
- FAIL_TO_PASS(버그가 이제 고쳐짐)와 PASS_TO_PASS(깨진 것이 없음)를 확인하는 test runner.
- question decomposition depth 기반의 GAIA 스타일 difficulty classifier.

실행:

```bash
python3 code/main.py
```

출력은 task별 + difficulty별 resolution rate를 보여 주고 evaluator rule을 구체화한다.

## 활용하기

- code agent에는 **SWE-bench Verified**를 사용하라. 항상 Verified score를 보고하라.
- generalist agent에는 **GAIA**를 사용하라. private leaderboard split을 사용하라.
- multi-environment comparison에는 **AgentBench**를 사용하라.
- 제품의 실제 형태에는 **Custom evals**(Lesson 30)를 사용하라.

## 출시하기

`outputs/skill-benchmark-harness.md`는 어떤 codebase-task pair에도 FAIL_TO_PASS / PASS_TO_PASS gating을 가진 SWE-bench 스타일 harness를 만든다.

## 연습

1. toy harness를 실제 repo(자신의 것 하나 선택)에서 실행하도록 이식하라. 알려진 bug에 대해 3개의 FAIL_TO_PASS test를 작성하라.
2. step-count metric을 추가하라. 3개 task에서 resolution마다 agent step은 몇 개인가?
3. SWE-bench+ paper를 읽어라. solution-leakage check를 구현하라(issue text를 diff와 pattern match).
4. public split에서 GAIA question 하나를 다운로드하라. GPT-4급 agent가 무엇을 할지 trace하라. 어떤 tool이 필요한가?
5. AgentBench의 environment별 breakdown을 읽어라. 어떤 environment가 제품 표면과 닮았는가? 그곳의 "SOTA"는 어떤 모습인가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| SWE-bench | "Code agent benchmark" | GitHub issue 2,294개. patch는 FAIL_TO_PASS test를 뒤집어야 한다 |
| SWE-bench Verified | "Clean SWE-bench" | OpenAI가 사람이 선별한 500 task |
| FAIL_TO_PASS | "Fix gate" | patch 후 반드시 통과해야 하는, 이전에는 실패하던 test |
| PASS_TO_PASS | "No-regression gate" | 이미 통과하던 test이며 계속 통과해야 한다 |
| GAIA | "Generalist benchmark" | 사람에게 쉽고 AI에게 어려운 multi-tool question 466개 |
| AgentBench | "Multi-env benchmark" | 8개 environment. long-horizon multi-turn |
| Contamination | "Training-set leak" | model training에 benchmark task가 존재함 |
| SWE-bench+ | "Contamination audit" | 성공한 SWE-bench patch 중 32.67% solution leakage를 발견 |

## 더 읽을거리

- [Jimenez et al., SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770) - 원 benchmark
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) - 선별 subset
- [Mialon et al., GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983) - generalist benchmark
- [Liu et al., AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688) - multi-environment suite
