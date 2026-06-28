# Benchmark: WebArena와 OSWorld

> WebArena는 self-hosted app 네 가지에서 web-agent capability를 테스트한다. OSWorld는 Ubuntu, Windows, macOS 전반의 desktop-agent capability를 테스트한다. release 시점(2023-2024)에는 둘 다 최고 수준 agent와 인간 사이의 큰 격차를 보여 주었다. 격차는 줄고 있지만 failure mode는 바뀌지 않았다.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## 학습 목표

- WebArena의 네 가지 self-hosted app과 execution-based evaluation이 중요한 이유를 설명한다.
- OSWorld가 accessibility API 대신 실제 OS screenshot을 사용하는 이유를 설명한다.
- OSWorld의 두 가지 주요 failure mode인 GUI grounding과 operational knowledge를 말한다.
- OSWorld-G와 OSWorld-Human이 base benchmark 위에 무엇을 추가하는지 요약한다.

## 문제

Generalist agent는 도구를 호출할 수 있다. 하지만 shopping checkout을 완료하기 위해 browser에서 20번의 click을 수행할 수 있는가? keyboard와 mouse만으로 Linux box를 설정할 수 있는가? WebArena와 OSWorld는 이 질문에 답한다.

## 개념

### WebArena(Zhou et al., ICLR 2024)

- 네 가지 self-hosted web app에 걸친 812개의 long-horizon task: shopping site, forum, GitLab-like dev tool, business CMS.
- 추가 utility: map, calculator, scratchpad.
- 평가는 gym API를 통한 execution-based 방식이다. 주문이 들어갔는가, issue가 닫혔는가, CMS page가 업데이트되었는가?
- release 시점: 최고 GPT-4 agent 성공률 14.41% vs human 78.24%.

self-hosted framing이 중요하다. target app이 고정되고 재현 가능하므로 benchmark가 flaky하지 않다.

### 확장

- **VisualWebArena** - image 해석에 성공이 달린 visually grounded task(screenshot이 first-class observation).
- **TheAgentCompany**(2024년 12월) - terminal + coding을 추가한다. 실제 remote-work environment에 더 가깝다.

### OSWorld(Xie et al., NeurIPS 2024)

- Ubuntu, Windows, macOS 전반의 실제 computer task 369개.
- 실제 application에 대한 free-form keyboard와 mouse control.
- observation은 1920x1080 screenshot.
- release 시점: 최고 model 12.24% vs human 72.36%.

### 주요 failure mode

1. **GUI grounding.** pixel에서 element로 매핑하는 문제. model은 1920x1080에서 UI element를 안정적으로 localize하는 데 어려움을 겪는다.
2. **Operational knowledge.** 어떤 menu에 setting이 있는지, 어떤 keyboard shortcut인지, 어떤 preference pane인지. 인간이 수년간 쌓는 지식의 긴 꼬리다.

### 후속 작업

- **OSWorld-G** - 564-sample grounding suite + Jedi training set. grounding과 planning을 분해하여 별도로 측정할 수 있게 한다.
- **OSWorld-Human** - 수작업으로 선별한 gold action trajectory. 최고 agent가 필요한 것보다 1.4-2.7배 더 많은 step을 사용함을 보여 준다(trajectory-efficiency gap).

### 이것이 중요한 이유

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use(Lesson 21)는 모두 WebArena와 OSWorld가 형성한 workload에서 학습한다. benchmark가 목표이고, production model은 출시된 답이다.

### Benchmarking이 잘못되는 지점

- **Screenshot-only eval.** OSWorld는 screenshot-driven이다. DOM이나 accessibility API를 쓰는 agent를 OSWorld에서 평가하면 grounding challenge를 놓친다.
- **Trajectory length 무시.** success rate만 점수화하면 OSWorld-Human이 드러내는 1.4-2.7배 step 비효율을 놓친다.
- **낡은 self-hosted app.** WebArena app은 특정 version에 고정되어 있다. 재선별 없이 update하면 비교 가능성이 깨진다.

## 직접 만들기

`code/main.py`는 toy web-agent harness를 구현한다.

- 최소 "shopping app" state machine: list_items, add_to_cart, checkout.
- 3개 task에 대한 gold trajectory.
- 각 task를 시도하는 scripted agent.
- execution-based evaluator(state check)와 trajectory-efficiency metric(steps vs gold).

실행:

```bash
python3 code/main.py
```

출력: OSWorld-Human의 방법론과 유사한 task별 success rate와 trajectory efficiency.

## 활용하기

- 지속 평가를 위해 내부 cluster에 self-hosted한 **WebArena Verified**.
- desktop agent를 위한 VM fleet의 **OSWorld**.
- **Computer-use agents**(Lesson 21) - Claude, OpenAI CUA, Gemini는 모두 이런 workload에서 학습했다.
- **자체 product flow** - 상위 20개 task의 gold trajectory를 capture하고 매주 agent를 실행하라.

## 출시하기

`outputs/skill-web-desktop-harness.md`는 execution-based eval과 trajectory efficiency metric을 갖춘 web/desktop agent harness를 만든다.

## 연습

1. toy harness에 두 번째 app(forum)을 추가하라. 3개 task와 gold trajectory를 작성하라.
2. task별 trajectory-efficiency reporting을 추가하라. toy에서 agent는 gold보다 1x, 2x, 또는 3x인가?
3. "distractor" tool을 구현하라. gold trajectory가 전혀 쓰지 않는 도구다. scripted agent가 유혹받는가?
4. OSWorld-G를 읽어라. 자체 eval에서 grounding failure와 planning failure를 어떻게 분리하겠는가?
5. WebArena의 apps README를 읽어라. 고정된 app version 중 하나를 upgrade하면 무엇이 깨지는가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| WebArena | "Web agent benchmark" | 4개 self-hosted app에 걸친 812개 task. gym 스타일 evaluation |
| VisualWebArena | "Visual WebArena" | visually grounded WebArena. screenshot이 observation이다 |
| OSWorld | "Desktop agent benchmark" | 실제 Ubuntu/Windows/macOS의 369개 task |
| GUI grounding | "Pixel-to-element mapping" | model이 1920x1080에서 UI element를 localize하는 것 |
| Operational knowledge | "OS know-how" | 어떤 menu, shortcut, preference pane인지 아는 지식 |
| OSWorld-G | "Grounding suite" | grounding-only sample 564개 + training set |
| OSWorld-Human | "Gold trajectories" | 효율을 측정하기 위한 수작업 expert action sequence |
| Trajectory efficiency | "Steps over gold" | agent step count를 human minimum으로 나눈 값 |

## 더 읽을거리

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) - four-app web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) - cross-OS desktop benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) - Claude의 benchmark-shaped capability
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) - OSWorld와 WebArena 수치
