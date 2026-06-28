# 멀티모달 에이전트와 컴퓨터 사용(캡스톤)

> 2026년 frontier product는 screenshot을 읽고, button을 click하고, web UI를 navigate하고, form을 채우며, workflow를 end-to-end로 완료하는 multimodal agent다. SeeClick과 CogAgent(2024)는 GUI-grounding primitive를 입증했다. Ferret-UI는 mobile을 추가했다. ChartAgent는 chart를 위한 visual tool-use를 도입했다. VisualWebArena와 AgentVista(2026)는 frontier가 추격하는 benchmark이며, Gemini 3 Pro와 Claude Opus 4.7조차 AgentVista의 어려운 task에서 약 30%를 기록한다. 이 capstone은 Phase 12의 모든 thread를 모은다: perception(고해상도 VLM), reasoning(tool use가 있는 LLM), grounding(coordinate output), long-horizon memory, evaluation.

**Type:** Build
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 minutes

## 학습 목표

- multimodal agent loop를 설계한다: perceive → reason → act → observe → repeat.
- VLM이 JSON으로 내보낼 수 있는 GUI grounding output schema(click coordinates, type text, scroll, drag)를 만든다.
- screenshot-only agent, accessibility-tree agent, hybrid agent를 비교한다.
- 작은 VisualWebArena slice에서 multimodal agent benchmark evaluation을 설정한다.

## 문제

Booking-site workflow: "4월 15일 도쿄행 항공편을 찾아줘. aisle seat이고 $800 미만으로 예약해줘."

Multimodal agent는 다음을 해야 한다.

1. 브라우저 screenshot을 찍는다.
2. screenshot + URL + goal을 plan으로 파싱한다.
3. 구조화된 action을 내보낸다: click(at x,y), type "Tokyo"(at element E), scroll down, select(radio button).
4. action을 브라우저에 적용한다.
5. 새 상태(다음 screenshot)를 관찰한다.
6. task가 끝날 때까지 반복한다.

각 단계는 multimodal VLM call이다. VLM 출력은 parse 가능한 JSON이어야 한다. 오류는 단계마다 누적되므로 recovery가 중요하다.

## 개념

### GUI 그라운딩 기본형

GUI grounding은 screenshot과 자연어 지시가 주어졌을 때 click할 (x, y) 좌표(또는 다른 action)를 출력하는 것이다.

SeeClick(arXiv:2401.10935)은 대규모 첫 공개 결과였다. synthetic + real GUI data로 VLM을 fine-tune하고 좌표를 plain text token으로 출력한다. 작동한다.

CogAgent(arXiv:2312.08914)는 조밀한 UI를 위해 1120x1120 고해상도 encoding을 추가했다. 점수: web navigation에서 약 84%.

Ferret-UI(arXiv:2404.05719)는 mobile UI에 초점을 두고 iOS accessibility data와 통합한다.

출력 형식은 보통 JSON이다.

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc`는 recovery를 돕는다. screenshot 사이에서 좌표가 drift하면 semantic hint가 system이 다시 ground하는 데 도움을 준다.

### 액션 스키마

일반적인 action schema에는 6-10개 action type이 있다.

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent는 step당 하나의 action을 내보낸다. Browser wrapper가 실행하고 새 상태를 반환한다.

### 스크린샷 전용과 접근성 트리

두 입력 mode:

- Screenshot-only: 전체 이미지, 구조 정보 없음. 가장 일반적이며 모든 app에서 작동한다.
- Accessibility tree: 구조화된 DOM / iOS accessibility info. grounding에 훨씬 안정적이며 tree를 사용할 수 있는 곳에서 작동한다.
- Hybrid: 둘 다 사용한다. atomic action에는 tree를 reliable grounder로 쓰고, semantic context에는 screenshot을 쓴다.

Production agent는 가능하면 hybrid를 사용한다. Browser automation(Selenium + accessibility)은 항상 tree를 가진다. Desktop app은 때때로 그렇다.

### 장기 메모리

20단계 workflow는 20개 screenshot을 만든다. VLM context는 빠르게 찬다. 세 가지 compression 전략:

- Summary-chain: 5단계마다 일어난 일을 summarize하고 오래된 screenshot을 버린다.
- Skip-frame: 첫 번째, 마지막, 그리고 3번째마다의 screenshot을 유지한다.
- Tool-recorded log: action을 실행하고, 수행한 일의 text log를 유지한다. 오래된 screenshot을 다시 보지 않는다.

Claude의 computer-use API는 log pattern을 사용한다. 더 단순하고 더 reliable하다.

### 시각 도구 사용

ChartAgent(arXiv:2510.04514)는 chart understanding을 위한 visual tool use를 도입한다: crop, zoom, OCR, external detection 호출. Agent는 "region (100, 200, 300, 400)으로 crop한 뒤 OCR 호출" 같은 tool call을 출력할 수 있다. Tool은 text를 반환하고, VLM은 reasoning을 계속한다.

이 패턴은 일반화된다. Set-of-mark prompting, region annotation, external detection tool은 모두 같은 "tool call을 출력하고 구조화된 response를 받는" schema에 맞는다.

### 2026년 benchmark

- ScreenSpot-Pro. 약 1k web screenshot에서 GUI grounding. 공개 SOTA Qwen2.5-VL-72B 약 85%. Frontier 약 90%.
- VisualWebArena. End-to-end web task(shop, forum, classifieds). 공개 SOTA 약 20%. Gemini 3 Pro 약 27%.
- AgentVista(arXiv:2602.23166). 가장 어려운 2026년 benchmark. 12개 domain의 현실적인 workflow. Frontier model은 27-40%, 공개 model은 10-20%.
- WebArena / WebShop. 더 오래된 benchmark이며 frontier에서는 saturated.

### 여전히 어려운 이유

Agent 성능 병목:

1. 세밀한 visual grounding. "작은 X를 click"은 mobile resolution에서 자주 실패한다.
2. Long-horizon planning. 10개 action 뒤에는 agent가 goal에서 drift한다.
3. Error recovery. click이 실패했을 때(잘못된 button) 이를 감지하고 복구하는 data는 거의 학습되지 않았다.
4. Cross-page context. tab 사이를 이동하거나 긴 form을 작성할 때 state를 잃는다.

연구 방향: memory architecture, explicit replanning, multimodal verification(action success에 대한 screenshot match).

### 캡스톤 구현

Capstone task: 다음을 수행하는 computer-use agent를 만든다.

1. booking-site mock page의 HTML + screenshot을 읽는다.
2. search → select → fill form → submit의 multi-step sequence를 계획한다.
3. action schema와 일치하는 JSON action을 내보낸다.
4. 고정된 10-task slice에서 평가한다.

Lesson은 실제 브라우저로 쉽게 확장할 수 있는 scaffold code를 제공한다.

## 활용하기

`code/main.py`는 capstone scaffold다.

- Action schema JSON definition(10 actions).
- dict 형태의 mock browser state.
- Agent loop skeleton: state를 받고, action을 내보내고, 적용하고, loop.
- end-to-end success rate를 측정하는 10-task mini-benchmark(synthetic pages).
- action 실패 시 사용하는 error-recovery hook.

## 산출물

이 lesson은 `outputs/skill-multimodal-agent-designer.md`를 만든다. computer-use product(domain, action set, evaluation target)가 주어지면 전체 agent loop, memory strategy, grounding mode, expected benchmark score를 설계한다.

## 연습 문제

1. `screenshot_region` tool(crop + zoom)로 action schema를 확장하라. 어떤 task가 이득을 보는가?

2. AgentVista(arXiv:2602.23166)를 읽어라. 가장 어려운 task category와 frontier model이 여전히 실패하는 이유를 설명하라.

3. Long-horizon memory compression: live로 유지하는 screenshot은 4개 이하, log는 원하는 만큼 쓸 수 있는 summary-chain을 설계하라.

4. Error-recovery hook을 만들어라. action failure(button not found)에서 agent는 다음에 무엇을 하는가?

5. Screenshot-only Claude 4.7과 hybrid screenshot + accessibility-tree Qwen2.5-VL을 10개 web task에서 비교하라. 어떤 task에서 어느 쪽이 이기는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | screenshot 위 지시 대상의 (x,y)를 모델이 출력 |
| Action schema | "Tool definitions" | 유효한 action(click, type, scroll, drag)의 JSON description |
| Accessibility tree | "Structured DOM" | browser/iOS API가 제공하는 machine-readable UI hierarchy |
| Hybrid agent | "Screenshot + tree" | 이미지와 구조 정보를 모두 사용하며, 둘 중 하나만 쓸 때보다 reliable함 |
| Visual tool use | "Zoom/crop/detect" | Agent가 plan 중간에 external vision tool(OCR, detection)을 호출 |
| Summary-chain | "Memory compression" | 긴 screenshot history를 주기적인 text summary로 대체 |
| VisualWebArena | "E2E web bench" | end-to-end web task를 위한 2024년 benchmark |
| AgentVista | "2026 hard bench" | 12개 domain의 현실적 workflow; Gemini 3 Pro도 약 30% |

## 더 읽을거리

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
