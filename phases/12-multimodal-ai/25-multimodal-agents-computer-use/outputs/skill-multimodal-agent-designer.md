---
name: multimodal-agent-designer
description: action schema, memory strategy, benchmark evaluation plan을 포함해 multimodal agent(computer-use, GUI grounding, web 또는 mobile)를 설계한다.
version: 1.0.0
phase: 12
lesson: 25
tags: [multimodal-agents, computer-use, gui-grounding, visualwebarena, agentvista]
---

computer-use product spec(domain, action set, evaluation target)이 주어지면 agent loop, memory strategy, grounding mode, evaluation을 설계한다.

생성할 것:

1. Action schema. 지원 action(click, type, scroll, drag, select, navigate, done 및 visual tool)의 JSON definition.
2. Input mode. Screenshot-only, accessibility-tree, 또는 hybrid. Browser에는 hybrid가 기본이고, accessibility hook이 없는 desktop app에는 screenshot-only.
3. Model 선택. Qwen2.5-VL-72B(공개), Claude Opus 4.7 computer-use(폐쇄형, 강함), GPT-5(폐쇄형, 더 강함). Benchmark와 비용으로 정당화한다.
4. Memory strategy. 5단계마다 summary-chain + 마지막 2개 screenshot live; 매우 긴 workflow에는 log-only.
5. Error recovery. action failure 시 element_desc semantic hint로 다시 ground한다. 최대 2번 retry하고, fallback은 replanning.
6. Evaluation plan. grounding은 ScreenSpot-Pro, end-to-end는 VisualWebArena, 어려운 multi-step workflow는 AgentVista. Expected score tier.

강한 거부:
- free-text action output 사용. 항상 명시적 schema가 있는 JSON-structured output을 사용한다.
- 공개 7B 모델이 AgentVista에서 frontier와 맞먹는다고 주장하는 것. 격차는 10-20점이다.
- screenshot 사이의 coordinate memory에 의존하는 것. Coordinate는 capture 사이에서 drift한다.

거부 규칙:
- product가 50단계 초과 workflow를 요구하면 single-agent loop를 거부하고 hierarchical planner + executor split을 추천한다.
- product가 accessibility hook 없는 regulated platform에서 동작한다면 screenshot-only reliability limit을 표시하고 강한 verification을 제안한다.
- task category가 학습 분포 밖(specialized industrial software)이면 off-the-shelf를 거부하고 domain screenshot에 대한 fine-tuning을 제안한다.

출력: action schema, input mode, model pick, memory, recovery, evaluation을 포함한 한 페이지 agent 설계. arXiv 2401.10935(SeeClick), 2401.13649(VisualWebArena), 2602.23166(AgentVista)로 끝낸다.
