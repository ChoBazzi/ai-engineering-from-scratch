---
name: web-desktop-harness
description: execution-based evaluation과 trajectory-efficiency metric을 갖춘 WebArena/OSWorld 스타일 harness를 만든다.
version: 1.0.0
phase: 14
lesson: 20
tags: [webarena, osworld, harness, trajectory-efficiency]
---

target app(web 또는 desktop)과 gold trajectory가 있는 task 목록이 주어지면 eval harness를 만든다.

다음을 만든다.

1. task definition: `(tid, description, gold_steps, success_predicate, state_reset)`.
2. runner: agent를 실행하고, 모든 action을 capture하고, step count + elapsed time + success state를 기록한다.
3. trajectory-efficiency metric: `agent_steps / gold_steps`. task별 및 aggregate로 보고한다.
4. task 사이의 state reset. 다른 task가 더럽힌 state에서 task 하나를 실행하지 않는다.
5. failure-mode classifier: failure마다 grounding miss(잘못된 element)인지 planning miss(잘못된 action)인지 tag한다.

Hard reject:

- task 사이 state reset이 없음. cross-task contamination은 모든 score를 무효화한다.
- success-rate-only reporting. trajectory efficiency는 2026년 표준이다.
- DOM parity 없는 screenshots-only harness. 일부 agent는 DOM+vision을 사용한다. 표면을 특별히 제한하는 경우가 아니라면 둘 다 제공하라.

거부 규칙:

- task에 gold trajectory가 없으면 거부한다. 그것 없이는 efficiency를 측정할 수 없다.
- app이 특정 version에 고정되어 있지 않으면 거부한다. drift는 cross-run comparison을 무효화한다.
- agent가 destructive tool(delete, publish)을 가진다면 app의 sandbox copy가 필요하다.

출력: `tasks.py`, `runner.py`, `failure_classifier.py`, `report.py`, 그리고 reset policy, gold-trajectory sourcing, grounding-vs-planning split을 설명하는 `README.md`. 마지막에는 Lesson 21(computer use models) 또는 Lesson 30(eval-driven development)을 가리키는 "what to read next"로 끝낸다.
