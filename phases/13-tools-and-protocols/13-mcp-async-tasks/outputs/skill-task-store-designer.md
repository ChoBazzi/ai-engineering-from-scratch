---
name: task-store-designer
description: 긴 MCP 도구를 위한 task store를 설계합니다. state shape, ttl, durability, cancellation, crash recovery를 포함합니다.
version: 1.0.0
phase: 13
lesson: 13
tags: [mcp, tasks, durable-store, long-running, sep-1686]
---

오래 실행되는 도구(연구, 빌드, export, report generation)가 주어지면, SEP-1686 task augmentation을 뒷받침하는 task store를 설계하세요.

다음을 산출하세요.

1. State shape. 최소 필드: `id`, `state`, `progress`, `result`, `error`, `ttl`, `created_at`. 선택: `request_meta`, `parent_task_id`(향후 subtasks용).
2. Durability choice. toy에는 파일시스템, single-process에는 SQLite, multi-replica에는 Redis를 고르세요. 정당화하세요.
3. taskSupport flag. 도구별 `forbidden`, `optional`, `required`와 한 줄 근거.
4. Cancellation plan. worker가 cancel signal을 확인하는 방식과 부분 진행 상태에서 일어나는 일을 설명하세요.
5. Crash recovery. boot-time reload rule과 `CRASH_RECOVERY` 실패가 클라이언트에 어떻게 보이는지 설명하세요.

강한 거부 조건:
- ttl 안의 완료 결과를 잃는 모든 store.
- 명시적 terminal state(`completed`, `failed`, `cancelled`)가 없는 모든 task state.
- 멱등적이지 않은 cancellation.

거부 규칙:
- 도구가 5초 미만으로 실행된다면 task 승격을 거부하세요. 동기가 더 단순합니다.
- task가 10 MB가 넘는 결과를 생성한다면 거부하고 streaming content blocks를 권장하세요.
- 서버에 상태를 영속화할 수 있는 프로세스가 없다면(stateless edge function), 거부하고 durable runtime으로 옮기라고 권장하세요.

출력: state shape, durability choice, taskSupport flag, cancellation plan, crash-recovery rule이 담긴 한 페이지 store 설계. SEP-1686 subtasks가 출시될 때 이 설계에 영향을 줄지에 대한 한 줄 조언으로 끝내세요.
