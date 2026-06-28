# Async Tasks(SEP-1686) — 긴 작업을 지금 호출하고 나중에 가져오기

> 실제 에이전트 작업은 몇 분에서 몇 시간이 걸립니다. CI 실행, deep-research 합성, 배치 export가 그렇습니다. 동기 도구 호출은 연결을 끊거나, 타임아웃되거나, UI를 막습니다. 2025-11-25에 병합된 SEP-1686은 Tasks primitive를 추가합니다. 어떤 요청이든 task가 되도록 보강할 수 있고, 결과는 나중에 가져오거나 상태 알림으로 스트리밍할 수 있습니다. 드리프트 위험 참고: Tasks는 2026년 상반기까지 실험적입니다. SDK 표면은 아직 스펙을 중심으로 설계 중입니다.

**Type:** Build
**Languages:** Python (stdlib, async task state machine)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 09 (transports)
**Time:** ~75 minutes

## 학습 목표

- 도구를 동기 방식에서 task 보강 방식으로 승격해야 하는 시점을 식별합니다. 기준은 서버 측 작업이 30초를 넘는 경우입니다.
- task 수명주기 `working` → `input_required` → `completed` / `failed` / `cancelled`를 따라갑니다.
- 진행 중 작업이 크래시로 사라지지 않도록 task 상태를 영속화합니다.
- `tasks/status`를 poll하고 `tasks/result`를 올바르게 가져옵니다.

## 문제

`generate_report` 도구가 몇 분이 걸리는 추출 파이프라인을 실행합니다. 동기 모델에서 가능한 선택지는 다음과 같습니다.

1. 연결을 3분 동안 열어 둡니다. 원격 transport는 연결을 끊고, 클라이언트는 타임아웃되며, UI는 멈춥니다.
2. placeholder를 즉시 반환하고 클라이언트가 커스텀 endpoint를 poll하게 합니다. MCP의 일관성을 깨뜨립니다.
3. fire-and-forget으로 실행합니다. 결과가 없습니다.

어느 것도 좋지 않습니다. SEP-1686은 네 번째 선택지인 task augmentation을 추가합니다. 어떤 요청(일반적으로 `tools/call`)이든 task로 태그할 수 있습니다. 서버는 즉시 task id를 반환합니다. 클라이언트는 `tasks/status`를 poll하고 완료되면 `tasks/result`를 가져옵니다. 서버 측 상태는 재시작 후에도 살아남습니다.

## 개념

### Task augmentation

요청은 `params._meta.task.required: true`(또는 `optional: true`, 서버가 결정)를 설정하면 task가 됩니다. 서버는 즉시 다음을 응답합니다.

```json
{
  "jsonrpc": "2.0", "id": 1,
  "result": {
    "_meta": {
      "task": {
        "id": "tsk_9f7b...",
        "state": "working",
        "ttl": 900000
      }
    }
  }
}
```

`ttl`은 서버가 상태를 보관하겠다는 약속입니다. ttl 이후 task 결과는 폐기됩니다.

### 도구별 opt-in

도구 annotation은 task 지원을 선언할 수 있습니다.

- `taskSupport: "forbidden"` — 이 도구는 항상 동기적으로 실행됩니다. 빠른 도구에 안전합니다.
- `taskSupport: "optional"` — 클라이언트가 task augmentation을 요청할 수 있습니다.
- `taskSupport: "required"` — 클라이언트는 task augmentation을 반드시 사용해야 합니다.

`generate_report` 도구는 `required`가 적절합니다. `notes_search` 도구는 `forbidden`이 적절합니다.

### 상태

```text
working  -> input_required -> working  (elicitation을 통한 루프)
working  -> completed
working  -> failed
working  -> cancelled
```

상태 머신은 append-only입니다. `completed`, `failed`, `cancelled`가 되면 task는 terminal 상태입니다.

### 메서드

- `tasks/status {taskId}` — 현재 상태와 progress hint를 반환합니다.
- `tasks/result {taskId}` — 아직 완료되지 않았으면 block하거나 404를 반환합니다.
- `tasks/cancel {taskId}` — 멱등적입니다. terminal 상태는 무시합니다.
- `tasks/list` — 선택 사항입니다. 활성 및 최근 완료 task를 열거합니다.

### 상태 변경 스트리밍

서버가 지원하면 클라이언트는 상태 알림을 구독할 수 있습니다.

```text
server -> notifications/tasks/updated {taskId, state, progress?}
```

poll 대신 스트리밍하는 클라이언트는 더 나은 UX를 얻습니다. polling은 항상 지원되는 최소 표면입니다.

### Durable state

스펙은 task 지원을 선언한 서버가 상태를 영속화할 것을 요구합니다. 크래시는 ttl 안의 완료 결과를 잃어버리면 안 됩니다. 저장소는 SQLite부터 Redis, 파일시스템까지 다양합니다. Lesson 13 하네스는 파일시스템을 사용합니다.

### 취소 semantics

`tasks/cancel`은 멱등적입니다. task가 실행 중이면 서버는 중지를 시도합니다(executor-cooperative cancellation 확인). 이미 terminal 상태라면 요청은 no-op입니다.

### 크래시 복구

서버 프로세스가 재시작되면 다음을 수행합니다.

1. 영속화된 모든 task 상태를 로드합니다.
2. 프로세스가 죽은 `working` task를 오류 `CRASH_RECOVERY`가 있는 `failed`로 표시합니다.
3. `completed` / `failed` / `cancelled`를 각각의 ttl 동안 보존합니다.

### Async tasks와 sampling

task 자체가 `sampling/createMessage`를 호출할 수 있습니다. 긴 research task가 이렇게 동작합니다. 서버의 task thread는 필요할 때 클라이언트의 모델을 sampling하고, 클라이언트 UI는 periodic progress update와 함께 task를 `working`으로 보여줍니다.

### 이것이 실험적인 이유

SEP-1686은 2025-11-25에 출시되었지만, 더 넓은 roadmap은 세 가지 열린 이슈를 명시합니다. durable subscription primitive, subtasks(parent-child task 관계), result-TTL 표준화입니다. 스펙은 2026년 내내 진화할 것으로 예상하세요. 프로덕션 코드는 Tasks를 common case에서만 안정적이라고 보고, subtasks를 위한 향후 SDK 변경에 대비해야 합니다.

## 사용하기

`code/main.py`는 durable task store(파일시스템 기반)와 백그라운드 thread에서 실행되는 `generate_report` 도구를 구현합니다. 클라이언트는 도구를 호출해 즉시 task id를 받고, worker가 progress를 업데이트하는 동안 `tasks/status`를 poll하며, 완료되면 `tasks/result`를 가져옵니다. 취소가 동작합니다. worker thread를 죽이고 상태를 다시 로드해 크래시 복구를 시뮬레이션합니다.

살펴볼 점:

- Task 상태 JSON은 `/tmp/lesson-13-tasks/<id>.json`에 영속화됩니다.
- Worker thread가 `progress` 필드를 업데이트합니다. poll 결과에서 진행되는 모습을 볼 수 있습니다.
- 클라이언트 측 취소는 event를 설정합니다. worker는 이를 확인하고 일찍 종료합니다.
- "crash" 후 상태를 다시 로드하면 진행 중이던 task가 `CRASH_RECOVERY`가 있는 `failed`로 표시됩니다.

## 산출물

이 lesson은 `outputs/skill-task-store-designer.md`를 만듭니다. 오래 실행되는 도구(연구, 빌드, export)가 주어지면, 이 skill은 task store(상태 형태, ttl, durability)를 설계하고, 올바른 taskSupport 플래그를 고르며, progress notification을 스케치합니다.

## 연습

1. `code/main.py`를 실행하세요. `generate_report` task를 시작하고, status를 poll한 다음 결과를 가져오세요.

2. 실행 중간에 `tasks/cancel` 호출을 추가하세요. worker가 이를 따르고 상태가 `cancelled`가 되는지 확인하세요.

3. 크래시 복구를 시뮬레이션하세요. worker thread를 죽이고 loader를 재시작한 뒤 `CRASH_RECOVERY` 실패 모드를 관찰하세요.

4. store를 SQLite로 확장하세요. durability 이점은 같고, query 옵션이 열립니다(세션 X의 모든 task 나열).

5. 2026년 MCP roadmap post를 읽으세요. 다음 해 SDK API 설계에 가장 영향을 줄 가능성이 높은 Tasks 관련 열린 이슈 하나를 찾으세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Task | "긴 도구 호출" | 비동기 실행을 위해 `_meta.task`로 보강된 요청 |
| SEP-1686 | "Tasks spec" | 2025-11-25에 Tasks를 추가한 Spec Evolution Proposal |
| `_meta.task` | "Task envelope" | id, state, ttl을 담는 요청별 metadata |
| taskSupport | "도구 플래그" | 도구별 `forbidden` / `optional` / `required` |
| `tasks/status` | "Poll method" | 현재 상태와 선택적 progress hint를 가져옴 |
| `tasks/result` | "결과 가져오기" | 완료 payload를 반환하거나 아직 완료 전이면 404 반환 |
| `tasks/cancel` | "멈춰" | 멱등적 취소 요청 |
| ttl | "보존 예산" | 서버가 task 상태를 유지하겠다고 약속한 밀리초 |
| `notifications/tasks/updated` | "상태 push" | 서버가 시작하는 상태 변경 이벤트 |
| Durable store | "크래시 안전 상태" | 파일시스템 / SQLite / Redis 영속화 계층 |

## 더 읽을거리

- [MCP — GitHub SEP-1686 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1686) — 최초 제안과 전체 토론
- [WorkOS — MCP async tasks for AI agent workflows](https://workos.com/blog/mcp-async-tasks-ai-agent-workflows) — 근거가 포함된 설계 walkthrough
- [DeepWiki — MCP task system and async operations](https://deepwiki.com/modelcontextprotocol/modelcontextprotocol/2.7-task-system-and-async-operations) — mechanics와 상태 머신
- [FastMCP — Tasks](https://gofastmcp.com/servers/tasks) — SDK 수준 task 구현 패턴
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — subtasks를 포함한 열린 이슈와 2026 우선순위
