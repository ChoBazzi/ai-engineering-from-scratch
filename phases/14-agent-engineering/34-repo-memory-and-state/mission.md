# 미션 - Repo Memory와 Durable State

## 목표
`agent_state.json`과 `task_board.json`용 JSON Schema를 작성하고, load, validate, mutate, atomic write를 수행하는 `StateManager`를 만들며, 두 turn에 걸친 round-trip을 증명합니다.

## 입력
- lesson 32의 3파일 workbench 형태
- required, type, enum, pattern, items를 다루는 stdlib-only validator

## 산출물
- code 옆의 `agent_state.schema.json` 및 `task_board.schema.json`
- temp-and-rename write를 사용하는 `StateManager.load`, `StateManager.update`, `StateManager.commit`
- 두 turn에 걸쳐 state를 변경하고 깨끗하게 reload하는 demo run

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- 잘못된 write(missing required field, bad enum)는 거부되고 persisted되지 않음
- 실행 후 `workdir/agent_state.json`이 schema에 대해 validate됨

## 범위 밖
- SQLite나 external storage backend. 이 레슨의 대상은 local file입니다.
- LangGraph checkpointer, Letta memory block. 같은 아이디어지만 다른 storage라 여기서는 범위 밖입니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-state-schema.md` - 추출된 skill
