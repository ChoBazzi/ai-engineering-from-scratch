# 미션 - 최소 Agent Workbench

## 목표
새 `workdir/`에 최소 3파일 workbench(router, state, task board)를 만들고, agent turn 하나가 state를 읽고, task를 가져오고, scope에 쓰고, 갱신된 state를 보존할 수 있음을 증명합니다.

## 입력
- lesson code 옆의 빈 `workdir/` directory
- 세 파일에 대한 지식: `AGENTS.md`, `agent_state.json`, `task_board.json`

## 산출물
- 세 파일을 만들고 한 turn을 실행하는 `code/main.py`
- state, board, verification command를 가리키는 짧은 `workdir/AGENTS.md` router
- active task id, touched files, next action을 담은 `workdir/agent_state.json`
- 작은 backlog와 status를 담은 `workdir/task_board.json`

## 합격 기준
- `python3 code/main.py`가 첫 번째와 두 번째 실행 모두에서 0으로 종료
- 두 번째 실행이 처음부터 다시 시작하지 않고 첫 번째가 멈춘 지점에서 이어감
- script가 출력한 diff가 해당 turn이 건드린 파일 하나를 보여줌

## 범위 밖
- Scope contract, verification gate, reviewer agent. 이후 레슨에서 위에 쌓습니다.
- 긴 단일 `AGENTS.md`. router는 의도적으로 짧게 유지합니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-minimal-workbench.md` - 추출된 skill
