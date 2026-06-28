# 미션 - Capstone: 재사용 가능한 Agent Workbench Pack 배포

## 목표
이전 열한 개 레슨을 versioned `outputs/agent-workbench-pack/` directory로 조립하고, 어떤 target repo에도 idempotently 배치하는 installer를 만듭니다.

## 입력
- lesson 32부터 40까지의 schema, script, doc
- pack layout: `AGENTS.md`, `docs/`, `schemas/`, `scripts/`, `bin/`, `README.md`, `VERSION`

## 산출물
- 전체 layout이 채워진 `outputs/agent-workbench-pack/`
- `--force` 없이는 overwrite를 거부하는 `bin/install.sh`(또는 `bin/install.py`)
- 무엇이 포함되고 무엇이 제외되는지 설명하는 `VERSION` file 및 `README.md`

## 합격 기준
- `python3 code/main.py`가 0으로 종료하고 pack tree를 출력
- assembler를 다시 실행해도 idempotent
- fresh target에 `bin/install.sh`를 실행하면 state, board, rules, scope, init, runner, gate, reviewer, handoff가 모두 제자리에 있는 동작하는 workbench가 남음

## 범위 밖
- project별 task content. task는 pack이 아니라 target repo의 board에 속합니다.
- vendor SDK call. pack은 설계상 framework-agnostic입니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-workbench-pack.md` - 추출된 skill
