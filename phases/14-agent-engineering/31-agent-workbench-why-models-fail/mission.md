# 미션 - Agent Workbench: 유능한 모델이 여전히 실패하는 이유

## 목표
같은 작은 repo task를 두 번 실행합니다. 한 번은 prompt-only로, 한 번은 일곱 workbench surface를 연결한 상태로 실행하고, 놓친 각 surface가 어떤 증상을 만들었는지 매핑하는 failure-mode report를 출력합니다.

## 입력
- 검증할 stub agent와 작은 FastAPI 스타일 handler
- 일곱 surface 목록(instructions, state, scope, feedback, verification, review, handoff)

## 산출물
- 두 pipeline을 연달아 실행하는 `code/main.py`
- prompt-only run을 요약하는 `failure_modes.json`
- workbench run에 대한 한 줄 verdict

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- 출력이 두 run의 로그를 나란히 보여줌
- `failure_modes.json`이 놓친 모든 surface와 대응하는 symptom을 나열

## 범위 밖
- 실제 모델 호출. stub은 의도적으로 rule-based입니다.
- 특정 surface 하나를 깊게 구축하는 것. 다음 열한 개 레슨에서 다룹니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-workbench-audit.md` - 추출된 skill
