# 미션 - 실행 가능한 제약으로서의 Agent Instructions

## 목표
prose instruction을 다섯 category의 machine-checkable rule로 바꾸고, reviewer가 채점할 수 있는 rule report를 출력합니다.

## 입력
- heading마다 하나의 rule이 있고, 각 rule이 slug, category, description, `check` field를 갖는 `docs/agent-rules.md`
- 일부러 두 rule을 위반하는 demo agent run

## 산출물
- `agent-rules.md`를 dataclass로 load하는 parser
- 참조된 `check`마다 하나씩 있는 `rule_checker.py` 스타일 함수
- rule별 pass/fail과 aggregate severity를 담은 `rule_report.json`

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- 출력이 parsed rule set, run trace, rule별 pass/fail을 보여줌
- `rule_report.json`이 의도된 두 위반을 잡아냄

## 범위 밖
- checker를 CI에 연결하는 것. 레슨은 작성된 report에서 끝납니다.
- framework guardrail(OpenAI SDK, LangGraph interrupt). rule set은 이들이 구현하는 human-readable contract입니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-rule-set-builder.md` - 추출된 skill
