# 미션 - Verification Gate

## 목표
scope report, rule report, feedback log, diff 위에서 동작하는 순수 deterministic function으로 `verify(task_id, artifacts)`를 구현하고, task close-out마다 하나의 `verification_report.json`을 출력합니다.

## 입력
- `scope_report.json`, `rule_report.json`, `feedback_record.jsonl`, diff용 stub loader
- check table: acceptance ran, acceptance exited zero, scope clean, `null` exit 없음, 모든 block-severity rule pass

## 산출물
- 순수 함수 `verify(task_id, artifacts) -> VerdictReport`
- check별 결과와 최종 pass/fail을 보여주는 printer
- disk에 쓰는 세 demo scenario: clean pass, scope creep, missing acceptance

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- clean-pass scenario는 `passed: true`, 나머지 둘은 `passed: false`를 보고
- 각 scenario가 `outputs/verification/` 아래 별도의 `verification_report.json` 작성

## 범위 밖
- LLM-as-judge logic. gate는 deterministic으로 유지하고, qualitative judgment는 lesson 39의 reviewer에게 둡니다.
- Signed override audit log. 연습 prompt에서 gate를 그 방향으로 확장합니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-verification-gate.md` - 추출된 skill
