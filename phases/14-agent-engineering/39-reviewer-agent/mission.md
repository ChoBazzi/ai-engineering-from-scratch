# 미션 - Reviewer Agent: Builder와 Marker 분리

## 목표
builder의 artifact를 read-only로 읽고 다섯 dimension에서 10점 만점으로 채점한 `review_report.json`을 출력하는 reviewer loop를 만듭니다. verdict는 pass, soft_fail, hard_fail 중 하나입니다.

## 입력
- 이전 레슨의 diff, state, feedback, verification verdict를 묶는 `ReviewerInputs`
- Rubric dimension: problem fit, scope discipline, assumptions, verification quality, handoff readiness

## 산출물
- dimension마다 하나의 scoring function(레슨용 stub-grade, deterministic)
- 다섯 score, total, verdict를 쓰는 `review_report.json` writer
- 두 demo case: 깨끗한 change와 "right tests, wrong problem" change

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- clean change는 7점 이상과 `pass` verdict를 받음
- wrong-problem change는 최소 한 dimension에서 5점 미만으로 떨어지고 verdict가 `hard_fail`로 바뀜

## 범위 밖
- 실제 LLM 호출. 레슨은 각 dimension을 stub으로 두며, skill은 나중에 모델로 바꿉니다.
- diff 수정. reviewer는 읽고, 채점하고, 보고합니다. patch는 다음 turn builder의 일입니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-reviewer-agent.md` - 추출된 skill
