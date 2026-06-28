# 미션 - Runtime Feedback Loop

## 목표
`subprocess.run`을 감싸 stdout, stderr, exit code, duration을 캡처하고, 출력을 결정적으로 잘라내며, 다음 turn과 verification gate가 모두 읽는 JSONL record를 append하는 `run_with_feedback`을 만듭니다.

## 입력
- runner를 테스트할 세 demo command: 성공 하나, 실패 하나, 느린 것 하나
- Token budget: `...truncated N lines...` marker가 있는 deterministic head plus tail

## 산출물
- `feedback_record.jsonl`에 쓰는 `run_with_feedback(command, agent_note)`
- JSONL을 Python list로 stream하는 loader
- command별 마지막 record를 보여주는 printer

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- `feedback_record.jsonl`이 재실행마다 command당 record 하나를 누적
- `exit_code: null`인 command는 loop가 successful로 표시할 수 없음

## 범위 밖
- Telemetry pipeline(OTel, Langfuse). feedback은 다음 turn을 위한 것이고, telemetry는 operator를 위한 것입니다.
- Redaction pass와 rotation policy. 레슨 연습 prompt에서 다룹니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-feedback-runner.md` - 추출된 skill
