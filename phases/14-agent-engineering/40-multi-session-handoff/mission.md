# 미션 - Multi-Session Handoff

## 목표
session이 끝날 때 workbench artifact에서 `handoff.md`와 `handoff.json`을 생성해 다음 session이 첫 1분부터 생산적이게 합니다. 두 형식은 같은 일곱 field를 담고, 불일치하면 JSON이 우선합니다.

## 입력
- 이전 레슨의 `agent_state.json`, `verification_report.json`, `review_report.json`, `feedback_record.jsonl`
- 일곱 field: summary, changed_files, commands_run, failed_attempts, open_risks, next_action, verdict_pointer

## 산출물
- 네 artifact를 묶어 load하는 `WorkbenchSnapshot`
- `generate_handoff(snapshot) -> (markdown, payload)`
- 마지막 K개 record와 모든 non-zero exit을 고르는 feedback filter
- script 옆에 쓰는 `handoff.md`와 `handoff.json`

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- 두 파일 모두 일곱 field와 비어 있지 않은 `next_action`을 담음
- 같은 입력으로 script를 다시 실행하면 동일한 packet을 생성

## 범위 밖
- Compaction 전략(Codex compact endpoint, Claude Code five-stage). handoff는 session을 닫고, compaction은 하나의 session을 연장합니다.
- PR templating. markdown은 PR body로 재사용할 수 있지만 레슨은 파일에서 멈춥니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-handoff-generator.md` - 추출된 skill
