# 미션 - Scope Contract와 Task Boundary

## 목표
task별 `scope_contract.json`과 glob-aware checker를 작성합니다. checker는 agent diff를 contract와 비교해 금지되었거나 scope 밖인 write를 flag합니다.

## 입력
- allowed globs, forbidden globs, acceptance commands, rollback paragraph, approvals required를 담은 task description
- 두 demo run: 하나는 scope 안에 머무르고, 하나는 creeping함

## 산출물
- `scope_contract.json` schema validator(JSON Schema subset, glob arrays)
- touched files와 commands run에서 `RunSummary`를 만드는 diff parser
- `scope_check(contract, run) -> (violations, in_scope, off_scope)`
- script 옆에 저장되는 `scope_report.json`

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- in-scope run이 violation 0개를 보고
- creeping run이 정확한 off-scope file과 각 이유를 보고

## 범위 밖
- 시간 예산, network egress allowlist. 레슨은 file glob을 제공하고, 연습 prompt가 이를 확장합니다.
- runtime interrupt에 연결하는 것. 레슨은 report에서 끝납니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-scope-contract.md` - 추출된 skill
