# 미션 - 실제 Repo에서의 Workbench

## 목표
같은 sample app에 대해 동일한 `/signup` validation task를 prompt-only pipeline과 workbench-guided pipeline으로 실행한 뒤, 회의적인 사람이 읽을 수 있는 before/after comparison report를 출력합니다.

## 입력
- `app.py`(validation 없음), `test_app.py`(happy-path test 하나), `README.md`, forbidden-zone bait인 `scripts/release.sh`를 포함한 `sample_app/`
- 두 pipeline 모두 완전히 scripted, 실제 LLM 호출 없음

## 산출물
- 같은 fixture에 대해 두 pipeline을 orchestrate하는 `code/main.py`
- 다섯 outcome table이 있는 `before-after-report.md`
- downstream charting용 `comparison.json`

## 합격 기준
- `python3 code/main.py`가 0으로 종료
- report가 다섯 outcome을 모두 측정: tests actually ran, acceptance met, files outside scope, handoff quality, reviewer total
- workbench pipeline이 다섯 항목 중 최소 네 항목에서 prompt-only pipeline을 이김

## 범위 밖
- 실제 LLM 연결. pipeline은 재현성을 위해 scripted입니다.
- 모델 튜닝. comparison은 설계상 model을 constant로 유지합니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-workbench-benchmark.md` - 추출된 skill
