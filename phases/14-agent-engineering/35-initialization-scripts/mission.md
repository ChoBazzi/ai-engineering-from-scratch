# 미션 - Agent용 Initialization Script

## 목표
runtime, dependency, test command, env var, state freshness를 probe한 뒤 `init_report.json`을 쓰는 `init_agent.py`를 만들고, block-severity probe가 실패하면 session을 크게 중단합니다.

## 입력
- `requirements.txt`(또는 동등한 파일), test command, lesson 34의 workbench state file이 있는 repo
- 레슨의 probe table(runtime, deps, paths, env, state freshness, last-known-good commit)

## 산출물
- probe마다 하나의 함수가 있고 `(name, status, detail)`을 반환하는 `init_agent.py`
- 전체 probe set과 timestamp를 담은 `init_report.json`
- block-severity probe failure가 하나라도 있으면 non-zero exit

## 합격 기준
- happy path에서 `python3 code/main.py`가 0으로 종료
- 연속 두 번 실행해도 timestamp를 제외하면 no-op
- simulated missing env var probe가 report에 드러나고 exit code를 뒤집음

## 범위 밖
- 빠진 dependency를 자동 설치하는 것. script는 중단하고 드러내며, 사람이 고칩니다.
- probe에서 LLM을 호출하는 것. probe는 deterministic plumbing으로 유지합니다.

## 참고
- `docs/en.md` - 전체 레슨
- `code/main.py` - reference implementation
- `outputs/skill-init-script.md` - 추출된 skill
