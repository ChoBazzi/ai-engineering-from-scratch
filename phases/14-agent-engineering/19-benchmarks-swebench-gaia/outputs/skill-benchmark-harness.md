---
name: benchmark-harness
description: FAIL_TO_PASS / PASS_TO_PASS gating, contamination check, step-count metric을 갖춘 codebase용 SWE-bench 스타일 harness를 만든다.
version: 1.0.0
phase: 14
lesson: 19
tags: [swe-bench, gaia, agentbench, harness, evaluation]
---

codebase와 (bug, fix) pair 목록이 주어지면 실제 unit test로 gate하고 operational metric을 기록하는 benchmark harness를 만든다.

다음을 만든다.

1. task별 definition: `(tid, description, state_before, fail_to_pass_tests, pass_to_pass_tests, solution)`.
2. agent의 patch를 적용하고, repo의 test suite를 sandbox에서 실행하며, FTP pass count, PTP pass count, step count, tokens, wall-clock, cost를 기록하는 runner.
3. contamination check: issue text를 생성된 patch와 pattern-match하고 overlap이 30% 이상이면 flag한다.
4. task별 score와 aggregate score를 JSON으로 내보내고 P50/P75/P95 step과 cost를 포함하는 reporter.
5. 모든 PR에서 harness를 실행하고 regression이 5% 이상이면 실패하는 CI job.

Hard reject:

- 단일 aggregate number만 보고하는 harness. task별 result와 distribution이 필요하다.
- sandbox 없이 test를 실행하는 harness. agent가 제공한 patch는 untrusted code다.
- PASS_TO_PASS gate가 없는 harness. 다른 test를 깨는 patch는 product를 조용히 regress시킨다.

거부 규칙:

- 사용자가 "FAIL_TO_PASS score만" 요구하면 거부한다. PASS_TO_PASS를 추가하라. 기존 test를 깨는 것은 fix를 놓치는 것보다 나쁜 regression이다.
- test가 특정 commit에 고정되어 있지 않으면 거부한다. test drift는 run 사이 score 비교를 불가능하게 한다.
- task가 training 중 보았을 수 있는 issue text와 겹치면 명시적으로 flag한다.

출력: `tasks.py`, `harness.py`, `contamination.py`, `report.py`, 그리고 sandbox, gate, contamination policy를 설명하는 `README.md`. 마지막에는 harness 위 eval-driven development를 위한 Lesson 30을 가리키는 "what to read next"로 끝낸다.
