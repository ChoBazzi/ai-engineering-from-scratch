# Agent Rules

## startup/state-file-fresh
- category: startup
- check: state_file_fresh
Agent는 어떤 tool call보다 먼저 agent_state.json을 읽어야 한다.

## forbidden/no-release-script-edits
- category: forbidden
- check: no_release_script_edits
승인된 release task 밖에서는 scripts/release.sh를 절대 수정하지 않는다.

## done/tests-pass
- category: definition_of_done
- check: tests_pass
acceptance command가 0으로 종료될 때만 task가 완료된 것이다.

## uncertainty/open-question-note
- category: uncertainty
- check: opened_question_when_unsure
confidence가 threshold보다 낮으면 추측하지 말고 question note를 작성한다.

## approval/new-dependency
- category: approval
- check: new_dependency_approved
runtime dependency를 추가하려면 명시적인 human approval이 필요하다.
