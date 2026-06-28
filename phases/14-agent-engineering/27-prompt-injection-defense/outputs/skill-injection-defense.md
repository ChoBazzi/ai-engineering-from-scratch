---
name: injection-defense
description: 모든 agent runtime을 위해 source-tagged content, injection-marker scanning, allowlist navigation을 갖춘 PVE(Prompt-Validator-Executor) layer를 만든다.
version: 1.0.0
phase: 14
lesson: 27
tags: [security, prompt-injection, pve, greshake, source-tag]
---

tool access와 retrieval이 있는 agent가 주어지면 injection-defense layer를 생성한다.

생성할 것:

1. 모든 content 조각의 source tag: `user_message`, `tool_output`, `retrieved_web`, `retrieved_memory`, `retrieved_file`. message history 전체에 tag를 propagate한다.
2. `Validator.assess(tool_call, contents)` — injection-shaped args 또는 retrieved content가 있는 tool call을 거부한다. source tag가 선언된 trust level과 맞을 때만 허용한다.
3. navigation용 allowlist / blocklist: agent가 touch할 수 있는 URL, domain, file path.
4. memory-write guardrail: directive처럼 보이는 write를 거부한다.
5. content-capture discipline(Lesson 23): retrieved content를 외부에 저장한다. span은 prose가 아니라 reference ID를 담는다.
6. test suite: 다섯 Greshake exploit class를 red-team case로 포함한다.

강한 거부 조건:

- source tag 없는 tool-use surface. provenance 없이는 permission level을 구분할 수 없다.
- final output에서만 실행되는 validator. late validation은 무의미하다. model이 이미 행동했다.
- "Trust me, the system prompt handles it." system-prompt hygiene는 control이 아니다.

거부 규칙:

- agent에 source tagging 없는 retrieval capability가 하나라도 있으면 출시를 거부한다. retrieved content는 표준 injection vector다.
- sensitive tool(send message, execute shell, write file in /)에 human-in-the-loop confirmation이 없으면 거부한다.
- memory write가 unguarded이면 거부한다. persistent memory poisoning은 다음 session을 다시 poison한다.

출력: six-control stack, residual risk, ongoing review cadence를 설명하는 `validator.py`, `source_tag.py`, `allowlist.py`, `memory_guard.py`, `red_team.py`, `README.md`. 마지막은 Lesson 21(computer use safety)과 Lesson 23(OTel을 통한 content capture)을 가리키는 "what to read next"로 끝낸다.
