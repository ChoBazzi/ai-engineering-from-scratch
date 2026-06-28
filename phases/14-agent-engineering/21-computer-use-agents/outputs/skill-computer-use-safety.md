---
name: computer-use-safety
description: allowlist navigation과 injection-marker filtering을 포함해 computer-use agent용 step별 safety classifier + confirmation gate를 만든다.
version: 1.0.0
phase: 14
lesson: 21
tags: [computer-use, safety, claude, openai-cua, gemini]
---

computer-use agent와 target app 목록이 주어지면 실행 전 모든 action을 classify하는 safety layer를 만든다.

다음을 만든다.

1. `allow`, `reason`, `needs_confirmation` field를 가진 `SafetyClassifier.assess(action, screen) -> SafetyVerdict`.
2. agent가 click할 수 있는 element label allowlist. 그 외에는 거부한다.
3. agent가 navigate할 수 있는 URL allowlist. 목록 밖 redirect는 거부한다.
4. DOM text, retrieved content, typed text에 대한 injection-marker filter. match가 있으면 action을 block한다.
5. sensitive action(login, purchase, delete, publish)을 위한 confirmation gate. human-in-the-loop callback interface.
6. trace emitter: 모든 decision을 (action, verdict, reason)과 함께 log한다.

Hard reject:

- 첫 action에서만 실행되는 safety classifier. 모든 action은 classify되어야 한다.
- `*` 형태의 allowlist. 모든 것을 허용하는 allowlist는 allowlist가 아니다.
- model이 "확신해 보인다"는 이유로 confirmation을 건너뛰는 것. confidence는 safety가 아니다.

거부 규칙:

- agent가 step별 safety 없이 computer-use access를 가진다면 ship을 거부한다.
- agent가 임의 URL로 navigate할 수 있다면 거부한다. allowlist 또는 blocklist가 필요하다.
- sensitive action이 어떤 mode에서든 confirmation gate를 우회하면 거부한다.

출력: `classifier.py`, `allowlist.py`, `confirmation.py`, `trace.py`, 그리고 gate policy, injection marker, allowlist maintenance process를 설명하는 `README.md`. 마지막에는 Lesson 27(prompt injection)과 Lesson 23(safety decision의 OTel span attribution)을 가리키는 "what to read next"로 끝낸다.
