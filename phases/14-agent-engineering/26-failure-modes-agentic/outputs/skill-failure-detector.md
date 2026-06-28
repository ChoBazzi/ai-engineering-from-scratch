---
name: failure-detector
description: trace store에 연결되어 다섯 industry-recurring mode와 domain-specific signature를 tag하는 agent trace용 failure-mode detector를 생성한다.
version: 1.0.0
phase: 14
lesson: 26
tags: [failure-modes, masft, detection, observability]
---

product domain과 trace store가 주어지면 agent failure mode detector를 생성한다.

생성할 것:

1. mode별 detector: `hallucinated_action`, `scope_creep`, `cascading_errors`, `context_loss`, `tool_misuse`, `success_hallucination`.
2. domain-specific detector(예: dev tool의 "issue link 없이 PR 생성", marketing tool의 "확인 없이 > 5 recipients에게 email 발송").
3. 모든 detector를 각 trace에 적용하고 distribution을 emit하는 tagger.
4. threshold-based alerting: 오늘 trace의 >=5%가 mode에 tag되면 page하거나 ticket을 연다.
5. sample retention: tag된 각 trace에 대해 operator review용 input + output + state snapshot을 보관한다.

강한 거부 조건:

- production에서 trace마다 LLM call이 필요한 detector. pattern-based detector를 사용하고 LLM-judge는 sampled review에 남겨 둔다.
- crash에서만 tag함. 대부분의 failure는 valid-looking output을 만든다. content + state에 대한 signature check가 필요하다.
- PII redaction 없이 tagged trace 저장. failure sample은 최악의 content를 담으므로 저장 전에 scrub한다.

거부 규칙:

- 사용자가 "all traces stored forever"를 원하면 cost + compliance 이유로 거부한다. tag + rate 기준으로 sample한다.
- product에 "known good" baseline이 없으면 drift alert를 거부한다. drift에는 reference가 필요하다.
- detector가 versioned가 아니면 거부한다. detector regression은 예고 없이 signal을 망가뜨린다.

출력: threshold, retention policy, alert routing을 설명하는 `detectors.py`, `tagger.py`, `alerts.py`, `retention.py`, `README.md`. 마지막은 Lesson 24(observability backends) 또는 adversarial failure mode를 위한 Lesson 27(prompt injection)을 가리키는 "what to read next"로 끝낸다.
