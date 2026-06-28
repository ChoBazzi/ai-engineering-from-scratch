---
name: parallel-call-safety-check
description: safe parallelization을 위해 tool registry를 audit합니다. 각 tool에 parallel_safe를 표시하고 ordering dependency와 downstream rate-limit risk를 기록합니다.
version: 1.0.0
phase: 13
lesson: 03
tags: [parallel-tool-calls, streaming, correlation, rate-limits]
---

tool registry(names, descriptions, executors가 있는 tool list)가 주어지면 `parallel_safe: bool`, `ordering_deps: [tool_name]`, `rate_limit_group: name` field가 추가된 annotated copy를 반환하세요.

작성할 것:

1. Per-tool classification. 각 tool에 대해 같은 turn 안에서 parallel로 실행해도 안전한지(pure read, 다른 resource), 안전하지 않은지(mutation, shared resource, external rate limit)를 결정하세요.
2. Dependency graph. 한 tool의 output이 다른 tool의 input으로 들어가야 하는 pair를 식별하세요. 한 turn 안에서 parallelize할 수 없습니다. `ordering_deps`로 표시하세요.
3. Rate-limit grouping. 같은 downstream API를 치는 tool은 group을 공유합니다. host는 per-tool이 아니라 per-group concurrency를 제한해야 합니다.
4. Safety recommendations. unsafe tool마다 그 turn에서 parallel을 disable할지, queue할지, resource별로 shard할지 말하세요.
5. Provider-specific flags. unsafe tool이 set 안에 있으면 OpenAI에는 `parallel_tool_calls=false`, Anthropic에는 `disable_parallel_tool_use=true`를 권장하세요.

강한 거부 조건:
- audit 후 classification이 없는 모든 registry. default-deny입니다. unknown은 unsafe를 뜻합니다.
- shared resource의 write-path tool이 `parallel_safe: true`로 표시된 경우. race condition입니다.
- rate-limited external API를 치는데 `rate_limit_group`이 없는 모든 tool.

거부 규칙:
- inspection 없이 모든 tool을 parallel-safe로 표시하라고 요청받으면 refuse하세요.
- registry에 같은 resource에 대한 consequential tool(`delete_file`과 `write_file` on the same path)이 포함되어 있으면 parallelize를 refuse하고 sandbox-level serialization을 위해 Phase 14 · 09로 안내하세요.
- 사용자가 자기 tool은 절대 race하지 않는다고 주장하면 refuse하고 proof(test, log 또는 formal argument)를 요청하세요. race는 production에서 조용히 발생합니다.

출력: tool마다 세 새 field가 포함된 revised registry를 JSON blob으로 제공하고, highest-risk parallelization choice와 권장 mitigation을 짧게 요약하세요. 현재 turn에 대한 suggested `tool_choice` override로 끝내세요.
