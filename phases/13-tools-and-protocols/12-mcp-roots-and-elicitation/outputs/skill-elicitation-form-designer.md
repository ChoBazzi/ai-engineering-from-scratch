---
name: elicitation-form-designer
description: 호출 중 사용자 확인이나 모호성 해소가 필요한 도구를 위한 elicitation form schema와 message template을 설계합니다.
version: 1.0.0
phase: 13
lesson: 12
tags: [mcp, elicitation, user-input, forms]
---

동작 중 호출 중간 사용자 입력이 필요할 수 있는 도구가 주어지면, elicitation schema와 message를 설계하세요.

다음을 산출하세요.

1. Trigger condition. 도구가 `elicitation/create`를 호출해야 하는 정확한 입력이나 모호성을 명시하세요.
2. Message template. 호스트가 사용자에게 보여줄 한 문장입니다. 평이하고, 구체적이며, 전문 용어가 없어야 합니다.
3. Schema. typed properties와 `enum` 목록(모호성 해소) 또는 `boolean`(확인)을 담은 평평한 JSON Schema입니다. 중첩하지 마세요.
4. Branch handling. `accept` / `decline` / `cancel`을 도구 동작에 매핑하세요.
5. Rate-limit rule. 도구 호출당 elicitation 수를 제한하세요. 루프 안에서 elicitation하지 마세요.

강한 거부 조건:
- 객체를 중첩하는 모든 schema. Elicitation v1은 평평합니다.
- LLM이 대화로 물어볼 수 있었던 누락 인자를 채우기 위해 쓰는 모든 elicitation.
- 고빈도 elicitation(도구 호출당 한 번 초과).

거부 규칙:
- 도구가 read-only이고 low-risk라면 elicitation을 거부하고 결과만 반환하세요.
- 도구가 파괴적이고 호스트가 `destructiveHint` annotation을 지원한다면, annotation을 사용하고 클라이언트가 native confirmation을 처리하게 하라고 제안하세요.
- 필요한 것이 OAuth sign-in이라면 URL-mode elicitation을 권장하고 SEP-1036 드리프트 위험을 표시하세요.

출력: trigger condition, message template, schema, branch handling, rate-limit rule, form mode와 URL mode 중 무엇이 더 맞는지에 대한 note가 담긴 한 페이지 설계.
