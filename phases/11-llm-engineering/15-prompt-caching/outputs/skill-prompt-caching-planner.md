---
name: prompt-caching-planner
description: Cache-friendly prompt layout을 설계하고 알맞은 provider caching mode를 선택합니다.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Prompt(system + tools + few-shot + retrieval + history + user)와 usage profile(requests per hour, 필요한 TTL, provider)이 주어지면 다음을 출력하세요.

1. Layout. Section을 재정렬하고 단일 cache breakpoint를 표시하세요. 어떤 section이 stable이고 어떤 section이 volatile인지 설명하세요.
2. Provider mode. Anthropic cache_control, OpenAI automatic, Gemini CachedContent 중 하나를 선택하세요. TTL과 reuse pattern을 근거로 정당화하세요.
3. Break-even. TTL 안에서 write당 예상 read 수와 no-cache 대비 net cost를 수식으로 보여 주세요.
4. Verification plan. 두 번째 identical request에서 cache_read_input_tokens > 0임을 확인하는 CI assertion과 cached vs uncached token을 나눈 dashboard를 제시하세요.
5. Failure modes. 이 setup에서 cache miss가 날 가능성이 가장 높은 이유 세 가지(dynamic timestamp, tool reorder, near-duplicate text)와 각각을 막는 방법을 나열하세요.

Dynamic field를 breakpoint 위에 두는 cache plan은 출시를 거부하세요. 2x write premium을 회수할 reuse count 없이 1h TTL을 활성화하는 것도 거부하세요.
