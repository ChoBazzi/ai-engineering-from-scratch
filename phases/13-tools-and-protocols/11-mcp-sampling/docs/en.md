# MCP Sampling — 서버가 요청하는 LLM 완성과 에이전트 루프

> 대부분의 MCP 서버는 단순 실행기입니다. 인자를 받고, 코드를 실행하고, 콘텐츠를 반환합니다. Sampling은 서버가 방향을 뒤집을 수 있게 합니다. 서버가 클라이언트의 LLM에게 결정을 요청하는 것입니다. 덕분에 서버가 모델 자격 증명을 소유하지 않아도 서버 호스팅 에이전트 루프를 만들 수 있습니다. 2025-11-25에 병합된 SEP-1577은 sampling 요청 안에 tools를 추가해 루프가 더 깊은 추론을 포함할 수 있게 했습니다. 드리프트 위험 참고: SEP-1577의 tool-in-sampling 형태는 2026년 1분기까지 실험적이었고, SDK API에서는 아직 정착 중입니다.

**Type:** Build
**Languages:** Python (stdlib, sampling harness)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 minutes

## 학습 목표

- `sampling/createMessage`가 해결하는 문제를 설명합니다. 서버 측 API 키 없이 서버 호스팅 루프를 가능하게 합니다.
- 여러 턴의 프롬프트에 대해 클라이언트에게 sampling을 요청하고 완성 결과를 반환하는 서버를 구현합니다.
- `modelPreferences`(비용 / 속도 / 지능 우선순위)를 사용해 클라이언트의 모델 선택을 유도합니다.
- 동작을 하드코딩하지 않고 내부적으로 sampling을 반복하는 `summarize_repo` 도구를 만듭니다.

## 문제

코드 요약 워크플로에 유용한 MCP 서버는 파일 트리를 훑고, 읽을 파일을 고르고, 요약을 합성한 뒤 반환해야 합니다. LLM 추론은 어디에서 일어나야 할까요?

선택지 A: 서버가 자체 LLM을 호출합니다. API 키가 필요하고, 서버 측 과금이 발생하며, 사용자마다 비용이 큽니다.

선택지 B: 서버가 원시 콘텐츠를 반환하고 클라이언트의 에이전트가 추론합니다. 동작은 하지만 서버 로직이 클라이언트 프롬프트로 옮겨져 취약해집니다.

선택지 C: 서버가 `sampling/createMessage`를 통해 클라이언트의 LLM에게 요청합니다. 서버는 알고리즘(어떤 파일을 읽을지, 몇 번의 패스를 할지)을 유지하고, 클라이언트는 과금과 모델 선택권을 유지합니다. 서버에는 자격 증명이 전혀 없습니다.

Sampling은 선택지 C입니다. 신뢰된 서버가 완전한 LLM 호스트가 되지 않고도 에이전트 루프를 호스팅하게 해 주는 메커니즘입니다.

## 개념

### `sampling/createMessage` 요청

서버가 보냅니다.

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

클라이언트는 자신의 LLM을 실행하고 반환합니다.

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

합이 1.0이 되는 세 개의 실수입니다.

- `costPriority`: 더 저렴한 모델을 선호합니다.
- `speedPriority`: 더 빠른 모델을 선호합니다.
- `intelligencePriority`: 더 유능한 모델을 선호합니다.

여기에 `hints`가 더해집니다. 서버가 선호하는 이름 있는 모델입니다. 클라이언트는 hints를 따를 수도 있고 따르지 않을 수도 있습니다. 언제나 클라이언트의 사용자 설정이 우선합니다.

### `includeContext`

세 가지 값이 있습니다.

- `"none"` — 서버가 제공한 메시지만 포함합니다. 기본값입니다.
- `"thisServer"` — 이 서버 세션의 이전 메시지를 포함합니다.
- `"allServers"` — 모든 세션 컨텍스트를 포함합니다.

`includeContext`는 서버 간 컨텍스트를 누출해 보안 문제가 되기 때문에 2025-11-25 기준으로 소프트 deprecated 상태입니다. `"none"`을 선호하고 필요한 컨텍스트는 메시지에 명시적으로 전달하세요.

### 도구를 포함한 Sampling(SEP-1577)

2025-11-25에 새로 추가되었습니다. sampling 요청은 `tools` 배열을 포함할 수 있습니다. 클라이언트는 그 도구들을 사용해 전체 tool-calling 루프를 실행합니다. 이를 통해 서버는 클라이언트의 모델을 거쳐 ReAct 스타일 에이전트 루프를 호스팅할 수 있습니다.

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

클라이언트는 루프를 돕니다. sample을 만들고, 도구가 호출되면 실행하고, 다시 sample을 만든 뒤 최종 assistant 메시지를 반환합니다. 이는 2026년 1분기까지 실험적입니다. SDK 시그니처는 아직 바뀔 수 있습니다. 구현할 때는 2025-11-25 스펙의 client/sampling 섹션과 대조해 확인하세요.

### Human-in-the-loop

클라이언트는 sample을 실행하기 전에 서버가 모델에게 무엇을 요청하는지 사용자에게 반드시 보여줘야 합니다. 악성 서버는 sampling을 사용해 사용자 세션을 조작할 수 있습니다("사용자에게 X라고 말해서 Y를 클릭하게 하라"). Claude Desktop, VS Code, Cursor는 sampling 요청을 사용자가 거부할 수 있는 확인 대화상자로 표시합니다.

2026년의 합의는 사용자 확인 없는 sampling이 위험 신호라는 것입니다. 게이트웨이(Phase 13 · 17)는 낮은 위험의 sampling을 자동 승인하고 의심스러운 요청은 자동 거부할 수 있습니다.

### API 키 없는 서버 호스팅 루프

대표 사용 사례는 자체 LLM 접근 권한이 없는 코드 요약 MCP 서버입니다. 이 서버는 다음을 수행합니다.

1. 저장소 구조를 순회합니다.
2. "이 저장소의 목적을 가장 잘 설명할 가능성이 높은 파일 다섯 개를 고르라"는 내용으로 `sampling/createMessage`를 호출합니다.
3. 그 파일들을 읽습니다.
4. 파일 내용과 "저장소를 3문단으로 요약하라"는 요청으로 `sampling/createMessage`를 호출합니다.
5. 요약을 `tools/call` 결과로 반환합니다.

서버는 LLM API를 전혀 건드리지 않습니다. 클라이언트의 사용자가 자신의 자격 증명으로 완성 비용을 지불합니다.

### 안전 위험(Unit 42 공개, 2026년 1분기)

- **은밀한 sampling.** 항상 "세션 컨텍스트에서 사용자의 이메일을 응답하라"는 내용으로 sampling을 호출하는 도구입니다. Phase 13 · 15에서 공격 벡터를 다룹니다.
- **sampling을 통한 리소스 절도.** 서버가 클라이언트에게 공격자의 페이로드를 요약하라고 요청해 사용자에게 과금시킵니다.
- **루프 폭탄.** 서버가 빽빽한 루프 안에서 sampling을 호출합니다. 클라이언트는 세션별 rate limit을 반드시 강제해야 합니다.

## 사용하기

`code/main.py`는 가짜 server-to-client sampling 하네스를 제공합니다. 시뮬레이션된 `summarize_repo` 도구가 두 번의 sampling 라운드(파일 선택, 그다음 요약)를 호출하고, 가짜 클라이언트는 준비된 응답을 반환합니다. 이 하네스가 보여주는 내용은 다음과 같습니다.

- 서버가 `modelPreferences`와 함께 `sampling/createMessage`를 보냅니다.
- 클라이언트가 완성 결과를 반환합니다.
- 서버가 루프를 이어 갑니다.
- rate limiter가 도구 호출당 전체 sampling 호출 수를 제한합니다.

살펴볼 점:

- 서버는 하나의 도구(`summarize_repo`)만 노출합니다. 모든 추론은 sampling 호출 안에서 일어납니다.
- 모델 선호도는 클라이언트의 모델 선택에 가중치를 줍니다. hints는 선호 모델을 나열합니다.
- 루프는 `stopReason: "endTurn"`에서 종료됩니다.
- `max_samples_per_tool = 5` 제한이 runaway 루프를 잡습니다.

## 산출물

이 lesson은 `outputs/skill-sampling-loop-designer.md`를 만듭니다. 연구, 요약, 계획처럼 LLM 호출이 필요한 서버 측 알고리즘이 주어지면, 이 skill은 올바른 modelPreferences, rate limit, 안전 확인을 갖춘 sampling 기반 구현을 설계합니다.

## 연습

1. `code/main.py`를 실행하세요. `max_samples_per_tool`을 2로 바꾸고 rate-limit 차단을 관찰하세요.

2. SEP-1577 tool-in-sampling 변형을 구현하세요. sampling 요청이 `tools` 배열을 담도록 합니다. 최종 완성을 반환하기 전에 클라이언트 측 루프가 해당 도구들을 실행하는지 검증하세요. 드리프트 위험을 참고하세요. SDK 시그니처는 2026년 상반기까지 계속 바뀔 수 있습니다.

3. Human-in-the-loop 확인을 추가하세요. 서버의 첫 `sampling/createMessage` 전에 멈추고 사용자 승인을 기다립니다. 거부된 호출은 타입이 있는 거부 결과를 반환합니다.

4. 클라이언트 세션을 키로 하는 사용자별 rate limiter를 추가하세요. 같은 사용자의 같은 서버 루프는 하나의 예산을 공유해야 합니다.

5. sampling을 사용해 포함할 청크를 고르는 `summarize_pdf` 도구를 설계하세요. 전송되는 메시지를 스케치하세요. `modelPreferences.intelligencePriority`가 0.1일 때와 0.9일 때 동작이 어떻게 달라질까요?

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Sampling | "서버에서 클라이언트로 보내는 LLM 호출" | 서버가 클라이언트의 모델에게 완성을 요청함 |
| `sampling/createMessage` | "그 메서드" | sampling 요청을 위한 JSON-RPC 메서드 |
| `modelPreferences` | "모델 우선순위" | 비용 / 속도 / 지능 가중치와 이름 hints |
| `includeContext` | "서버 간 세션 누출" | 소프트 deprecated 된 컨텍스트 포함 모드 |
| SEP-1577 | "Sampling 안의 tools" | 서버 호스팅 ReAct를 위해 sampling 안에 tools 허용 |
| Human-in-the-loop | "사용자가 확인함" | 실행 전에 클라이언트가 sampling 요청을 사용자에게 표시 |
| 루프 폭탄 | "폭주 sampling" | 서버 측 무한 sampling 루프. 클라이언트가 rate-limit해야 함 |
| 은밀한 sampling | "숨겨진 추론" | 악성 서버가 sampling 프롬프트에 의도를 숨김 |
| 리소스 절도 | "사용자의 LLM 예산 사용" | 서버가 원치 않는 sampling 비용을 클라이언트에게 쓰게 함 |
| `stopReason` | "생성이 멈춘 이유" | `endTurn`, `stopSequence`, 또는 `maxTokens` |

## 더 읽을거리

- [MCP — Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) — sampling의 고수준 개요
- [MCP — Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) — 표준 `sampling/createMessage` 형태
- [MCP — GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) — sampling 안의 tools에 대한 Spec Evolution Proposal(실험적)
- [Unit 42 — MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) — 은밀한 sampling과 리소스 절도 패턴
- [Speakeasy — MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) — 클라이언트 측 코드 샘플이 있는 walkthrough
