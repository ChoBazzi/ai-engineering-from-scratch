# Roots와 Elicitation — 범위 지정과 실행 중 사용자 입력

> 하드코딩된 경로는 사용자가 다른 프로젝트를 여는 순간 깨집니다. 미리 채운 도구 인자는 사용자가 덜 구체적으로 말할 때 깨집니다. Roots는 서버를 사용자가 제어하는 URI 집합으로 제한합니다. Elicitation은 도구 호출 중간에 멈춰 사용자에게 양식이나 URL로 구조화된 입력을 요청합니다. 두 개의 클라이언트 primitive가 MCP의 흔한 실패 모드 두 가지를 고칩니다. SEP-1036(URL-mode elicitation, 2025-11-25)은 2026년 상반기까지 실험적입니다. 의존하기 전에 SDK 버전을 확인하세요.

**Type:** Build
**Languages:** Python (stdlib, roots + elicitation demo)
**Prerequisites:** Phase 13 · 07 (MCP server)
**Time:** ~45 minutes

## 학습 목표

- `roots`를 선언하고 `notifications/roots/list_changed`에 응답합니다.
- 서버 파일 작업을 선언된 root 집합 안의 URI로 제한합니다.
- `elicitation/create`를 사용해 도구 호출 중간에 사용자에게 확인이나 구조화된 입력을 요청합니다.
- form-mode elicitation과 URL-mode elicitation 중 무엇을 쓸지 선택합니다. 후자는 실험적이며 드리프트 위험이 있습니다.

## 문제

프로덕션의 notes MCP 서버가 맞닥뜨리는 구체적인 실패 두 가지가 있습니다.

**깨진 경로 가정.** 서버가 `~/notes`를 기준으로 작성되어 있습니다. 다른 머신에서 `~/Documents/Notes`에 노트를 둔 사용자는 조용히 실패하는 도구 호출(파일을 찾지 못함)을 받거나, 더 나쁘게는 잘못된 위치에 쓰는 호출을 받습니다.

**사용자만 알 수 있는 누락 인자.** 사용자가 "오래된 TPS 보고서 노트를 삭제해"라고 요청합니다. 모델은 `notes_delete(title: "TPS report")`를 호출하지만 2023년, 2024년, 2025년의 일치 노트가 세 개 있습니다. 도구는 추측할 수 없습니다. "모호함"으로 실패하는 것은 성가시고, 셋 모두에 실행하는 것은 재앙입니다.

Roots는 첫 번째 문제를 고칩니다. 클라이언트가 `initialize` 시점에 서버가 건드릴 수 있는 URI 집합을 선언합니다. Elicitation은 두 번째 문제를 고칩니다. 서버가 도구 호출을 멈추고 `elicitation/create`를 보내 사용자에게 어느 항목인지 고르게 합니다.

## 개념

### Roots

클라이언트는 `initialize`에서 root 목록을 선언합니다.

```json
{
  "capabilities": {"roots": {"listChanged": true}}
}
```

그러면 서버는 `roots/list`를 호출할 수 있습니다.

```json
{"roots": [{"uri": "file:///Users/alice/Documents/Notes", "name": "Notes"}]}
```

서버는 roots를 경계로 취급해야 합니다. root 집합 밖의 모든 파일 읽기나 쓰기는 거부됩니다. 이것이 클라이언트에 의해 강제되는 것은 아닙니다. 서버는 여전히 사용자가 신뢰한 코드입니다. 하지만 스펙을 준수하는 서버는 이를 지킵니다.

사용자가 root를 추가하거나 제거하면 클라이언트는 `notifications/roots/list_changed`를 보냅니다. 서버는 `roots/list`를 다시 호출하고 경계를 갱신합니다.

### Roots가 클라이언트 primitive인 이유

Roots는 사용자의 동의 모델을 나타내므로 클라이언트가 선언합니다. 사용자는 Claude Desktop에게 "이 notes 서버에 이 두 디렉터리 접근 권한을 줘"라고 말했습니다. 서버는 그 범위를 넓힐 수 없습니다.

### Elicitation: 기본 form mode

`elicitation/create`는 양식 스키마와 자연어 프롬프트를 받습니다.

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Delete 'TPS report'? Multiple notes match; pick one.",
    "requestedSchema": {
      "type": "object",
      "properties": {
        "note_id": {
          "type": "string",
          "enum": ["note-3", "note-7", "note-14"]
        },
        "confirm": {"type": "boolean"}
      },
      "required": ["note_id", "confirm"]
    }
  }
}
```

클라이언트는 양식을 렌더링하고 사용자의 답을 수집해 반환합니다.

```json
{
  "action": "accept",
  "content": {"note_id": "note-14", "confirm": true}
}
```

가능한 action은 세 가지입니다. `accept`(사용자가 입력함), `decline`(사용자가 닫음), `cancel`(사용자가 전체 도구 호출을 중단함)입니다.

양식 스키마는 평평합니다. v1에서는 중첩 객체가 지원되지 않습니다. SDK는 일반적으로 한 단계보다 복잡한 스키마를 거부합니다.

### Elicitation: URL mode(SEP-1036, 실험적)

2025-11-25에 새로 추가되었습니다. 서버는 스키마 대신 URL을 보냅니다.

```json
{
  "method": "elicitation/create",
  "params": {
    "message": "Sign in to GitHub",
    "url": "https://github.com/login/oauth/authorize?client_id=..."
  }
}
```

클라이언트는 브라우저에서 URL을 열고 완료를 기다린 뒤 사용자가 돌아오면 반환합니다. 양식으로는 부족한 OAuth 흐름, 결제 승인, 문서 서명에 유용합니다.

드리프트 위험 참고: SEP-1036 응답 형태는 아직 정착 중입니다. 일부 SDK는 callback URL을 반환하고, 다른 SDK는 completion token을 반환합니다. 프로덕션에서 URL mode를 사용하기 전에 SDK 릴리스 노트를 읽으세요.

### Elicitation이 맞는 도구인 경우

- 파괴적 작업 전 사용자 확인(destructive hint + elicitation).
- 모호성 해소(N개 후보 중 하나 선택).
- 최초 실행 설정(API 키, 디렉터리, 선호 설정).
- OAuth 스타일 흐름(URL mode).

### Elicitation이 잘못된 경우

- 모델이 대화로 물어볼 수 있었던 도구의 필수 인자를 채우는 경우. elicitation 대화상자가 아니라 일반 재질문을 사용하세요.
- 고빈도 호출. Elicitation은 대화를 끊습니다. 루프 안에서 발생시키지 마세요.
- 서버가 사후에 검증할 수 있는 모든 것. 검증하고 오류를 반환한 뒤 모델이 텍스트로 사용자에게 묻게 하세요.

### Human-in-the-loop 연결

Elicitation과 sampling을 함께 쓰면 MCP의 "human-in-the-loop" 모델이 가능해집니다. 서버의 에이전트 루프는 사용자 입력(elicitation)이나 모델 추론(sampling)을 위해 멈출 수 있습니다. Phase 13 · 11은 sampling을 다뤘고, 이 lesson은 elicitation을 다룹니다. 둘을 합치면 루프 중간 제어가 완성됩니다.

## 사용하기

`code/main.py`는 notes 서버를 다음 기능으로 확장합니다.

- root-list-changed 알림 뒤 서버가 다시 질의하는 `roots/list` 응답.
- 여러 노트가 일치할 때 `elicitation/create`로 모호성을 해소하는 `notes_delete` 도구.
- 최초 실행 설정 페이지를 여는 URL-mode elicitation을 사용하는 `notes_setup` 도구(시뮬레이션).
- 선언된 roots 밖의 URI에 대한 작업을 거부하는 경계 검사.

데모는 세 가지 시나리오를 실행합니다. 정상 경로(하나의 일치), 모호성 해소(세 개의 일치와 elicitation 발생), root 밖 쓰기(거부)입니다.

## 산출물

이 lesson은 `outputs/skill-elicitation-form-designer.md`를 만듭니다. 사용자 확인이나 모호성 해소가 필요할 수 있는 도구가 주어지면, 이 skill은 elicitation 양식 스키마와 메시지 템플릿을 설계합니다.

## 연습

1. `code/main.py`를 실행하세요. 모호성 해소 경로를 트리거하고, 시뮬레이션된 사용자 답변이 도구로 다시 라우팅되는지 확인하세요.

2. 매번 elicitation 확인이 필요한 새 도구 `notes_archive`를 추가하세요(destructive hint). UX를 확인하세요. 모델이 텍스트로 다시 묻는 것과 어떻게 다른가요?

3. 최초 실행 OAuth 흐름을 위해 URL-mode elicitation을 구현하세요. 드리프트 위험을 적고 SDK 버전 guard를 추가하세요.

4. `roots/list` 처리를 확장하세요. 알림이 도착하면 서버는 지금 범위 밖이 되었을 수 있는 열린 파일 핸들을 원자적으로 다시 읽고 다시 스캔해야 합니다.

5. GitHub의 SEP-1036 이슈 토론 스레드를 읽으세요. 서버가 URL-mode callback을 처리하는 방식에 영향을 주는 열린 질문 하나를 찾으세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|----------------|-----------|
| Root | "동의 경계" | 클라이언트가 서버의 접근을 허용한 URI |
| `roots/list` | "서버가 범위를 요청함" | 클라이언트가 현재 root 집합을 반환함 |
| `notifications/roots/list_changed` | "사용자가 범위를 바꿈" | 클라이언트가 root 집합의 변화를 알림 |
| Elicitation | "호출 중간에 사용자에게 묻기" | 서버가 시작하는 구조화된 사용자 입력 요청 |
| `elicitation/create` | "그 메서드" | elicitation 요청을 위한 JSON-RPC 메서드 |
| Form mode | "스키마 기반 양식" | 클라이언트 UI에서 양식으로 렌더링되는 평평한 JSON Schema |
| URL mode | "브라우저 리디렉션" | SEP-1036 실험 기능. URL을 열고 기다림 |
| `accept` / `decline` / `cancel` | "사용자 응답 결과" | 서버가 처리하는 세 가지 분기 |
| 모호성 해소 | "하나 고르기" | 도구에 N개 후보가 있을 때의 일반적인 elicitation 사용 사례 |
| 평평한 양식 | "최상위 속성만" | Elicitation 스키마는 중첩할 수 없음 |

## 더 읽을거리

- [MCP — Client roots spec](https://modelcontextprotocol.io/specification/draft/client/roots) — 표준 roots 참고 문서
- [MCP — Client elicitation spec](https://modelcontextprotocol.io/specification/draft/client/elicitation) — 표준 elicitation 참고 문서
- [Cisco — What's new in MCP elicitation, structured content, OAuth enhancements](https://blogs.cisco.com/developer/whats-new-in-mcp-elicitation-structured-content-and-oauth-enhancements) — 2025-11-25 추가 사항 walkthrough
- [MCP — GitHub SEP-1036](https://github.com/modelcontextprotocol/modelcontextprotocol) — URL-mode elicitation 제안(실험적, 드리프트 위험)
- [The New Stack — How elicitation brings human-in-the-loop to AI tools](https://thenewstack.io/how-elicitation-in-mcp-brings-human-in-the-loop-to-ai-tools/) — UX walkthrough
