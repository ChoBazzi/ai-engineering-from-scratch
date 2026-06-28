# OpenAI Agents SDK: 핸드오프, 가드레일, 트레이싱

> OpenAI Agents SDK는 Responses API 위에 구축된 경량 멀티 에이전트 프레임워크다. 다섯 가지 primitive는 Agent, Handoff, Guardrail, Session, Tracing이다. 핸드오프는 `transfer_to_<agent>`라는 이름의 도구다. 가드레일은 입력이나 출력에서 발동한다. 트레이싱은 기본으로 켜져 있다.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 minutes

## 학습 목표

- OpenAI Agents SDK의 다섯 가지 primitive를 말한다.
- 핸드오프를 설명한다. 왜 도구로 모델링되는지, 모델이 어떤 이름 형태를 보는지, 컨텍스트가 어떻게 전달되는지 설명한다.
- 입력 가드레일, 출력 가드레일, 도구 가드레일을 구분하고, `run_in_parallel`과 blocking 모드의 차이를 설명한다.
- 핸드오프, 가드레일, span 스타일 트레이싱을 갖춘 stdlib 런타임을 구현한다.

## 문제

깔끔하게 위임하지 못하는 에이전트는 결국 모든 것을 하나의 프롬프트에 밀어 넣는다. 가드레일이 없는 에이전트는 PII, 정책 위반 출력, 또는 끝나지 않는 루프를 배포하게 된다. OpenAI SDK는 멀티 에이전트 작업을 다룰 만하게 만드는 세 가지 primitive를 코드로 고정한다.

## 개념

### 다섯 가지 primitive

1. **Agent.** LLM + instructions + tools + handoffs.
2. **Handoff.** 다른 에이전트로 위임하는 것. 모델에는 `transfer_to_<agent_name>`이라는 이름의 도구로 표현된다.
3. **Guardrail.** 입력(첫 에이전트만), 출력(마지막 에이전트만), 또는 도구 호출(함수 도구별)에 대한 검증.
4. **Session.** 턴 사이의 대화 기록을 자동으로 유지한다.
5. **Tracing.** LLM 생성, 도구 호출, 핸드오프, 가드레일에 대한 내장 span.

### 도구로서의 핸드오프

모델은 도구 목록에서 `transfer_to_billing_agent`를 본다. 이를 호출하면 런타임은 다음을 수행하라는 신호로 해석한다.

1. 대화 컨텍스트를 복사한다. 또는 `nest_handoff_history` 베타 기능으로 압축한다.
2. 대상 에이전트를 해당 instructions로 초기화한다.
3. 대상 에이전트로 실행을 계속한다.

이는 Lesson 13 / Lesson 28의 supervisor 패턴을 제품화한 것이다.

### 가드레일

세 가지 종류가 있다.

- **입력 가드레일.** 첫 에이전트의 입력에서 실행된다. LLM 호출 전에 안전하지 않거나 범위를 벗어난 요청을 거부한다.
- **출력 가드레일.** 마지막 에이전트의 출력에서 실행된다. PII 유출, 정책 위반, 잘못된 형식의 응답을 잡는다.
- **도구 가드레일.** 함수 도구별로 실행된다. 인자를 검증하고, 권한을 확인하고, 실행을 감사한다.

모드:

- **Parallel**(기본값). 가드레일 LLM이 메인 LLM과 함께 실행된다. 꼬리 지연 시간이 낮다. 발동되면 메인 LLM의 작업은 폐기된다(토큰 낭비).
- **Blocking**(`run_in_parallel=False`). 가드레일 LLM이 먼저 실행된다. 발동되면 메인 호출에 토큰을 낭비하지 않는다.

Tripwire는 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`를 발생시킨다.

### 트레이싱

기본으로 켜져 있다. 모든 LLM 생성, 도구 호출, 핸드오프, 가드레일은 span을 내보낸다. `OPENAI_AGENTS_DISABLE_TRACING=1`은 이를 끈다. `add_trace_processor(processor)`는 OpenAI 쪽 백엔드와 함께 자체 백엔드로 span을 팬아웃한다.

### 세션

`Session`은 대화 기록을 백엔드(SQLite, Redis, custom)에 저장한다. `Runner.run(agent, input, session=session)`은 기록을 자동으로 로드하고 이어 붙인다.

### 이 패턴이 잘못되는 지점

- **핸드오프 drift.** Agent A가 Agent B로 넘기고, Agent B가 다시 Agent A로 넘긴다. hop counter를 추가하라.
- **가드레일 우회.** 도구 가드레일은 함수 도구에서만 발동한다. 내장 도구(파일 리더, 웹 fetch)에는 별도 정책이 필요하다.
- **과도한 트레이싱.** 민감한 내용이 span에 들어간다. OTel GenAI content-capture 규칙(Lesson 23)과 함께 사용하라. 외부에 저장하고 ID로 참조한다.

## 직접 만들기

`code/main.py`는 SDK의 형태를 stdlib로 구현한다.

- `Agent`, `FunctionTool`, `Handoff`(transfer 의미를 가진 함수 도구).
- 입력/출력/도구 가드레일, 핸드오프 dispatch, hop counter를 갖춘 `Runner`.
- trace 형태를 보여 주는 단순 span emitter.
- 사용자 질의에 따라 billing 또는 support로 핸드오프하는 triage 에이전트. 입력 하나는 가드레일을 발동시킨다.

실행:

```bash
python3 code/main.py
```

trace는 성공한 핸드오프 두 번, 입력 가드레일 발동 한 번, 실제 SDK가 내보내는 것과 닮은 span tree를 보여 준다.

## 활용하기

- **OpenAI Agents SDK**는 OpenAI 우선 제품에 사용한다.
- **Claude Agent SDK**(Lesson 17)는 Claude 우선 제품에 사용한다.
- **LangGraph**(Lesson 13)는 명시적 상태와 durable resume이 필요할 때 사용한다.
- **Custom**은 정확한 제어가 필요할 때 사용한다. 예: voice, multi-provider, federated deployment.

## 출시하기

`outputs/skill-agents-sdk-scaffold.md`는 triage 에이전트, 핸드오프, 입력/출력/도구 가드레일, 세션 저장소, trace processor를 갖춘 Agents SDK 앱을 scaffold한다.

## 연습

1. 핸드오프 hop counter를 추가하라. N번 transfer 뒤에는 거부한다. 동작을 trace로 확인하라.
2. `nest_handoff_history`를 옵션으로 구현하라. transfer 전에 이전 메시지를 하나의 요약으로 압축한다.
3. blocking 출력 가드레일을 작성하라. 발동될 프롬프트와 통과할 프롬프트에서 지연 시간을 비교하라.
4. `add_trace_processor`를 JSON logger에 연결하라. span마다 어떤 형태를 내보내는가?
5. SDK 문서를 읽어라. stdlib 장난감을 `openai-agents-python`으로 이식하라. 무엇을 잘못 모델링했는가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agent | "LLM + instructions" | SDK의 Agent 타입. 도구와 핸드오프를 소유한다 |
| Handoff | "Transfer" | 모델이 다른 에이전트로 위임하기 위해 호출하는 도구 |
| Guardrail | "Policy check" | 입력 / 출력 / 도구 호출에 대한 검증 |
| Tripwire | "Guardrail trip" | 가드레일이 거부할 때 발생하는 예외 |
| Session | "History store" | 실행 사이에 유지되는 대화 메모리 |
| Tracing | "Spans" | LLM + 도구 + 핸드오프 + 가드레일에 대한 내장 관측성 |
| Blocking guardrail | "Sequential check" | 가드레일이 먼저 실행된다. 발동 시 토큰 낭비가 없다 |
| Parallel guardrail | "Concurrent check" | 가드레일이 함께 실행된다. 지연 시간은 낮지만 발동 시 토큰을 낭비한다 |

## 더 읽을거리

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) - primitive, handoff, guardrail, tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) - Claude 스타일 counterpart
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - 애초에 언제 handoff를 쓸지
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - Agents SDK span이 매핑되는 표준
