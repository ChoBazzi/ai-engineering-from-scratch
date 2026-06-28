# CrewAI: 역할 기반 Crew와 Flow

> CrewAI는 2026년의 역할 기반 멀티 에이전트 프레임워크다. 네 가지 primitive는 Agent, Task, Crew, Process다. 최상위 형태는 둘이다. Crews는 자율적이고 역할 기반인 협업이며, Flows는 event-driven이고 deterministic하다. 문서는 직설적이다. "production-ready application이라면 Flow로 시작하라."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 minutes

## 학습 목표

- CrewAI의 네 가지 primitive(Agent, Task, Crew, Process)와 각각이 소유하는 것을 말한다.
- Sequential, Hierarchical, 계획된 Consensus process를 구분하고 workload마다 하나를 고른다.
- Crews(자율적 역할 기반)와 Flows(event-driven deterministic)를 구분하고, 문서의 production 권장사항을 설명한다.
- `@tool` decorator와 `BaseTool` subclass로 도구를 연결하고 structured output과 free text의 차이를 추론한다.
- CrewAI의 네 가지 memory type과 각각이 언제 값을 하는지 말한다.
- brief를 만드는 stdlib three-agent crew(researcher, writer, editor)를 구현한다.
- 세 가지 CrewAI failure mode를 찾아낸다. prompt-bloat, manager-LLM tax, brittle handoff.

## 문제

멀티 에이전트 프레임워크를 도입하는 팀은 같은 벽에 부딪힌다. demo에서 "autonomous collaboration"은 멋져 보인다. 그러다 고객이 bug를 제보하고 deterministic replay가 필요해진다. 또는 finance가 LLM-routed crew의 run당 비용을 묻는다. 또는 on-call이 새벽 3시에 어떤 agent가 멈췄는지 알아야 한다.

free-form LLM-routed crew는 이 질문에 깔끔하게 답하지 못한다. 순수 DAG는 모두 답하지만 brainstorming agent에 필요한 탐색적 형태를 잃는다.

CrewAI의 분리는 이 trade-off에 정직하다. 협업적이고 역할 기반이며 탐색적인 작업에는 Crew. event-driven이고 code-owned이며 auditable한 production에는 Flow. 같은 프레임워크 안의 두 형태이며, surface마다 고른다.

## 개념

### 네 가지 primitive

CrewAI의 표면은 작다. 이것을 외우면 나머지는 config다.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. backstory는 load-bearing이다. tone, judgment, agent가 멈추는 시점을 형성한다. tool은 agent가 호출할 수 있는 함수다(아래에서 더 다룬다).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. 재사용 가능한 work unit. `expected_output`이 contract다. `context`는 출력이 전달되는 upstream task를 나열한다. `output_pydantic`은 structured shape를 강제한다.
- **Crew.** container. `agents` 목록, `tasks` 목록, `process`, 선택적 `memory` + `verbose` + `manager_llm` 설정을 소유한다.
- **Process.** execution strategy. Sequential, Hierarchical, Consensus(계획됨). run의 형태를 고른다.

agent는 서로를 직접 보지 않는다. task가 agent를 참조한다. Crew가 task를 순서대로 실행한다. Process가 다음 task를 누가 고르는지 결정한다. 이것이 전체 mental model이다.

> **Validated against** CrewAI 0.86 (2026-05). 새 version은 process type의 이름을 바꾸거나 병합할 수 있다. 특정 형태에 의존하기 전에 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)를 확인하라.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** task가 선언 순서대로 실행된다. task N의 출력은 task N+1의 `context`로 사용할 수 있다. 비용이 가장 낮고 가장 예측 가능하다. 순서가 고정되어 있을 때 사용한다.
- **Hierarchical.** manager Agent(별도 LLM call)가 specialist 사이를 route한다. CrewAI는 `manager_llm` config 또는 default에서 manager를 생성한다. manager는 매 round마다 다음 task를 고르고 거부하거나 다시 route할 수 있다. specialist가 4명 이상이고 순서가 실제로 이전 output에 의존할 때 사용한다.
- **Consensus.** 계획된 기능이며 현재 public API에는 구현되어 있지 않다. 문서는 향후 voting-based process를 위해 이 이름을 예약한다. 오늘은 의존하지 말라.

Hierarchical은 모든 specialist call 위에 round마다 LLM call(manager)을 하나 더 추가한다. 다섯 step run에서는 token cost가 세 배가 될 수 있다. routing이 필요할 때만 비용을 지불하라.

### Crew vs Flow

이것이 2026년 문서의 핵심 framing이다.

- **Crew.** LLM-driven autonomy. 프레임워크가 runtime에 형태를 고른다. research, brainstorming, first draft처럼 path 자체가 답의 일부인 곳에 좋다. replay가 어렵다. test가 어렵다. prototype은 싸다.
- **Flow.** 사용자가 소유하는 event-driven graph. `@start`는 entry를 표시한다. `@listen(topic)`은 다른 step이 해당 topic을 emit할 때 실행되는 step을 표시한다. 각 step은 plain Python이다(내부에서 Crew를 호출할 수 있다). production에 좋다. Observable, testable, deterministic하다.

문서의 2026 production 권장사항: Flow로 시작하라. autonomy가 비용을 정당화할 때 Flow step 안에서 `Crew.kickoff()` call로 Crew를 접어 넣어라. Flow는 audit trail을 주고, Crew는 exploration을 준다. 하나만 고르지 말고 조합하라.

### 도구 통합

Agent에 도구를 제공하는 방법은 세 가지다. 맞는 것 중 가장 단순한 방식을 고른다.

1. **`@tool` decorator.** 순수 함수가 도구가 된다. signature는 schema이고, docstring은 LLM이 보는 description이다. one-off helper에 가장 좋다.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` subclass.** 명시적 args schema, async support, retry를 가진 class-based tool. 도구가 state(client, cache)를 가지거나 structured args가 필요할 때 사용한다.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Built-in toolkits.** CrewAI는 first-party adapter를 제공한다. `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. import 하나로 연결된다.

structured output은 Pydantic을 사용한다. Task에 `output_pydantic=MyModel`을 넘긴다. CrewAI는 LLM response를 model에 대해 validate하고 coerce하거나 retry한다. 이를 타이트한 `expected_output` 문자열과 함께 사용하라. draft에는 free-text output도 괜찮다. downstream Flow가 소비할 것은 structured output이다.

### Memory hook

CrewAI는 기본으로 네 가지 memory type을 제공한다. 이들은 조합된다. Crew 하나가 네 가지를 모두 동시에 enable할 수 있다.

> **Validated against** CrewAI 0.86 (2026-05). 최근 release는 이 네 store를 감싸는 통합 `Memory` system으로 모든 것을 route한다. 아래 개념 모델은 여전히 유효하지만, 새 version에서는 public class surface가 단일 `Memory` entry-point로 접힐 수 있다. 현재 API는 [CrewAI memory docs](https://docs.crewai.com/concepts/memory)를 확인하라.

- **Short-term.** 단일 run 안의 conversation buffer. 끝나면 지워진다.
- **Long-term.** run 사이에 persist된다. vector DB에 저장된다(기본 Chroma, 교체 가능). 현재 task와의 similarity로 retrieve된다.
- **Entity.** entity별 fact. "Customer X is on the enterprise plan." similarity가 아니라 entity로 keying된다. run 사이에 살아남는다.
- **Contextual.** assembly-time retrieval. 미리 load하지 않고 Agent가 필요로 하는 순간 관련 memory를 가져온다.

Crew에서 `memory=True` 또는 type별 config로 enable한다. 설정한 embeddings provider가 뒷받침한다(기본 OpenAI, local로 교체 가능). Memory는 CrewAI가 더 얇은 framework에 비해 값을 하는 영역 중 하나다. 순수 LangGraph에서는 이 각각을 직접 연결해야 한다.

### CrewAI가 맞는 경우

- 이름 붙은 역할과 협업 workflow를 가진 agent 3-6개. drafting, reviewing, planning, brainstorming.
- 다음 step에 대한 LLM의 판단이 가치의 일부인 routing(Hierarchical).
- graph definition보다 `role + goal + backstory`를 읽는 것이 팀에 더 편한 곳.

### CrewAI가 맞지 않는 경우

- 엄격한 순서를 가진 deterministic DAG. LangGraph(Lesson 13)를 사용하라. graph shape가 올바른 abstraction이며 CrewAI의 role framing은 마찰이다.
- sub-second latency budget. Hierarchical은 round trip을 추가한다. Sequential도 backstory와 prior output을 포함하는 prompt를 직렬화한다.
- single-agent loop. framework를 건너뛰어라. agent loop(Lesson 1)와 tool registry가 더 짧다.

Lesson 17(Agent Framework Tradeoffs)은 이를 matrix로 정리한다. 짧게 말하면 CrewAI는 "collaborative role-based" corner에 있다.

### Dependency shape

LangChain과 독립적이다. Python 3.10부터 3.13까지. `uv`를 사용한다. star count는 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)를 보라(2026-05 snapshot). AWS Bedrock integration이 문서화되어 있다. vendor benchmark는 QA workload에서 LangGraph 대비 상당한 speedup을 보고하지만 methodology(dataset, hardware, evaluation metric)가 공개되어 있지 않으므로 framework-vendor 숫자는 방향성으로만 취급하라.

### 이 패턴이 잘못되는 지점

- **Backstory로 인한 prompt-bloat.** agent마다 2000단어 backstory가 있고 5-agent crew라면 첫 tool call 전에 context budget을 태운다. backstory는 200단어 아래로 유지하라. agent 사이에 phrase를 재사용하고, house style을 다섯 번 반복하지 말라.
- **Manager-LLM token tax.** Hierarchical process는 모든 specialist call 전에 manager LLM call을 추가한다. 5-task crew에서는 LLM call이 5번이 아니라 6번이고, manager call은 전체 task list와 prior output을 들고 간다. routing이 output에 의존하지 않으면 Sequential로 바꿔라.
- **Brittle handoff.** Task N의 `expected_output`은 "an outline"이다. Task N+1은 이를 `context`로 읽고 section 세 개를 parse하려 한다. LLM은 네 개를 만들었다. downstream Agent가 즉흥적으로 처리한다. Task N에 `output_pydantic`을 붙여 Task N+1이 free text가 아니라 typed object를 읽게 하라.
- **Crew-as-prod.** Flow wrapper 없이 free-form Crew를 production에 배포했다. output variability가 높고, replay가 불가능하며, on-call은 나쁜 run과 좋은 run을 diff할 수 없다. Flow로 감싸라.

## 직접 만들기

`code/main.py`는 두 형태의 stdlib version과 three-agent crew를 구현한다.

형태:

- CrewAI 표면과 맞춘 `Agent`, `Task` dataclass.
- `SequentialCrew.kickoff(inputs)`는 선언 순서대로 task를 실행하고 output을 `context`로 threading한다.
- `HierarchicalCrew.kickoff(topic)`은 manager Agent가 매 round마다 다음 specialist를 고르게 하고, "done"에서 멈춘다.
- `@start`와 `@listen(topic)` decorator, 작은 event loop, trace를 가진 `Flow`.
- CrewAI의 `@tool` 형태를 mirror하는 `tool(name)` decorator.
- `short_term`, `long_term`, `entity` store를 가진 `Memory`. mock similarity는 numpy를 사용한다.
- mock LLM response는 role과 input prefix를 key로 하는 hardcoded string이다. network 없음. deterministic하다.

구체 demo: "agent engineering 2026"에 대한 brief를 만드는 researcher, writer, editor crew. Researcher는 (mocked) source를 가져온다. Writer가 draft를 만든다. Editor가 다듬는다. 같은 crew를 Flow로도 실행해 deterministic shape를 보여 준다.

실행:

```bash
python3 code/main.py
```

trace가 다루는 것: sequential crew가 `context`를 통해 output을 threading하는 모습, manager pick(researcher, writer, editor, 이후 "done")을 가진 hierarchical crew, 명시적 topic(`researched`, `drafted`, `edited`)으로 같은 세 step을 실행하는 flow, `@tool`을 통해 route되는 tool call, 두 kickoff 사이에 살아남는 long-term memory.

Crew trace는 유동적이다. manager는 원칙적으로 순서를 바꿀 수 있다. Flow trace는 고정되어 있다. 이 선택이 lesson의 핵심이다.

## 활용하기

- production에는 **CrewAI Flow**. Flow가 `Crew.kickoff()`를 호출하는 한 step뿐이어도 그렇다. Flow가 audit boundary를 준다.
- 명확한 순서의 collaborative work, 특히 first draft와 review loop에는 **CrewAI Crew(Sequential)**.
- routing이 output에 의존하고 specialist가 4명 이상이면 **CrewAI Crew(Hierarchical)**.
- 명시적 state machine, durable resume, strict ordering에는 **LangGraph**(Lesson 13).
- actor-model concurrency와 fault isolation에는 **AutoGen v0.4**(Lesson 14).
- handoff와 guardrail이 있는 OpenAI 우선 제품에는 **OpenAI Agents SDK**(Lesson 16).
- subagent와 session store가 있는 Claude 우선 제품에는 **Claude Agent SDK**(Lesson 17).

## 출시하기

`outputs/skill-crew-or-flow.md`는 task에 대해 Crew와 Flow 중 하나를 고르고 최소 구현을 scaffold한다. backstory 없는 Crew, 명시적 topic 없는 Flow, specialist가 3명 미만인 Hierarchical은 hard reject한다.

## 함정

- **Flavor로서의 backstory.** backstory는 output을 형성한다. agent마다 세 가지 variant를 test하라. variance는 실제다. 하나를 고르고 freeze하라.
- **`expected_output` 생략.** task별 contract가 없으면 downstream task는 LLM이 만든 무엇이든 집어 든다. Crew는 실행되지만 audit는 실패한다.
- **항상 켜진 memory.** long-term은 매 run write한다. vector DB가 커진다. retrieval이 noisy해진다. fact가 persistent한 task로 write scope를 제한하라.
- **Manager prompt drift.** Hierarchical의 manager prompt는 implicit하다. routing이 이상하면 verbose mode에서 dump하고 읽어라.
- **Crew 안의 tool side effect.** Crew는 예상보다 많이 tool을 호출할 수 있다. POST, DELETE, payment는 Crew tool이 아니라 반드시 Flow step에 있어야 한다.

## 연습

1. Sequential crew를 Flow로 변환하라. variability가 줄어드는 touchpoint를 세어라. readability가 떨어진 지점을 기록하라.
2. crew에 entity memory를 추가하라. customer에 대한 fact가 kickoff 사이에 persist된다. retrieval이 올바른 entity를 가져오는지 확인하라.
3. writer output이 최소 세 paragraph가 되기 전까지 manager가 editor로 route하기를 거부하는 Hierarchical process를 구현하라. retry를 trace하라.
4. (mocked) web search용 `BaseTool` subclass를 연결하라. trace shape를 `@tool` decorator version과 비교하라.
5. editor task에 `output_pydantic=Brief`를 추가하라. `Brief`에는 `title`, `summary`, `sections`가 있다. writer task가 한 번 malformed JSON을 output하게 만들고, trace에서 CrewAI의 retry behavior를 확인하라.
6. CrewAI docs intro를 읽어라. toy를 실제 `crewai` API로 이식하라. stdlib version은 어떤 guarantee를 생략했는가?
7. AgentOps 또는 Langfuse(Lesson 24)를 실제 run에 연결하라. stdlib version에서 놓친 trace는 무엇인가?

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|----------------|------------------------|
| Agent | "Persona" | Role + goal + backstory + tools |
| Task | "Unit of work" | Description + expected output + assignee + optional structured output |
| Crew | "Agent team" | Agents + Tasks + Process를 담는 container |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensus(계획됨) |
| Flow | "Deterministic workflow" | Event-driven, code-owned, testable |
| Backstory | "Persona prompt" | Agent의 tone과 judgment를 형성한다 |
| `@tool` | "Function tool" | 함수를 Agent가 호출할 수 있는 도구로 바꾸는 decorator |
| `BaseTool` | "Class tool" | args schema, retry, async support를 가진 class-based tool |
| Entity memory | "Per-entity facts" | customer / account / issue에 scope가 잡힌 memory |
| Long-term memory | "Cross-run memory" | kickoff 사이에 살아남는 vector-backed memory |
| Contextual memory | "Just-in-time retrieval" | Agent가 필요로 하는 순간 가져오는 memory |
| Manager LLM | "Router agent" | Hierarchical process에서 다음 task를 고르는 추가 LLM |
| `expected_output` | "Task contract" | Agent와 audit에 어떤 shape를 반환해야 하는지 알려 주는 문자열 |

## 더 읽을거리

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction) - 개념과 권장 production path
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows) - event-driven shape, `@start`, `@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools) - `@tool`, `BaseTool`, built-in toolkits
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory) - short-term, long-term, entity, contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - multi-agent가 도움이 되는 때와 그렇지 않은 때
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) - state-machine alternative
