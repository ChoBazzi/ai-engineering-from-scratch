---
name: stategraph-designer
description: Agent task를 named nodes, typed state, reducers, checkpointer, human interrupts를 갖춘 LangGraph StateGraph로 바꿉니다.
version: 1.0.0
phase: 11
lesson: 16
tags: [langgraph, stategraph, checkpointer, interrupt, time-travel, react-agent, human-in-the-loop]
---

Agent task(user-facing goal, available tools, expected turn count, safety blast radius가 있는 side effect, durability requirements, target latency budget)가 주어지면 다음을 출력하세요.

1. Node list. 모든 discrete step에 이름을 붙이세요. LLM thinker, 각 tool runner, 모든 human review step, summarizer나 critic, retriever가 포함됩니다. 한 node가 둘 이상의 concern을 건드리면 설계를 거부하고 쪼개세요.
2. State schema. 모든 list에 reducer가 있는 TypedDict(또는 Pydantic) field를 정의하세요. Message log에는 항상 Annotated[list, add_messages]를 사용하세요. Task-specific list(plan, budget counter, retrieved-docs list)는 messages 밖으로 끌어올려 parallel update에서도 reducer가 올바르게 유지되게 하세요.
3. Edge map. 다음 step이 deterministic이면 static edge를 쓰세요. Model이 다음 step을 고르는 곳에만 named router function이 있는 conditional edge를 쓰세요. Router function이 이전 node에서 이미 수행하지 않은 fresh LLM call에 의존하는 graph는 거부하세요.
4. Interrupt placement. 되돌릴 수 없는 side effect(write, delete, payment, 비용이 드는 external API call)가 있는 모든 node에는 interrupt_before를 두세요. Output validation이 별도 process에서 실행될 때 model node에는 interrupt_after를 둘 수 있습니다. Side-effecting node의 interrupt_after는 거부하세요. 그 시점에는 이미 side effect가 발생했습니다.
5. Checkpointer. MemorySaver는 test 전용입니다. Restart 이후에도 살아남아야 하는 환경에는 PostgresSaver, SQLiteSaver, RedisSaver 중 하나를 고르세요. thread_id strategy(per-user, per-session, per-conversation)와 checkpoint TTL을 확인하세요.

Checkpointer가 없는 LangGraph는 출시를 거부하세요. Checkpointer가 없다는 것은 resume도, time-travel도, human-in-the-loop replay도 없다는 뜻입니다. add_messages 없는 messages field도 거부하세요. 두 번째 write가 첫 번째를 조용히 덮어써 conversation의 절반이 사라집니다. 모든 transition이 planner LLM이 route하는 conditional edge인 graph도 거부하세요. 그것은 단계를 더 얹은 AutoGen이며 매 turn token을 태웁니다.

예시 입력: "Anthropic Claude 위의 refund-handling agent. Tool 세 개(lookup_order, issue_refund, send_email)를 사용하며, 100달러가 넘는 refund 전에는 human을 위해 반드시 pause해야 하고, server restart 후 resume되어야 하며, p95 latency budget은 8초입니다."

예시 출력:
- Nodes: agent(LLM call), lookup_tool, refund_tool, email_tool, human_review.
- State: add_messages를 쓰는 messages, order_context(overwrite), refund_amount(overwrite), reviewer_decision(overwrite).
- Edges: agent에서 lookup_tool, refund_tool, email_tool, human_review, END branch를 가진 should_continue router로 이동합니다. Tool node는 agent로 돌아갑니다.
- Interrupts: refund_amount > 100일 때 refund_tool에 interrupt_before를 둡니다. lookup_tool이나 email_tool에는 interrupt가 없습니다.
- Checkpointer: thread_id "user:{user_id}:case:{case_id}"와 30-day TTL을 쓰는 PostgresSaver.
