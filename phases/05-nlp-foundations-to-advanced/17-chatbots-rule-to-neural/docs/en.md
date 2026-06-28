# 챗봇 — Rule-Based에서 Neural, LLM Agent까지

> ELIZA는 pattern match로 답했다. DialogFlow는 intent를 매핑했다. GPT는 weight에서 답을 만들었다. Claude는 tool을 실행하고 검증한다. 각 시대는 이전 시대의 가장 큰 실패를 해결했다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 minutes

## 문제

사용자가 "항공편을 변경하고 싶어요"라고 말한다. 시스템은 사용자가 무엇을 원하는지, 어떤 정보가 빠졌는지, 그 정보를 어떻게 얻을지, 그리고 동작을 어떻게 완료할지 알아내야 한다. 그런 다음 사용자가 "잠깐, 대신 취소하면 어떻게 돼요?"라고 말하면, 시스템은 context를 기억하고 task를 전환하며 state를 보존해야 한다.

대화는 ML 시스템에 어렵다. input은 open-ended다. output은 여러 turn에 걸쳐 일관되어야 한다. 시스템은 실제 세계에 action을 취해야 할 수도 있다(항공편 변경, 카드 결제). 잘못된 모든 단계는 사용자에게 바로 보인다.

Chatbot architecture는 네 가지 paradigm을 거쳐 왔고, 각각은 이전 방식의 실패가 너무 눈에 띄었기 때문에 등장했다. 이 lesson은 그 흐름을 순서대로 따라간다. 2026년 production 환경은 마지막 두 방식의 hybrid다.

## 개념

![Chatbot evolution: rule-based에서 retrieval, neural, agent로](../assets/chatbot.svg)

**Rule-based (ELIZA, AIML, DialogFlow).** 사람이 작성한 pattern이 user input과 match되어 response를 만든다. Intent classifier는 predefined flow로 routing한다. Slot-filling state machine은 필요한 정보를 수집한다. 설계된 좁은 scope 안에서는 매우 잘 동작한다. 그 밖에서는 즉시 실패한다. hallucination을 허용할 수 없는 safety-critical domain(은행 인증, 항공권 예약)에서는 여전히 production에 쓰인다.

**Retrieval-based.** FAQ 스타일 시스템이다. 모든 (utterance, response) pair를 encode한다. runtime에는 사용자의 message를 encode하고 가장 가까운 저장 response를 retrieve한다. Zendesk의 고전적인 "similar articles" 기능을 떠올리면 된다. rule보다 paraphrase를 더 잘 처리한다. generation이 없으므로 hallucination도 없다.

**Neural (seq2seq).** conversation log로 학습한 encoder-decoder다. response를 처음부터 생성한다. 유창하지만 generic output("I don't know")과 factual drift에 취약하다. topic을 안정적으로 유지하지 못했다. 2016-2019년에 Google, Facebook, Microsoft의 chatbot이 모두 실망스러웠던 이유다.

**LLM agents.** plan하고 tool을 호출하며 outcome을 검증하는 loop로 감싼 language model이다. 긴 prompt가 붙은 chatbot이 아니다. Agent loop는 plan → call tool → observe result → decide next step이다. Retrieval-first grounding(RAG)은 hallucination을 줄인다. Tool call은 실제 action을 가능하게 한다. 이것이 2026년 architecture다.

네 paradigm은 순차적 replacement가 아니다. 2026년 production chatbot은 네 가지를 모두 거쳐 routing한다. 인증과 destructive action에는 rule-based, FAQ에는 retrieval, 자연스러운 phrasing에는 neural generation, 모호한 open-ended query에는 LLM agent를 쓴다.

## 직접 만들기

### 1단계: rule-based pattern matching

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20줄짜리 ELIZA다. reflection trick("I feel sad" → "Why do you feel sad")은 Weizenbaum 1966의 canonical psychotherapist demo다. 지금도 배울 점이 많다.

### 2단계: retrieval-based (FAQ)

이 illustrative snippet은 `pip install sentence-transformers`가 필요하다(torch도 함께 설치된다). 이 lesson의 실행 가능한 `code/main.py`는 대신 stdlib Jaccard similarity를 사용하므로, lesson은 external dependency 없이 실행된다.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Threshold-based refusal이 핵심 design choice다. best match가 충분히 가깝지 않으면 `None`을 반환하고 시스템이 escalate하게 한다.

### 3단계: neural generation (baseline)

작은 instruction-tuned encoder-decoder(FLAN-T5)나 fine-tuned conversational model을 사용한다. 2026년 기준 단독으로는 production에 쓸 수 없지만(contradiction, off-topic drift, factual nonsense), 자연스러운 phrasing을 위해 hybrid system 안에는 들어간다. DialoGPT 스타일 decoder-only model은 coherent reply를 만들려면 명시적인 turn separator와 EOS 처리가 필요하다. FLAN-T5 text2text pipeline은 teaching example로 바로 동작한다.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 4단계: LLM agent loop

2026년 production 형태:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

세 가지를 짚자. Tool은 LLM이 호출할 수 있는 callable function이다. loop는 LLM이 tool call 대신 final answer를 반환할 때 종료된다. step budget은 모호한 task에서 infinite loop를 막는다.

실제 production에는 retrieval-first grounding(각 LLM call 전에 relevant docs 주입), guardrail(confirmation 없는 destructive action 거부), observability(모든 step logging), evaluation(agent behavior가 spec을 벗어나지 않는지 자동 check)이 추가된다.

### 5단계: hybrid routing

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

pattern은 이렇다. destructive한 것은 deterministic rule, canned FAQ는 retrieval, 나머지는 LLM agent. 이것이 2026년 customer-support system에 배포되는 방식이다.

## 활용하기

2026년 stack:

| 사용 사례 | 아키텍처 |
|---------|---------------|
| 예약, 결제, 인증 | Rule-based state machine + slot filling |
| 고객 지원 FAQ | curated answer 위의 retrieval |
| Open-ended help chat | RAG + tool call을 갖춘 LLM agent |
| Internal tool / IDE assistant | tool call(search, read, write)을 갖춘 LLM agent |
| Companion / character chatbot | persona system prompt를 둔 tuned LLM, knowledge retrieval |

production에서는 항상 hybrid routing을 사용하라. 어떤 단일 architecture도 모든 request를 잘 처리하지 못한다. routing layer 자체는 보통 작은 intent classifier다.

## 여전히 production에 들어가는 failure mode

- **Confident fabrication.** LLM agent가 실제로 하지 않은 action을 완료했다고 주장한다. Mitigation: outcome을 검증하고, tool call을 log하며, successful tool return 없이 LLM이 무언가를 했다고 주장하게 두지 않는다.
- **Prompt injection.** 사용자가 system prompt를 override하는 text를 삽입한다. OWASP Top 10 for LLM Applications 2025에서 LLM01로 ranked되었다. 두 종류가 있다. direct injection(chat에 붙여 넣음)과 indirect injection(agent가 읽는 document, email, tool output에 숨김)이다.

  Attack rate는 scenario마다 다르다. general tool-use와 coding benchmark에서 frontier model 전반의 measured success rate는 약 0.5-8.5% 범위다. 특정 high-risk setup(AI coding agent에 대한 adaptive attack, 취약한 orchestration)은 약 84%에 도달했다. Production CVE에는 EchoLeak(CVE-2025-32711, CVSS 9.3)이 있다. attacker-controlled email로 trigger되는 Microsoft 365 Copilot의 zero-click data-exfiltration flaw다.

  Mitigation: loop 전체에서 user input을 untrusted로 취급한다. tool call 전에 sanitize한다. tool output을 main prompt에서 격리한다. agent가 먼저 plan한 뒤, 실행 전 각 action을 그 plan과 대조해 verify하는 Plan-Verify-Execute(PVE) pattern을 사용한다(이렇게 하면 tool result가 새로운 unplanned action을 inject하지 못한다). destructive action에는 user confirmation을 요구한다. tool scope에는 least-privilege를 적용한다.

  어떤 prompt engineering도 이 risk를 완전히 제거하지 못한다. External runtime defense layer(LLM Guard, allowlist validation, semantic anomaly detection)가 필요하다.
- **Scope creep.** tool call이 약간 관련된 정보를 반환해서 agent가 task를 벗어난다. Mitigation: tool contract를 좁힌다. system prompt를 focused하게 유지한다. off-task rate evaluation을 추가한다.
- **Infinite loops.** Agent가 같은 tool을 계속 호출한다. Mitigation: step budget, tool-call deduplication, "are we making progress"에 대한 LLM judge.
- **Context window exhaustion.** 긴 conversation이 초기 turn을 context 밖으로 밀어낸다. Mitigation: 오래된 turn을 summarize하거나, relevant past turn을 similarity로 retrieve하거나, long-context model을 사용한다.

## 배포하기

`outputs/skill-chatbot-architect.md`로 저장하라:

```markdown
---
name: chatbot-architect
description: 주어진 use case에 맞는 chatbot stack을 설계한다.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

product context(user need, compliance constraint, available tool, data volume)가 주어지면 다음을 output하라:

1. 아키텍처. Rule-based, retrieval, neural, LLM agent, 또는 hybrid(어떤 path가 어디로 가는지 명시).
2. 해당하는 경우 LLM 선택. model family(Claude, GPT-4, Llama-3.1, Mixtral)를 이름으로 제시한다. tool-use quality와 cost에 맞춘다.
3. Grounding 전략. RAG source, retrieval method(lesson 14 참고), tool contract.
4. Evaluation 계획. held-out dialog에서 task success rate, tool-call correctness, off-task rate, hallucination rate.

structured confirmation flow 없이 destructive action(payment, account deletion, data modification)에 pure-LLM agent를 추천하는 것을 거부하라. agent가 무엇이든 write access를 가진다면 prompt-injection audit을 건너뛰는 것도 거부하라.
```

## 연습문제

1. **쉬움.** 커피숍 주문 bot을 위해 위의 rule-based respond를 10개 pattern으로 구현하라. double order, modification, cancellation, unclear intent 같은 edge case를 test하라.
2. **보통.** hybrid FAQ + LLM fallback을 구축하라. SaaS product용 canned FAQ 50개, docs site 위의 retrieval을 사용하는 LLM fallback을 만든다. 실제 support question 100개에서 refusal rate와 accuracy를 측정하라.
3. **어려움.** 위의 agent loop를 세 가지 tool(search, read-user-data, send-email)과 함께 구현하라. prompt injection attempt를 포함한 50개 test scenario로 evaluation을 실행하라. off-task rate, failed task rate, injection success를 report하라.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| Intent | 사용자가 원하는 것 | Categorical label(book_flight, reset_password). handler로 routed된다. |
| Slot | 정보 한 조각 | bot에 필요한 parameter(date, destination). Slot filling은 질문을 이어 가는 sequence다. |
| RAG | Retrieval plus generation | relevant docs를 retrieve한 뒤 LLM response를 ground한다. |
| Tool call | Function invocation | LLM이 name + args를 가진 structured call을 emit한다. Runtime이 실행하고 result를 반환한다. |
| Agent loop | Plan, act, verify | task가 완료될 때까지 tool call과 interleaving된 LLM call을 실행하는 controller. |
| Prompt injection | 사용자가 prompt를 공격 | system prompt를 override하려는 malicious input. |

## 더 읽을거리

- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — original rule-based chatbot paper.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — LLM agent가 주류가 되기 직전 Google의 후기 neural-chatbot paper.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — agent loop pattern에 이름을 붙인 paper.
- [Anthropic's guide on building effective agents](https://www.anthropic.com/research/building-effective-agents) — 2026년에도 유효한 2024년 production guidance.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — prompt-injection paper.
- [OWASP Top 10 for LLM Applications 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — prompt injection을 최상위 security concern으로 만든 ranking.
- [AWS — Securing Amazon Bedrock Agents against Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — Plan-Verify-Execute와 user-confirmation flow를 포함한 실용적인 orchestration-layer defense.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — indirect prompt injection에서 나온 canonical zero-click data-exfiltration CVE. write-access agent에 runtime defense가 필요한 이유를 보여 주는 reference case.
