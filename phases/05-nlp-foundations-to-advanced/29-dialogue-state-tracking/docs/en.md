# Dialogue State Tracking

> "북쪽에 있는 저렴한 식당을 원해요... 아니, 보통 가격대로 바꾸고... 이탈리아 음식도 추가해 주세요." 세 턴, 세 번의 state update입니다. DST는 예약이 제대로 동작하도록 slot-value dict를 동기화합니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75분

## 문제

task-oriented dialogue system에서 사용자의 목표는 `{cuisine: italian, area: north, price: moderate}` 같은 slot-value 쌍의 집합으로 인코딩됩니다. 사용자의 모든 턴은 slot을 추가하거나, 변경하거나, 제거할 수 있습니다. 시스템은 전체 대화를 읽고 현재 state를 정확하게 출력해야 합니다.

slot 하나만 틀려도 시스템은 잘못된 식당을 예약하고, 잘못된 항공편을 잡거나, 잘못된 카드에 청구합니다. DST는 사용자가 말한 내용과 backend가 실행하는 작업 사이의 핵심 연결부입니다.

LLM이 있는 2026년에도 이것이 여전히 중요한 이유:

- 규정 준수가 중요한 도메인(은행, 의료, 항공권 예약)은 free-form generation이 아니라 deterministic slot value를 요구합니다.
- tool-use agent도 API를 호출하기 전에 slot resolution이 필요합니다.
- Multi-turn correction은 보기보다 어렵습니다. "아니, 목요일로 해 주세요."

현대적 pipeline은 classical DST concepts + LLM extractors + structured-output guardrails입니다.

## 개념

![DST: dialog history → slot-value state](../assets/dst.svg)

**Task structure.** schema는 domain(restaurant, hotel, taxi)과 그 slot(cuisine, area, price, people)을 정의합니다. 각 slot은 비어 있거나, closed set의 값(price: {cheap, moderate, expensive})으로 채워지거나, free-form value(name: "The Copper Kettle")가 될 수 있습니다.

**두 가지 DST formulation.**

- **Classification.** 각 (slot, candidate_value) 쌍마다 yes/no를 예측합니다. closed-vocab slot에 적합합니다. 2020년 이전의 표준 방식입니다.
- **Generation.** dialogue가 주어지면 slot value를 free text로 생성합니다. open-vocab slot에 적합합니다. 현대의 기본 방식입니다.

**Metric.** Joint Goal Accuracy (JGA)는 *모든* slot이 맞은 턴의 비율입니다. All-or-nothing입니다. MultiWOZ 2.4 leaderboard의 최상위 성능은 2026년 기준 약 83%입니다.

**Architectures.**

1. **Rule-based (slot regex + keyword).** 좁은 도메인에서 강력한 baseline입니다. Debugging이 쉽습니다.
2. **TripPy / BERT-DST.** BERT encoding을 사용한 copy-based generation입니다. LLM 이전의 표준입니다.
3. **LDST (LLaMA + LoRA).** domain-slot prompting을 사용하는 instruction-tuned LLM입니다. MultiWOZ 2.4에서 ChatGPT 수준의 품질에 도달합니다.
4. **Ontology-free (2024–26).** schema를 건너뛰고 slot name과 value를 직접 생성합니다. open domain을 처리합니다.
5. **Prompt + structured output (2024–26).** Pydantic schema + constrained decoding을 사용하는 LLM입니다. 5줄의 코드로 production-ready입니다.

### 고전적인 failure mode

- **Co-reference across turns.** "첫 번째 옵션으로 할게요." 어느 옵션인지 해소해야 합니다.
- **Over-write vs append.** 사용자가 "이탈리아 음식도 추가해 주세요"라고 말합니다. cuisine을 바꿔야 할까요, 추가해야 할까요?
- **Implicit confirmations.** "좋아요"는 제안된 예약을 수락한 것일까요?
- **Correction.** "아니, 오후 7시로 해 주세요." 다른 slot을 지우지 않고 time을 update해야 합니다.
- **Coreference to previous system utterance.** "네, 그걸로요." 여기서 "그것"은 무엇일까요?

## 직접 만들기

### 1단계: rule-based slot extractor

`code/main.py`를 보세요. Regex + synonym dictionary는 좁은 도메인의 canonical utterance 중 70%를 처리합니다.

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

canonical vocabulary 밖에서는 취약합니다. deterministic slot confirmation에는 잘 동작합니다.

### 2단계: state update loop

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

세 가지 invariant:

- 사용자가 건드리지 않은 slot은 절대 reset하지 않습니다.
- 명시적 negation("cuisine은 신경 쓰지 마세요")은 반드시 clear해야 합니다.
- 사용자 correction("actually...")은 append가 아니라 overwrite해야 합니다.

### 3단계: structured output을 사용하는 LLM-driven DST

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic은 유효한 state object를 보장합니다. Regex도, schema mismatch도, hallucinated slot도 없습니다.

### 4단계: JGA evaluation

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

calibrate할 질문은 이것입니다. 시스템이 전체 slot을 모두 맞히는 턴의 비율은 얼마인가요? MultiWOZ 2.4에서 2026년 최상위 시스템은 80-83%입니다. 좁은 vocabulary의 in-domain system은 이보다 높아야 합니다. 그렇지 않으면 LLM baseline이 이깁니다.

### 5단계: correction 처리

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

correction이 감지되면 append하지 말고 마지막으로 update된 slot을 overwrite합니다. LLM의 도움 없이 제대로 처리하기 어렵습니다. 현대적 패턴은 incremental update 대신 항상 LLM이 history에서 전체 state를 다시 생성하게 하는 것입니다. 그러면 correction이 자연스럽게 처리됩니다.

## Pitfalls

- **Full-history regeneration cost.** LLM이 매 턴 state를 다시 생성하게 하면 전체 token 비용이 O(n²)입니다. history를 제한하거나 오래된 턴을 summarize하세요.
- **Schema drift.** 나중에 새 slot을 추가하면 오래된 training data가 깨집니다. schema를 version 관리하세요.
- **Case sensitivity.** "Italian" vs "italian" vs "ITALIAN" — 모든 곳에서 normalize하세요.
- **Implicit inheritance.** 사용자가 이전에 "4명"이라고 지정했다면, 다른 시간으로 새 요청을 해도 people을 clear하면 안 됩니다. 항상 full history를 전달하세요.
- **Free-form vs closed-set.** 이름, 시간, 주소에는 free-form slot이 필요하고 cuisine과 area는 closed set입니다. schema 안에서 둘을 섞으세요.

## 사용하기

2026년 stack:

| Situation | Approach |
|-----------|----------|
| 좁은 도메인(하나 또는 두 개의 intent) | Rule-based + regex |
| 넓은 도메인, labeled data 있음 | LDST (MultiWOZ-style data에서 LLaMA + LoRA) |
| 넓은 도메인, label 없음, prod-ready 필요 | LLM + Instructor + Pydantic schema |
| Spoken / voice | ASR + normalizer + LLM-DST |
| Multi-domain booking flow | domain별 Pydantic model을 사용하는 schema-guided LLM |
| 규정 준수가 중요한 경우 | Rule-based primary, confirmation flow를 포함한 LLM fallback |

## 출시하기

`outputs/skill-dst-designer.md`로 저장하세요.

```markdown
---
name: dst-designer
description: dialogue state tracker를 설계합니다 — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

use case(domain, languages, vocab openness, compliance needs)가 주어지면 다음을 출력하세요.

1. Schema. domain list, domain별 slot, slot별 open vs closed vocabulary.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. 이유.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. held-out dialogue set에서 Joint Goal Accuracy, slot-level precision/recall, 가장 어려운 slot의 confusion.
5. Confirmation flow. 언제 사용자에게 명시적으로 confirm을 요청할지(destructive actions, low-confidence extractions).

rule-based secondary check가 없는 compliance-sensitive slot의 LLM-only DST는 거부하세요. user correction에서 slot을 roll back할 수 없는 DST는 거부하세요. version tag가 없는 schema를 flag하세요.
```

## 연습문제

1. **Easy.** `code/main.py`에서 3개 slot(cuisine, area, price)에 대한 rule-based state tracker를 만드세요. 손으로 만든 dialogue 10개에서 test하세요. JGA를 측정하세요.
2. **Medium.** 같은 dataset을 Instructor + Pydantic + small LLM으로 실행하세요. JGA를 비교하세요. 가장 어려운 턴을 inspect하세요.
3. **Hard.** 둘 다 구현하고 route하세요. rule-based를 primary로 사용하고, rule-based가 confidence와 함께 <2개 slot을 emit하면 LLM fallback을 사용하세요. combined JGA와 턴당 inference cost를 측정하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | dialogue turn 전체에서 slot-value dict를 유지합니다. |
| Slot | 사용자 intent의 단위 | backend가 필요로 하는 named parameter(cuisine, date). |
| Domain | task area | Restaurant, hotel, taxi — slot의 집합입니다. |
| JGA | Joint Goal Accuracy | 모든 slot이 맞은 턴의 비율입니다. All-or-nothing입니다. |
| MultiWOZ | benchmark | Multi-domain WOZ dataset이며, 표준 DST evaluation입니다. |
| Ontology-free DST | schema 없음 | 고정 list 없이 slot name과 value를 직접 생성합니다. |
| Correction | "Actually..." | 이전에 채워진 slot을 overwrite하는 turn입니다. |

## 더 읽을거리

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — canonical benchmark입니다.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — DST를 위한 LLaMA + LoRA instruction tuning입니다.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — copy-based DST workhorse입니다.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — EM 기반 unsupervised TOD입니다.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — canonical DST results입니다.
