# Long-Context 평가 — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro는 10M token context를 광고한다. 1M token에서는 8-needle MRCR이 26.3%까지 떨어진다. 광고된 값 ≠ 사용할 수 있는 값이다. long-context evaluation은 당신이 ship하는 model의 실제 capacity를 알려준다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (질의응답), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 minutes

## 문제

200쪽짜리 계약서가 있다. model은 1M-token context를 주장한다. 당신은 계약서를 붙여 넣고 "해지 조항은 무엇인가?"라고 묻는다. model은 답한다. 하지만 해지 조항은 120k token 깊이에 있고 model이 실제로 attention을 주는 범위를 지나 있기 때문에, model은 표지에서 답한다.

이것이 2026년의 context-capacity gap이다. spec sheet는 1M 또는 10M이라고 말한다. 현실에서는 그중 60-70%만 usable하며, "usable"은 task에 따라 달라진다.

- **Retrieval(single needle in haystack):** frontier model에서는 광고된 max까지 거의 완벽하다.
- **Multi-hop / aggregation:** 대부분의 model에서 ~128k를 넘으면 급격히 degrade된다.
- **Reasoning over dispersed facts:** 가장 먼저 실패하는 task다.

Long-context evaluation은 이 축들을 측정한다. 이 lesson은 benchmark의 이름, 각 benchmark가 실제로 측정하는 것, 그리고 당신의 domain을 위한 custom needle test를 만드는 방법을 설명한다.

## 개념

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** 긴 context의 제어된 depth에 fact("the magic word is pineapple")를 배치한다. model에게 그것을 retrieve하라고 요청한다. depth × length를 sweep한다. 최초의 long-context benchmark다. 현재 frontier model은 이를 거의 saturate하므로, 필요하지만 충분하지 않은 baseline이다.

**RULER (Nvidia, 2024).** 4개 category에 걸친 13개 task type: retrieval(single / multi-key / multi-value), multi-hop tracing(variable tracking), aggregation(common word frequency), QA. context length(4k부터 128k+)를 설정할 수 있다. NIAH는 saturate하지만 multi-hop에서는 실패하는 model을 드러낸다. 2024년 release에서는 32k+ context를 주장한 17개 model 중 절반만 32k에서 quality를 유지했다.

**LongBench v2 (2024).** 503개의 multiple-choice question, 8k-2M word context, 여섯 task category: single-doc QA, multi-doc QA, long in-context learning, long dialogue, code repo, long structured data. 실제 long-context behavior를 위한 production benchmark다.

**MRCR (Multi-Round Coreference Resolution).** 대규모 multi-turn coreference. 8-needle, 24-needle, 100-needle variant가 있다. attention이 degrade되기 전 model이 얼마나 많은 fact를 동시에 다룰 수 있는지 드러낸다.

**NoLiMa.** "Non-lexical needle." needle과 query가 literal overlap을 공유하지 않는다. retrieval에는 한 단계의 semantic reasoning이 필요하다. NIAH보다 어렵다.

**HELMET.** 여러 document를 concatenate하고 그중 하나에서 나온 질문을 한다. selective attention을 test한다.

**BABILong.** irrelevant haystack 안에 bAbI reasoning chain을 embed한다. 단순 retrieval이 아니라 reasoning-in-a-haystack을 test한다.

### 실제로 report할 것

- **Advertised context window.** spec-sheet number.
- **Effective retrieval length.** 특정 threshold(예: 90%)에서 NIAH pass.
- **Effective reasoning length.** 해당 threshold에서 multi-hop 또는 aggregation pass.
- **Degradation curve.** task type별로 plot한 accuracy vs context length.

spec sheet에 필요한 숫자는 두 개다. retrieval-effective와 reasoning-effective. 보통 reasoning-effective는 advertised window의 25-50%다.

## 직접 만들기

### 1단계: domain용 custom NIAH

`code/main.py`를 보라. skeleton은 다음과 같다:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

`depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}를 sweep하라. heatmap을 plot하라. 이것이 target model의 NIAH card다.

### 2단계: multi-needle variant

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

"세 가지 magic word는 무엇인가?" 같은 질문은 세 가지 모두를 retrieve해야 한다. single-needle success가 multi-needle success를 예측하지는 않는다.

### 3단계: multi-hop variable tracing(RULER-style)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

답하려면 세 assignment를 chaining해야 한다. 128k의 frontier model도 여기서는 accuracy가 50-70%로 자주 떨어진다.

### 4단계: 당신의 stack에서 LongBench v2 실행하기

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

category별 accuracy를 report하라. aggregate score는 큰 task-level 차이를 숨긴다.

## 함정

- **NIAH-only evaluation.** 1M token에서 NIAH를 통과했다는 것은 multi-hop에 대해 아무것도 말해 주지 않는다. 항상 RULER나 custom multi-hop test를 실행하라.
- **Uniform depth sampling.** 많은 implementation은 depth=0.5만 test한다. depth=0, 0.25, 0.5, 0.75, 1.0을 test하라. "lost in the middle" effect는 실제다.
- **Filler와의 lexical overlap.** needle이 filler와 keyword를 공유하면 retrieval이 trivial해진다. NoLiMa-style non-overlapping needle을 사용하라.
- **Latency 무시.** 1M-token prompt는 prefill에 30-120초가 걸린다. accuracy와 함께 time-to-first-token을 측정하라.
- **Vendor-self-reported numbers.** OpenAI, Google, Anthropic 모두 자체 score를 publish한다. 항상 당신의 use case에서 독립적으로 다시 실행하라.

## 사용하기

2026년 stack:

| 상황 | Benchmark |
|-----------|-----------|
| 빠른 sanity check | 3 depths × 3 lengths의 custom NIAH |
| production용 model selection | target length에서 RULER(13 tasks) |
| real-world QA quality | LongBench v2 single-doc-QA subset |
| multi-hop reasoning | BABILong 또는 custom variable-tracing |
| conversational / dialogue | target length에서 MRCR 8-needle |
| model upgrade regression | 고정된 in-house NIAH + RULER harness, 새 model마다 실행 |

production rule of thumb: 의도한 length에서 NIAH + reasoning task 1개를 실행하기 전까지 context window를 믿지 마라.

## 배포하기

`outputs/skill-long-context-eval.md`로 저장하라:

```markdown
---
name: long-context-eval
description: 주어진 model과 use case를 위한 long-context evaluation battery를 설계한다.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

target model, target context length, use case가 주어지면 다음을 output하라:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. 각 length에서 depth 0, 0.25, 0.5, 0.75, 1.0.
3. Metrics. retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. effective retrieval length(90% pass)와 effective reasoning length(70% pass). 둘 다 report하라.
5. Regression. fixed harness, model upgrade마다 rerun, delta를 surface하라.

model card만 보고 context window를 믿는 것을 거부하라. multi-hop workload에 대해 NIAH-only evaluation을 거부하라. vendor self-reported long-context score를 독립적 evidence로 받아들이는 것을 거부하라.
```

## 연습 문제

1. **쉬움.** 3 depths(0.25, 0.5, 0.75) × 3 lengths(1k, 4k, 16k)의 NIAH를 만들라. 아무 model에서나 실행하라. pass rate를 3×3 heatmap으로 plot하라.
2. **보통.** 3-needle variant를 추가하라. 각 length에서 3개 모두의 retrieval을 measure하라. 같은 length의 single-needle pass rate와 비교하라.
3. **어려움.** 64k filler 안에 embed된 variable-tracing task(X1 → X2 → X3, 3 hops)를 구성하라. 3개 frontier model에서 accuracy를 measure하라. model별 effective reasoning length를 report하라.

## 핵심 용어

| Term | 사람들이 말하는 뜻 | 실제 의미 |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | filler에 fact를 심고 model에게 retrieve하게 한다. |
| RULER | 강화된 NIAH | retrieval / multi-hop / aggregation / QA에 걸친 13개 task type. |
| Effective context | 실제 capacity | accuracy가 threshold 위로 유지되는 length. |
| Lost in the middle | Depth bias | model이 긴 input의 중간 content에 attention을 덜 준다. |
| Multi-needle | 한 번에 많은 fact | 여러 fact를 심는다. retrieval만이 아니라 attention juggling을 test한다. |
| MRCR | Multi-round coref | 8, 24, 또는 100-needle coreference. attention saturation을 드러낸다. |
| NoLiMa | Non-lexical needle | needle과 query가 literal token을 공유하지 않는다. reasoning이 필요하다. |

## 더 읽을거리

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — original NIAH repo.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — multi-task benchmark.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — 실제 환경의 long-context eval.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — 더 어려운 needle.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — reasoning-in-haystack.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — depth-bias 논문.
