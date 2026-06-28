# LLM 평가 — RAGAS, DeepEval, G-Eval

> exact-match와 F1은 의미적 동등성을 놓친다. 사람 리뷰는 규모를 키울 수 없다. LLM-as-judge가 프로덕션의 답이다. 단, 그 숫자를 믿을 만큼 충분히 calibration해야 한다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (질의응답), Phase 5 · 14 (정보 검색)
**Time:** ~75 minutes

## 문제

당신의 RAG 시스템이 "June 29th, 2007."이라고 답한다.
gold reference는 "June 29, 2007."이다.
Exact Match는 0점을 준다. F1은 ~75%를 준다. 사람이라면 100%로 채점할 것이다.

이제 여기에 테스트 케이스 10,000개를 곱해 보자. retriever, chunking, prompt, model이 바뀔 때마다 다시 곱해 보자. 의미를 이해하고, 대규모에서도 저렴하게 실행되며, regression을 속이지 않고, 올바른 failure mode를 드러내는 evaluator가 필요하다.

2026년에는 이 문제를 다루는 세 가지 framework가 있다.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. NLI + LLM-judge backend을 사용하는 네 가지 RAG metric(faithfulness, answer-relevance, context-precision, context-recall). 연구 기반이며 가볍다.
- **DeepEval.** LLM을 위한 pytest. G-Eval, task-completion, hallucination, bias metric. CI/CD에 자연스럽게 맞는다.
- **G-Eval.** 하나의 방법이자 DeepEval metric이다. chain-of-thought, custom criteria, 0-1 score를 사용하는 LLM-as-judge.

세 가지 모두 LLM-as-judge에 기대고 있다. 이 lesson은 그 방법과 그 주변의 신뢰 계층에 대한 직관을 만든다.

## 개념

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** 정적 metric을 rubric에 따라 output을 채점하는 LLM으로 바꾼다. `(query, context, answer)`가 주어지면 judge LLM에 "faithfulness를 0-1로 채점하라"고 prompt한다. score를 반환한다.

작동하는 이유: LLM은 훨씬 적은 비용으로 사람의 판단에 가까운 값을 낸다. 채점 케이스당 ~$0.003인 GPT-4o-mini를 쓰면 1000개 sample regression eval을 $5 미만으로 실행할 수 있다.

조용히 실패하는 이유:

1. **Judge bias.** judge는 더 긴 답, 자기 model family에서 나온 답, prompt style과 맞는 답을 선호한다.
2. **JSON parsing failures.** 잘못된 JSON → NaN score → aggregate에서 조용히 제외된다. RAGAS 사용자는 이 고통을 안다. try/except + 명시적 failure mode로 gate하라.
3. **Drift over model versions.** judge를 upgrade하면 모든 metric이 바뀐다. judge model + version을 고정하라.

**RAG 네 가지.**

| Metric | 질문 | Backend |
|--------|----------|---------|
| Faithfulness | 답의 각 claim이 retrieved context에서 나왔는가? | NLI-based entailment |
| Answer relevance | 답이 질문에 응답하는가? | 답에서 가상 질문을 생성하고 실제 질문과 비교 |
| Context precision | retrieved chunk 중 관련 있는 비율은 얼마인가? | LLM-judge |
| Context recall | retrieval이 필요한 모든 것을 반환했는가? | gold answer에 대한 LLM-judge |

**G-Eval.** "답이 올바른 source를 인용했는가?" 같은 custom criterion을 정의한다. framework가 이를 chain-of-thought evaluation step으로 자동 확장한 뒤 0-1로 채점한다. RAGAS가 다루지 않는 domain-specific quality dimension에 좋다.

**Calibration.** human label과의 correlation을 확인하기 전까지 raw judge score를 믿지 마라. 사람이 label한 example 100개를 실행하라. judge vs human을 plot하라. Spearman rho를 계산하라. rho < 0.7이면 judge rubric을 손봐야 한다.

## 직접 만들기

### 1단계: NLI로 faithfulness 계산하기(RAGAS-style)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

답을 atomic claim으로 분해한다. 각 claim을 retrieved context에 대해 NLI-check한다. Faithfulness = support된 비율이다.

### 2단계: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

답이 실제 질문과 다른 질문들을 암시한다면 relevance가 떨어진다.

### 3단계: G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

evaluation step 자체가 rubric이다. 명시적 step은 암묵적인 "score 0-1" prompt보다 더 안정적이다.

### 4단계: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

pytest file로 ship하라. 모든 PR에서 실행하라. regression이 있으면 merge를 막아라.

### 5단계: toy eval from scratch

`code/main.py`를 보라. stdlib만으로 faithfulness(답 claim과 context의 overlap)와 relevance(답 token과 질문 token의 overlap)를 근사한다. 프로덕션용은 아니다. 구조를 보여준다.

## 함정

- **Calibration 없음.** human label과 correlation이 0.3인 judge는 noise다. ship 전에 calibration run을 요구하라.
- **Self-evaluation.** 같은 LLM으로 생성하고 judge하면 score가 10-20% 부풀려진다. judge에는 다른 model family를 사용하라.
- **Pairwise judging의 positional bias.** judge는 먼저 제시된 option을 선호한다. 항상 순서를 randomize하고 양쪽 순서로 실행하라.
- **Raw aggregate는 failure를 숨긴다.** mean score 0.85는 5%의 catastrophic failure를 자주 숨긴다. 항상 bottom quantile을 inspect하라.
- **Golden dataset rot.** version이 없는 eval set이 시간이 지나며 drift하면 longitudinal comparison이 깨진다. 변경할 때마다 dataset에 tag를 붙여라.
- **LLM cost.** 규모가 커지면 judge call이 비용을 지배한다. calibration threshold를 만족하는 가장 저렴한 model을 사용하라. GPT-4o-mini, Claude Haiku, Mistral-small.

## 사용하기

2026년 stack:

| 사용 사례 | Framework |
|---------|-----------|
| RAG quality monitoring | RAGAS(4 metrics) |
| CI/CD regression gate | DeepEval + pytest |
| custom domain criteria | DeepEval 안의 G-Eval |
| online live-traffic monitoring | reference-free mode의 RAGAS |
| human-in-the-loop spot check | annotation UI가 있는 LangSmith 또는 Phoenix |
| red-teaming / safety eval | Promptfoo + DeepEval |

일반적인 stack: monitoring에는 RAGAS, CI에는 DeepEval, 새로운 dimension에는 G-Eval. 세 가지를 모두 실행하라. 서로 다른 의견을 내는 것이 유용하다.

## 배포하기

`outputs/skill-eval-architect.md`로 저장하라:

```markdown
---
name: eval-architect
description: calibrated judge와 CI gate를 갖춘 LLM evaluation plan을 설계한다.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

use case(RAG / agent / generative task)가 주어지면 다음을 output하라:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + criteria가 있는 custom G-Eval metrics.
2. Judge model. 명명된 model + version, cost vs accuracy의 근거.
3. Calibration. hand-labeled set size, human 대비 target Spearman rho > 0.7.
4. Dataset versioning. tag strategy, change log, stratification.
5. CI gate. metric별 threshold, regression-window logic, bottom-quantile alert.

≥50개 human-labeled example에 대해 test되지 않은 judge에 의존하는 것을 거부하라. self-evaluation(같은 model이 generate + judge)을 거부하라. bottom-10%를 드러내지 않는 aggregate-only reporting을 거부하라. parallel baseline eval 없이 judge upgrade가 반영되는 pipeline을 flag하라.
```

## 연습 문제

1. **쉬움.** 알려진 hallucination이 있는 RAG example 10개에 RAGAS를 사용하라. faithfulness metric이 각각을 잡아내는지 verify하라.
2. **보통.** QA answer 50개를 correctness 기준으로 0-1 hand-label하라. G-Eval로 score하라. judge와 human 사이의 Spearman rho를 measure하라.
3. **어려움.** DeepEval로 pytest CI gate를 만들라. 의도적으로 retriever를 regress시키라. gate가 실패하는지 verify하라. 가장 낮은 10%에 대한 threshold check로 bottom-quantile alerting을 추가하라.

## 핵심 용어

| Term | 사람들이 말하는 뜻 | 실제 의미 |
|------|-----------------|-----------------------|
| LLM-as-judge | LLM으로 채점하기 | rubric이 주어졌을 때 judge model에 output을 0-1로 채점하게 prompt한다. |
| RAGAS | RAG metric library | 4개의 reference-free RAG metric을 가진 open-source eval framework. |
| Faithfulness | 답이 근거를 갖는가? | retrieved context가 entail하는 answer claim의 비율. |
| Context precision | retrieved chunk가 관련 있었는가? | top-K chunk 중 실제로 중요했던 비율. |
| Context recall | retrieval이 모든 것을 찾았는가? | retrieved chunk가 support하는 gold-answer claim의 비율. |
| G-Eval | Custom LLM judge | rubric + chain-of-thought eval step + 0-1 score. |
| Calibration | 믿되 검증하라 | judge score와 human score 사이의 Spearman correlation. |

## 더 읽을거리

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — RAGAS 논문.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — G-Eval 논문.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — open production stack.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — bias, calibration, limit.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — RAGAS, DeepEval, Phoenix를 통합하는 framework.
