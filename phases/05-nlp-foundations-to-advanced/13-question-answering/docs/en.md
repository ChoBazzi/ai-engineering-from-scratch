# 질의응답 시스템

> 세 가지 system이 현대 QA를 만들었습니다. Extractive는 span을 찾았습니다. Retrieval-augmented는 답을 document에 grounding했습니다. Generative는 answer를 생성했습니다. 모든 현대 AI assistant는 이 셋의 조합입니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (기계 번역), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 minutes

## 문제

사용자가 "When did the first iPhone launch?"라고 입력하면 "June 29, 2007."을 기대합니다. "Apple의 역사는 길고 다양합니다."가 아닙니다. 문장 없이 고립된 "2007"도 아닙니다. 직접적이고, grounded되어 있고, 정확한 answer입니다.

지난 10년 동안 세 architecture가 QA를 지배했습니다.

- **Extractive QA.** Question과 answer가 포함되어 있다고 알려진 passage가 주어지면, passage 안에서 answer span의 start index와 end index를 찾습니다. SQuAD가 canonical benchmark입니다.
- **Open-domain QA.** Passage가 주어지지 않습니다. 먼저 관련 passage를 retrieve한 뒤, answer를 extract하거나 generate합니다. 이것이 오늘날 모든 RAG pipeline의 기반입니다.
- **Generative / Closed-book QA.** Large language model이 parametric memory에서 answer합니다. Retrieval이 없습니다. Inference는 가장 빠르지만 fact에는 가장 덜 reliable합니다.

2026년의 흐름은 hybrid입니다. 가장 좋은 passage 몇 개를 retrieve한 뒤, generative model에 그 passage에 grounded된 answer를 요청합니다. 이것이 RAG이고, lesson 14는 retrieval 절반을 깊게 다룹니다. 이 lesson은 QA 절반을 만듭니다.

## 개념

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** Question과 passage를 transformer(BERT family)로 함께 encode합니다. Answer의 start token index와 end token index를 예측하는 두 head를 train합니다. Loss는 valid position 위의 cross-entropy입니다. Output은 passage에서 나온 span입니다. 구조상 hallucination하지 않으며, 구조상 passage가 answer할 수 없는 question은 처리하지 못합니다.

**Retrieval-augmented (RAG).** 두 stage입니다. 먼저 retriever가 corpus에서 top-`k` passage를 찾습니다. 다음으로 reader(extractive 또는 generative)가 그 passage를 사용해 answer를 만듭니다. Retriever-reader 분리는 각각을 독립적으로 train하고 evaluate할 수 있게 합니다. 현대 RAG는 둘 사이에 reranker를 자주 추가합니다.

**Generative.** Decoder-only LLM(GPT, Claude, Llama)이 학습된 weight에서 answer합니다. Retrieval step이 없습니다. Common knowledge에는 뛰어나지만 rare 또는 recent fact에는 치명적일 수 있습니다. Hallucination rate는 pretraining data에서 fact frequency와 반비례합니다.

## 직접 만들기

### 1단계: pretrained model로 extractive QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2`는 answer할 수 없는 question을 포함하는 SQuAD 2.0으로 train되었습니다. 기본적으로 `question-answering` pipeline은 model의 null score가 이기는 경우에도 가장 높은 score의 span을 반환합니다. 즉, empty answer를 자동으로 반환하지는 않습니다. 명시적인 "no answer" 동작을 얻으려면 pipeline call에 `handle_impossible_answer=True`를 넘깁니다. 그러면 null score가 모든 span score를 초과할 때만 empty answer를 반환합니다. 어느 쪽이든 항상 `score` field를 확인하세요.

### 2단계: retrieval-augmented pipeline (sketch)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Two-stage pipeline입니다. Dense retriever(Sentence-BERT)가 semantic similarity로 relevant passage를 찾습니다. Extractive reader(RoBERTa-SQuAD)가 combined top passage에서 answer span을 뽑습니다. Small corpus에서는 잘 동작합니다. Million-document corpus라면 FAISS나 vector database를 사용하세요.

### 3단계: RAG로 generative answer 만들기

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt pattern이 중요합니다. Model에 context에 grounding하고 context가 부족하면 "I don't know"를 반환하라고 명시하면, naive prompting과 비교해 hallucination rate가 40-60% 낮아집니다. 더 정교한 pattern은 citation, confidence score, structured extraction을 추가합니다.

### 4단계: 현실을 반영하는 evaluation

SQuAD는 **Exact Match (EM)**와 **token-level F1**을 사용합니다. EM은 normalization(lowercase, strip punctuation, remove articles) 뒤 strict match입니다. Prediction이 정확히 일치하거나 0점을 받습니다. F1은 prediction과 reference 사이의 token overlap으로 계산하며 partial credit을 줍니다. 둘 다 paraphrase를 낮게 평가합니다. "June 29, 2007"과 "June 29th, 2007"은 보통 EM 0점(ordinal이 normalization을 깨뜨림)을 받지만, 겹치는 token 덕분에 F1은 상당히 받습니다.

Production QA에서는 다음을 봅니다.

- **Answer accuracy** (metric이 semantic equivalence를 잡지 못하므로 LLM-judged 또는 human-judged).
- **Citation accuracy.** Cited passage가 실제로 answer를 support하나요? Generated citation과 retrieved passage 사이의 string match로 자동 확인하기 쉽습니다.
- **Refusal calibration.** Answer가 retrieved passage에 없을 때 system이 "I don't know"라고 제대로 말하나요? False confidence rate를 측정합니다.
- **Retrieval recall.** Reader를 evaluate하기 전에 retriever가 right passage를 top-`k` 안에 넣는지 측정합니다. Reader는 missing passage를 고칠 수 없습니다.

### RAGAS: 2026 production eval framework

`RAGAS`는 RAG system을 위해 만들어졌고 2026년의 shipping default입니다. Gold reference 없이 네 dimension을 scoring합니다.

- **Faithfulness.** Answer의 각 claim이 retrieved context에서 나오나요? NLI-based entailment로 측정합니다. Primary hallucination metric입니다.
- **Answer relevance.** Answer가 question을 다루나요? Answer에서 hypothetical question을 생성하고 real question과 비교해 측정합니다.
- **Context precision.** Retrieved chunk 중 실제로 relevant한 비율은 얼마인가요? Low precision = prompt 안의 noise입니다.
- **Context recall.** Retrieved set이 필요한 정보를 모두 담고 있었나요? Low recall = reader가 성공할 수 없습니다.

Reference-free scoring은 curated gold answer 없이 live production traffic을 evaluate하게 해 줍니다. Exact-match metric이 쓸모없는 open-ended question에는 그 위에 LLM-as-judge를 얹으세요.

`pip install ragas`. Retriever + reader를 plug in하세요. Query마다 네 scalar를 얻습니다. Regression에 alert를 걸면 됩니다.

## 활용하기

2026년 stack입니다.

| Use case | Recommended |
|---------|-------------|
| 주어진 passage에서 answer span 찾기 | `deepset/roberta-base-squad2` |
| Fixed corpus에서 closed-book이 acceptable하지 않음 | RAG: dense retriever + LLM reader |
| Document store 위의 real-time 검색 | Hybrid(BM25 + dense) retriever + reranker를 쓰는 RAG (lesson 14) |
| Conversational QA (follow-up question) | Conversation history + 매 turn RAG를 쓰는 LLM |
| 매우 factual한 regulated domain | Authoritative corpus 위의 extractive; generative alone은 사용하지 않음 |

Extractive QA는 2026년에 유행은 아니지만, LLM을 쓰는 RAG가 더 많은 case를 처리하기 때문입니다. 그래도 literal quotation이 필요한 context에서는 여전히 shipping됩니다. Legal research, regulatory compliance, audit tool이 여기에 해당합니다.

## 출시하기

`outputs/skill-qa-architect.md`로 저장하세요.

```markdown
---
name: qa-architect
description: QA architecture, retrieval strategy, evaluation plan을 선택합니다.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Requirement(corpus size, question type, factuality constraint, latency budget)가 주어지면 다음을 출력합니다.

1. Architecture. Extractive, extractive reader를 쓰는 RAG, generative reader를 쓰는 RAG, 또는 closed-book LLM. 한 문장 reason.
2. Retriever. None, BM25, dense(encoder 이름 명시), 또는 hybrid.
3. Reader. SQuAD-tuned model, 이름을 명시한 LLM, 또는 "domain-fine-tuned DistilBERT."
4. Evaluation. Extractive benchmark에는 EM + F1; production에는 answer accuracy + citation accuracy + refusal calibration. 무엇을 어떻게 측정하는지 이름 붙입니다.

Regulatory 또는 compliance-sensitive question에는 closed-book LLM answer를 거부합니다. Retrieval-recall baseline이 없는 QA system도 거부합니다(retriever가 right passage를 surfaced했는지 모르면 reader를 evaluate할 수 없습니다). Multi-hop reasoning이 필요한 question은 HotpotQA-trained system 같은 specialized multi-hop retriever가 필요하다고 flag합니다.
```

## 연습문제

1. **Easy.** 위 SQuAD extractive pipeline을 Wikipedia passage 10개에 setup하세요. Question 10개를 hand-craft하세요. Answer가 correct인 빈도를 측정합니다. Passage와 question이 clean하다면 7-9개 correct를 볼 수 있어야 합니다.
2. **Medium.** Refusal classifier를 추가하세요. Top retrieval score가 threshold(예: cosine 0.3)보다 낮으면 reader를 호출하지 말고 "I don't know"를 반환합니다. Held-out set에서 threshold를 tune하세요.
3. **Hard.** 원하는 10,000-document corpus 위에 RAG pipeline을 만드세요. RRF fusion을 쓰는 hybrid retrieval(BM25 + dense)을 구현합니다(lesson 14 참고). Hybrid step 유무에 따른 answer accuracy를 측정합니다. 어떤 question type이 가장 benefit을 얻는지 document하세요.

## 핵심 용어

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Extractive QA | Answer span 찾기 | 주어진 passage 안에서 answer의 start index와 end index를 예측합니다. |
| Open-domain QA | Corpus 위의 QA | Passage가 주어지지 않으며, 먼저 retrieve한 뒤 answer해야 합니다. |
| RAG | Retrieve then generate | Retrieval-augmented generation. Retriever + reader pipeline입니다. |
| SQuAD | Canonical benchmark | Stanford Question Answering Dataset. EM + F1 metric입니다. |
| Hallucination | 지어낸 answer | Retrieved context가 support하지 않는 reader output입니다. |
| Refusal calibration | 말하지 말아야 할 때 알기 | Answer할 수 없을 때 system이 "I don't know"라고 올바르게 말합니다. |

## 더 읽을거리

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) - benchmark paper입니다.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) - QA를 위한 canonical dense retriever인 DPR입니다.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) - RAG라는 이름을 붙인 paper입니다.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) - 포괄적인 RAG survey입니다.
