---
name: spec-decode-picker
description: 새 LLM inference workload에 맞는 speculative decoding 전략(vanilla / Medusa / EAGLE / lookahead)과 tuning parameter를 고른다.
version: 1.0.0
phase: 7
lesson: 16
tags: [inference, decoding, latency, speculative, optimization]
---

# Speculative Decoding 선택기

엔지니어가 vanilla speculative, Medusa, EAGLE, lookahead decoding 중에서 고르고, 특정 workload에 맞게 `N`(draft length)을 조정하도록 돕는다.

## 수집할 입력

1. **Verifier model** — 최종 출력을 만드는 LLM. 크기가 중요하다(draft cost가 verifier cost보다 작아야 speedup이 난다).
2. **workload 유형** — code, chat, structured output, summarization. acceptance rate를 결정한다.
3. **sampling 전략** — greedy, low-T, high-T, beam. High-T sampling은 acceptance를 떨어뜨린다.
4. **하드웨어 대상** — 메모리 예산이 별도 draft model을 올릴 수 있는지 결정한다.
5. **엔지니어링 예산** — Medusa와 EAGLE은 fine-tuning이 필요하다. Vanilla와 lookahead는 필요 없다.
6. **지연 시간 목표** — interactive chat(<500ms TTFT, 토큰당 <50ms) 대 batch(throughput-first).

## 결정 규칙

- **빠른 시작, training 없음**: 같은 계열 1B-3B 모델을 vanilla draft로 쓴다. 보통 2배.
- **fine-tune 가능**: verifier의 hidden state를 사용하는 EAGLE-2 또는 EAGLE-3. 보통 3-4배.
- **fine-tune은 가능하지만 두 모델은 실행할 수 없음**: Medusa(verifier 위의 extra head). 2-3배.
- **training budget도 사용 가능한 draft model도 없음**: lookahead decoding. 1.3-1.6배.
- **batch-heavy serving**: continuous batching이 더 중요하다. batch가 이미 포화되어 있으면 speculative 이득은 줄어든다.
- **high temperature 또는 stochastic sampling**: acceptance가 급격히 떨어진다. 더 낮은 N(2-3)을 고려하거나 비활성화하라.
- **structured output(JSON, code)**: acceptance가 높다. 최대 speedup을 위해 N을 7+까지 밀어라.

## 튜닝

- **N (draft length)**: 5에서 시작한다. acceptance를 측정한다. α > 0.9이면 7로 올린다. α < 0.6이면 3으로 낮춘다.
- **draft temperature**: verifier의 temperature와 맞춘다. draft sampling이 불일치하면 α를 잃는다.
- **tree depth (EAGLE-2 / Medusa)**: 3-5 branch. 더 넓은 tree는 α > 0.8에서만 도움이 된다.
- **draft model size**: α > 0.7을 달성하는 가장 작은 모델. 70B verifier에는 1B draft가 일반적이다. verifier의 tokenizer / embedding compatibility 아래로 내려가지 말라.

## 항상 표시할 것

- Draft와 verifier가 tokenizer를 공유하는지 확인하라. 다른 BPE split은 speculative guarantee를 깨뜨린다.
- Spec decoding은 vLLM의 continuous batching과 상호작용한다. Batch가 이미 포화되어 있으면 요청당 speedup이 줄어든다.
- EAGLE의 hidden-state input은 verifier 내부가 필요하다. HF API로 항상 노출되지는 않는다. vLLM 또는 SGLang runtime을 선호하라.
- Medusa head는 verifier 자체 출력에 대한 supervised fine-tune이 필요하다. Data-gathering step이 지배 비용인 경우가 많다.

## 출력 형식

다음을 반환하라.

1. **추천안** — 하나의 전략 이름과 tuning parameter(예: "EAGLE-2, N=5, tree_depth=4").
2. **예상 speedup** — 명시적인 α 가정 포함.
3. **호환성 점검** — tokenizer 일치, runtime support, KV cache rollback support.
4. **fallback 계획** — primary strategy가 기대보다 못할 때 다음으로 시도할 것.
5. **측정 계획** — 대표 샘플에서 acceptance rate와 speedup을 검증하는 방법.
