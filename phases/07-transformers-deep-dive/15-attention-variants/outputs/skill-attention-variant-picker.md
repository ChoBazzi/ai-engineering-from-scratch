---
name: attention-variant-picker
description: context 길이, retrieval 요구, 계산 프로파일이 주어졌을 때 새 모델의 full / sliding-window / sparse / differential attention topology를 고른다.
version: 1.0.0
phase: 7
lesson: 15
tags: [attention, transformer, long-context, inference, memory]
---

# Attention 변형 선택기

개발자가 새 트랜스포머 또는 더 긴 context로 확장 중인 기존 트랜스포머에 대해 attention topology를 고르고 정당화하도록 돕는다.

## 수집할 입력

1. 훈련과 추론에서의 **목표 context 길이**(둘이 다른 경우가 많다. 많은 모델은 16K에서 학습하고 추론에서 확장한다).
2. 1-5 척도의 **retrieval 요구**: 1 = 순수 chat, 5 = needle-in-haystack / RAG / 긴 repository context가 있는 code.
3. **추론 메모리 예산**: 요청당 KV cache 허용치(토큰당 layer당 byte가 올바른 단위다).
4. **훈련 비용 허용치** — SWA를 처음부터 학습하는 것은 싸다. 사전학습 모델에 differential attention을 retrofitting하는 것은 비싸다.
5. **하드웨어 대상** — Hopper+는 full FlashAttention-3, Ada는 FA2, 더 오래된 GPU는 mask 제약이 크다.

## 결정 규칙

- **context ≤ 16K, retrieval ≤ 3**: FlashAttention이 있는 full attention. 너무 이르게 최적화하지 말라.
- **context 16-128K, retrieval ≤ 3**: 5:1의 mixed SWA + global, window 1024(Gemma 3 모양). KV를 크게 줄이면서 retrieval을 쓸 만하게 유지한다.
- **context > 128K**: 4-6 layer마다 global layer가 있는 full SWA, position interpolation / YaRN scaling(Lesson 04) 포함.
- **retrieval = 5, training budget 허용**: 상위 4개 layer에만 differential attention을 고려하라(KV doubling은 절반, sink-cancellation 이득은 대부분).
- **public API를 배포하는 경우**: 안정적인 pattern(full, SWA, Gemma-3 mix)을 선호하라. Kernel engineer가 없다면 native-sparse / DIFF는 건너뛰라.
- **base model을 바꿀 수 없는 경우**: SWA는 masking을 통해 추론 시 retrofitting할 수 있다. Differential과 sparse는 어렵다.

## 항상 표시할 것

- Pure-SWA 모델은 7B 아래에서 reasoning benchmark가 눈에 띄게 떨어지는 경우가 많다. 반대하라고 권장한다.
- Window size < 512는 거의 언제나 맞지 않다. 더 키우거나 다른 topology를 써라.
- Differential attention 논문의 보고는 작은 모델(3-7B)에 대한 것이다. 2026년 초 기준 scale-up 증거는 얇다.
- 모든 변형은 RoPE / YaRN scaling(Lesson 04)과 상호작용한다. Position scheme을 명시하라.

## 출력 형식

다음을 반환하라.

1. **추천안** — 하나의 이름 있는 topology(예: "Gemma-3 mix, W=1024, 5:1 SWA:global").
2. **근거** — 각 입력을 위 결정 규칙에 매핑한다.
3. **KV cache 추정** — 목표 context에서 토큰당 layer당 byte와 batch 1의 GB.
4. **마이그레이션 경로** — 기본 모델이 이미 학습된 경우 retrofitting 방법.
5. **알려진 위험** — 어떤 benchmark / workload가 나빠질 수 있는지.
