---
name: skill-perceptron
description: 퍼셉트론 패턴과 단층 아키텍처와 다층 아키텍처를 각각 언제 써야 하는지 이해합니다
version: 1.0.0
phase: 3
lesson: 1
tags: [perceptron, neural-networks, classification, deep-learning]
---

# 퍼셉트론 패턴

퍼셉트론은 입력의 가중합에 편향을 더한 뒤 계단 함수를 적용해 이진 출력을 만듭니다. 이것은 신경망의 기본 단위입니다.

```text
output = step(w1*x1 + w2*x2 + ... + wn*xn + bias)
```

## 단일 퍼셉트론으로 충분한 경우

- 문제가 선형 분리 가능합니다. 직선 또는 초평면이 두 클래스를 나눌 수 있습니다
- 논리 게이트: AND, OR, NOT, NAND
- 단순 threshold 결정: "score가 X보다 큰가?"
- 데이터가 겹치지 않는 두 영역으로 군집되는 이진 분류기

## 여러 층이 필요한 경우

- 문제가 선형 분리 가능하지 않습니다. 하나의 직선으로 클래스를 분리할 수 없습니다
- XOR와 parity 문제
- "이것이지만 저것은 아닌" 추론이 필요한 모든 작업, 즉 조건 조합
- 실제 classification: images, text, audio는 거의 항상 비선형입니다

## 결정 체크리스트

1. 데이터를 그리거나 살펴보세요. 클래스 사이에 하나의 직선 경계를 그릴 수 있나요?
   - 예: 단일 퍼셉트론이 작동합니다
   - 아니요: 적어도 두 층이 필요합니다
2. 문제를 더 단순한 선형 결정의 AND/OR로 분해할 수 있나요?
   - 이 분해가 최소 network structure를 알려 줍니다
   - XOR = (A OR B) AND (NOT (A AND B)) = 2개 층의 3개 퍼셉트론
3. 클래스가 두 개보다 많다면 클래스마다 출력 노드가 하나씩 필요합니다

## 훈련 규칙

```text
error = expected - predicted
weight_new = weight_old + learning_rate * error * input
bias_new = bias_old + learning_rate * error
```

예측이 맞으면 아무것도 바뀌지 않습니다. 틀리면 가중치가 오차를 줄이는 방향으로 이동합니다. 이것은 단층 퍼셉트론에서만 작동합니다. 다층 네트워크에는 역전파가 필요합니다.

## 흔한 실수

- 단일 퍼셉트론으로 비선형 패턴을 학습하려고 하기. 절대 수렴하지 않습니다
- 학습률을 너무 높게 설정하기. 가중치가 진동합니다. 너무 낮게 설정하면 훈련이 너무 오래 걸립니다
- 편향 항을 잊기. 편향이 없으면 결정 경계는 반드시 원점을 지나야 합니다
- 퍼셉트론 수렴(선형 분리 가능한 데이터에서 보장됨)을 일반 신경망 수렴(보장되지 않음)과 혼동하기
