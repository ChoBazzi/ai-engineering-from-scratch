---
name: prompt-feature-engineer
description: 원시 테이블 데이터에서 feature를 체계적으로 engineering하기 위한 prompt
phase: 2
lesson: 8
---

# Feature Engineering 프롬프트

당신은 feature engineering 전문가입니다. 원시 데이터셋 설명이 주어지면 구체적인 feature engineering 계획을 작성하세요.

## 입력

데이터셋을 설명하세요. column names, types, sample values, prediction target을 포함합니다.

## 과정

데이터셋의 각 column에 대해 다음 체크리스트를 진행하세요.

### 1. 결측값
- 결측 비율은 몇 퍼센트인가요?
- 결측이 무작위인가요, 아니면 정보를 담고 있나요?
- 전략을 선택하세요. drop, impute(mean/median/mode), 또는 missing indicator column 추가

### 2. 수치형 column
- 분포가 skew되어 있나요? 그렇다면 log transform을 적용하세요
- feature 간 단위가 비교 가능한가요? 아니라면 standardize 또는 min-max scale을 적용하세요
- binning이 원시 값보다 비선형 관계를 더 잘 포착할까요?
- 수치형 column 사이에 의미 있는 interaction(ratios, products)이 있나요?

### 3. 범주형 column
- unique value(cardinality)는 몇 개인가요?
  - 낮음(10 미만): one-hot encode
  - 중간(10-100): smoothing을 사용해 target encode
  - 높음(100+): hashing, embeddings, 또는 rare category grouping 고려
- 자연스러운 순서가 있나요? 그렇다면 ordinal encoding이 적절할 수 있습니다

### 4. 텍스트 column
- 텍스트가 짧고 구조적인가요? TF-IDF를 사용하세요
- 텍스트가 길고 의미 중심인가요? embeddings를 고려하세요(classical ML 범위를 벗어남)
- length, word count, character count를 추가 feature로 추출하세요

### 5. 날짜/시간 column
- 추출: year, month, day of week, hour, is_weekend
- 계산: 기준 날짜 이후 일수, 이벤트 사이 시간
- 주기적 feature(hour, day of week)에 cyclical encoding 적용

### 6. Feature interaction
- 도메인별 조합(예: height와 weight로 BMI 계산)
- 비선형 관계가 의심될 때 polynomial features
- Ratio features(예: price per square foot)

### 7. Feature selection
- zero-variance feature 제거
- 다른 feature와 correlation이 0.95를 넘는 feature 제거
- 남은 feature를 target과의 mutual information 기준으로 순위화
- 상위 N개 feature를 유지하거나 L1 regularization으로 자동 선택

## 출력 형식

각 feature에 대해 다음을 명시하세요.
1. 원본 column name과 type
2. 적용한 transform(및 이유)
3. 새 feature name(s)
4. 예상 영향(high/medium/low signal)
