---
name: skill-tokenizer
description: LLM 프로젝트를 위한 tokenizer 선택과 구축
version: 1.0.0
phase: 10
lesson: 1
tags: [tokenizer, bpe, wordpiece, sentencepiece, llm, nlp]
---

# Tokenizer 선택과 구현

LLM 프로젝트를 시작할 때 tokenizer 선택에는 이 의사결정 프레임워크를 적용하세요.

## 각 tokenizer를 언제 사용할까

**Byte-level BPE (tiktoken):** GPT 계열 모델 위에 구축하거나 fine-tuning합니다. 어떤 입력 byte sequence든 처리 보장이 필요합니다. unknown token이 없어야 합니다.

**WordPiece (Hugging Face):** classification, NER, embedding 작업을 위해 BERT 계열 모델을 사용합니다. 단어 경계 신호에 의존하는 downstream task에 "##" continuation prefix가 필요합니다.

**SentencePiece (BPE 또는 Unigram):** 처음부터 학습합니다. language-agnostic tokenization이 필요합니다. 데이터에 CJK 언어, 태국어, 또는 공백 단어 경계가 없는 다른 문자 체계가 포함됩니다. LLaMA, T5, 대부분의 multilingual model이 이를 사용합니다.

## Vocabulary 크기 가이드라인

- 32K tokens: 단일 언어 모델의 좋은 기본값이며 embedding layer를 작게 유지합니다
- 50K-64K tokens: multilingual 또는 코드 비중이 큰 모델에 더 적합합니다
- 100K+ tokens: 방대한 training data가 있고 짧은 sequence를 원할 때만 사용합니다

vocabulary가 클수록 sequence는 짧아지지만(더 저렴한 inference) embedding matrix의 parameter는 늘어납니다. 4096차원 embedding을 쓰는 100K vocabulary에서는 embedding layer만 400M parameter입니다.

## 중요한 pre-tokenization 규칙

1. 단어 간 병합을 막기 위해 BPE 전에 공백 기준으로 나눕니다
2. 모델이 산술을 학습하길 원한다면 숫자를 개별 digit으로 분리합니다
3. 일관된 동작을 위해 tokenization 전에 Unicode(NFC)를 정규화합니다
4. 사용 사례에 맞는 special token을 추가합니다: `<pad>`, `<eos>`, `<bos>`, `<unk>`, 그리고 task-specific marker

## tokenizer 동작의 위험 신호

- 목표 언어에서 fertility가 2.0 초과: 모델이 context window를 낭비합니다
- 흔한 domain 단어가 3개 이상의 token으로 분리됨: domain data로 다시 학습하세요
- 숫자의 tokenization이 일관되지 않음: digit-splitting 규칙을 확인하세요
- single-use token이 많은 큰 vocabulary: vocabulary 크기를 줄이세요

## custom tokenizer 구축 체크리스트

1. 대표성 있는 training data를 수집합니다(목표 domain 텍스트 최소 1GB)
2. algorithm을 선택합니다: 일반 용도는 BPE, multilingual은 Unigram
3. 위 가이드라인에 따라 vocabulary 크기를 정합니다
4. pre-tokenization을 구성합니다: 공백 분리, digit 처리, 구두점
5. special token을 추가합니다
6. Hugging Face tokenizers library로 학습합니다(Rust backend, 빠름)
7. 검증합니다: 모든 목표 언어의 held-out text에서 fertility를 확인합니다
8. edge case를 테스트합니다: 빈 문자열, 매우 긴 입력, binary data, emoji, RTL text
9. tokenizer를 model checkpoint와 함께 저장하고 versioning합니다
