---
name: prompt-tokenizer-builder
description: LLM 프로젝트를 위한 production-quality tokenizer 구축과 debug
version: 1.0.0
phase: 10
lesson: 2
tags: [tokenizer, bpe, byte-level, special-tokens, chat-template, multilingual]
---

# Production Tokenizer 구축기

LLM 프로젝트용 tokenizer를 구축하거나 debug할 때 이 프레임워크를 따르세요.

## Pipeline 체크리스트

모든 production tokenizer에는 다음 다섯 단계가 필요합니다. 하나라도 빠지면 production에서 edge case를 만나게 됩니다.

1. **Normalize** -- NFKC Unicode normalization을 적용합니다. 이는 ligature("fi" -> "fi")를 풀고, fullwidth 문자를 정규화하며, 공백을 표준화합니다. 이 단계를 건너뛰면 같은 단어도 입력 방식에 따라 다른 token ID를 얻습니다.

2. **Pre-Tokenize** -- BPE 전에 텍스트를 chunk로 나눕니다. 영어 중심 모델에는 GPT-2의 regex pattern을 사용합니다. multilingual model에는 SentencePiece의 raw-byte 접근을 사용합니다. 이 선택은 BPE가 단어 경계를 가로질러 병합할 수 있는지를 결정합니다.

3. **BPE Merge** -- 각 chunk 안의 byte sequence에 학습된 merge table을 적용합니다. merge table이 곧 tokenizer의 학습된 지식입니다. 나머지는 연결 배관입니다.

4. **Special Token Injection** -- BPE 실행 전에 special token을 정확히 match합니다. [BOS], [EOS], [PAD], chat template marker는 고정 ID를 얻습니다. 이들은 병합에 참여하지 않습니다.

5. **ID Mapping** -- token string을 정수로 변환합니다. 모델은 정수만 봅니다.

## Tokenizer 문제 debug

**증상: 모델이 chat input에서 엉망인 출력을 생성함**
- chat template을 확인하세요. 모델마다 형식이 다릅니다. Llama 3는 `<|start_header_id|>` marker를 사용합니다. ChatGPT는 `<|im_start|>` marker를 사용합니다. 잘못된 template은 입력을 training distribution 밖으로 밀어냅니다.

**증상: 비영어 텍스트가 너무 많은 token을 사용함**
- fertility(단어당 token 수)를 확인하세요. 2.0을 넘으면 tokenizer가 해당 언어에서 context window를 낭비한다는 뜻입니다. 해결책: 더 많은 multilingual data로 재학습, vocabulary 크기 증가, 또는 Unigram 기반 SentencePiece 사용.

**증상: 숫자와 산술이 실패함**
- digit이 어떻게 tokenization되는지 확인하세요. "1234"가 하나의 token이면 모델은 digit-level operation을 할 수 없습니다. pre-tokenization 중 digit을 개별적으로 분리하세요.

**증상: code token이 비효율적임**
- indentation이 어떻게 처리되는지 확인하세요. GPT-2 tokenizer는 공백에 token을 낭비합니다. Codex와 StarCoder는 special indentation token(공백 4개 = token 1개)을 사용합니다.

## Vocabulary 크기 결정

- 32K tokens: 단일 언어, 작은 모델, 제한된 compute. Embedding layer는 32K * d_model parameter입니다.
- 50K-64K: multilingual 또는 코드 비중이 큼. 대부분의 프로젝트에서 좋은 균형입니다.
- 100K+ (GPT-4, Llama 3): 방대한 training data가 있을 때만. sequence는 짧아지지만 embedding parameter가 100K * d_model입니다.

4096차원 모델 기준: 32K vocab = 131M embedding parameter. 128K vocab = 524M embedding parameter. embedding layer에서만 400M parameter 차이입니다.

## 속도 요구사항

- Training data tokenization: Rust 기반 library(tiktoken, HuggingFace tokenizers)를 사용하세요. 순수 Python은 10-100배 느립니다.
- Inference tokenization: latency의 중요성은 더 낮지만(단일 sequence), 그래도 compiled implementation을 사용하세요.
- Benchmark: 텍스트 1GB를 tokenization하고 wall clock time을 측정하세요. 60초 이상 걸리면 Rust backend로 전환하세요.

## Chat Template 검증

chat model을 배포하기 전에 template을 검증하세요.

1. 알려진 대화를 tokenizer로 encode합니다
2. 다시 텍스트로 decode합니다
3. 모델 문서의 기대 형식과 문자 단위로 비교합니다
4. header token 뒤의 newline, content 앞의 공백, end-of-turn marker에 주의합니다
5. edge case를 테스트합니다: 빈 system message, 매우 긴 user message, 여러 assistant turn

chat template 오류는 chat model 성능 저하의 가장 흔한 원인입니다.
