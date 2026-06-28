# 서브워드 토큰화 — BPE, WordPiece, Unigram, SentencePiece

> 단어 tokenizer는 본 적 없는 단어에서 막힙니다. 문자 tokenizer는 시퀀스 길이를 폭증시킵니다. 서브워드 tokenizer는 그 사이의 균형점입니다. 모든 현대 LLM은 이것 위에서 동작합니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## 문제

어휘집에 단어가 50,000개 있습니다. 사용자가 "untokenizable"을 입력합니다. tokenizer는 `[UNK]`를 반환합니다. 이제 모델은 그 단어에 대한 신호를 전혀 갖지 못합니다. 더 나쁜 점은, corpus의 90번째 percentile 문서에 rare word가 40개 있다면 문서마다 40 bits의 정보가 버려진다는 뜻입니다.

서브워드 토큰화는 이 문제를 해결합니다. 흔한 단어는 하나의 token으로 남습니다. rare word는 의미 있는 조각으로 분해됩니다: `untokenizable` → `un`, `token`, `izable`. 어떤 문자열이든 결국 byte의 시퀀스이므로 training data는 모든 입력을 포괄합니다.

2026년의 모든 frontier LLM은 세 알고리즘(BPE, Unigram, WordPiece) 중 하나를 사용하고, 세 라이브러리(tiktoken, SentencePiece, HF Tokenizers) 중 하나로 감쌉니다. 언어 모델을 배포하려면 반드시 하나를 선택해야 합니다.

## 개념

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** 문자 수준 vocabulary에서 시작합니다. 모든 인접 pair를 셉니다. 가장 자주 등장하는 pair를 새 token으로 병합합니다. 목표 vocabulary size에 도달할 때까지 반복합니다. 지배적인 알고리즘: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** 같은 알고리즘이지만 Unicode 문자 대신 raw bytes(기본 token 256개) 위에서 동작합니다. `[UNK]` token이 0개임을 보장합니다. 어떤 byte sequence든 encode할 수 있습니다. GPT-2는 token 50,257개를 사용합니다(256 bytes + 50,000 merges + special 1개).

**Unigram.** 거대한 vocabulary에서 시작합니다. 각 token에 unigram probability를 할당합니다. 제거했을 때 corpus log-likelihood를 가장 적게 악화시키는 token을 반복적으로 prune합니다. 추론 시 확률적일 수 있습니다. tokenization을 sample할 수 있어 subword regularization을 통한 data augmentation에 유용합니다. T5, mBART, ALBERT, XLNet, Gemma가 사용합니다.

**WordPiece.** raw frequency가 아니라 training corpus의 likelihood를 최대화하는 pair를 병합합니다. BERT, DistilBERT, ELECTRA가 사용합니다.

**SentencePiece vs tiktoken.** SentencePiece는 raw Unicode text에서 직접 vocabulary(BPE 또는 Unigram)를 *훈련*하는 라이브러리이며, whitespace를 `▁`로 encode합니다. tiktoken은 미리 만들어진 vocabulary에 대한 OpenAI의 빠른 *encoder*입니다. 훈련은 하지 않습니다.

경험칙:

- **새 vocabulary 훈련:** SentencePiece(다국어, pre-tokenization 없음) 또는 HF Tokenizers.
- **GPT vocab 대상 빠른 추론:** tiktoken(`cl100k_base`, `o200k_base`).
- **둘 다:** HF Tokenizers — 하나의 라이브러리로 training + serving.

```figure
bpe-merge
```

## 직접 만들기

### 1단계: 처음부터 구현하는 BPE

`code/main.py`를 보세요. loop는 다음과 같습니다.

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

이 알고리즘은 세 가지 사실을 encode합니다. `</w>`는 단어 끝을 표시하므로 "low"(suffix)와 "lower"(prefix)가 구분됩니다. Frequency weighting은 high-frequency pair가 초기에 이기게 합니다. merge list에는 순서가 있습니다. 추론은 training order대로 merge를 적용합니다.

### 2단계: 학습한 merge로 encode하기

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

naive 구현은 O(n·|merges|)입니다. Production 구현(tiktoken, HF Tokenizers)은 merge-rank lookup과 priority queue를 사용해 거의 linear time으로 실행됩니다.

### 3단계: 실전의 SentencePiece

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

주목할 점: pre-tokenization이 필요 없고, space는 `▁`로 encode되며, `character_coverage`는 rare character를 얼마나 적극적으로 보존할지와 `<unk>`로 mapping할지를 제어합니다.

### 4단계: OpenAI-compatible vocab을 위한 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Encoding-only입니다. 빠릅니다(Rust backend). byte counting, cost estimation, context-window budgeting에서 GPT-4/5 tokenization과 정확히 일치합니다.

## 2026년에도 배포되는 함정

- **Tokenizer drift.** vocab A로 훈련하고 vocab B로 배포하는 경우입니다. Token ID가 달라지고 모델 출력은 망가집니다. CI에서 `tokenizer.json` hash를 확인하세요.
- **Whitespace ambiguity.** BPE에서 "hello"와 " hello"는 서로 다른 token을 만듭니다. 항상 `add_special_tokens`와 `add_prefix_space`를 명시하세요.
- **Multilingual undertraining.** 영어 중심 corpus는 비라틴 문자 체계를 5-10배 더 많은 token으로 쪼개는 vocabulary를 만듭니다. GPT-3.5에서 같은 prompt가 일본어/아랍어로는 5-10배 더 비싸집니다. `o200k_base`는 이를 일부 고쳤습니다.
- **Emoji splits.** 하나의 emoji가 token 5개를 차지할 수 있습니다. context를 예산화할 때 emoji 처리를 checkpoint하세요.

## 사용하기

2026년 stack:

| 상황 | 선택 |
|-----------|------|
| 단일 언어 모델을 처음부터 훈련 | HF Tokenizers (BPE) |
| 다국어 모델 훈련 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| OpenAI-compatible API serving | tiktoken (`o200k_base` for GPT-4+) |
| Domain-specific vocab (code, math, protein) | domain corpus에서 custom BPE를 훈련하고 base vocab과 병합 |
| Edge inference, small model | Unigram (더 작은 vocabulary가 더 잘 작동) |

Vocabulary size는 상수가 아니라 scaling decision입니다. 대략적인 heuristic: <1B params는 32k, 1-10B는 50-100k, multilingual/frontier는 200k+.

## 배포하기

`outputs/skill-bpe-vs-wordpiece.md`로 저장하세요:

```markdown
---
name: tokenizer-picker
description: 주어진 corpus와 deployment target에 맞는 tokenizer algorithm, vocab size, library를 선택합니다.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

corpus(size, languages, domain)와 deployment target(training from scratch / fine-tuning / API-compatible inference)이 주어지면 다음을 출력합니다.

1. Algorithm. BPE, Unigram, 또는 WordPiece. 한 문장 이유.
2. Library. SentencePiece, HF Tokenizers, 또는 tiktoken. 이유.
3. Vocab size. 가장 가까운 1k로 반올림. model size와 language coverage에 연결된 이유.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. held-out set의 average tokens-per-word, OOV rate, compression ratio, round-trip decode equality.

rare-script content가 있는 corpus에서 character-coverage <0.995 tokenizer를 훈련하는 것을 거부하세요. CI에서 고정된 `tokenizer.json` hash check 없이 vocab을 배포하는 것을 거부하세요. vocab 16k 미만의 monolingual tokenizer는 likely under-spec으로 flag하세요.
```

## 연습 문제

1. **Easy.** `code/main.py`의 tiny corpus에서 500-merge BPE를 훈련하세요. held-out word 3개를 encode하세요. 정확히 1 token이 된 것은 몇 개이고 >1 token이 된 것은 몇 개인가요?
2. **Medium.** 영어 Wikipedia 문장 100개에서 `cl100k_base`, `o200k_base`, 그리고 vocab=32k로 직접 훈련한 SentencePiece BPE의 token count를 비교하세요. 각각의 compression ratio를 보고하세요.
3. **Hard.** 같은 corpus를 BPE, Unigram, WordPiece로 훈련하세요. 작은 sentiment classifier에서 각각을 사용할 때 downstream accuracy를 측정하세요. 선택이 F1을 1 point 넘게 움직이나요?

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 목표 vocab size에 도달할 때까지 가장 빈번한 character pair를 greedy하게 merge합니다. |
| Byte-level BPE | unknown token이 절대 없음 | raw 256 bytes 위의 BPE입니다. GPT-2 / Llama가 이를 사용합니다. |
| Unigram | Probabilistic tokenizer | log-likelihood를 사용해 큰 candidate set에서 prune합니다. T5, Gemma가 사용합니다. |
| SentencePiece | whitespace를 다루는 것 | raw text에서 BPE/Unigram을 훈련하는 라이브러리입니다. space는 `▁`로 encode됩니다. |
| tiktoken | 빠른 것 | 미리 만들어진 vocab을 위한 OpenAI의 Rust-backed BPE encoder입니다. 훈련은 하지 않습니다. |
| Merge list | magic number들 | `(a, b) → ab` merge의 순서 있는 목록입니다. 추론은 순서대로 적용합니다. |
| Character coverage | 얼마나 rare하면 너무 rare한가? | tokenizer가 training corpus에서 cover해야 하는 character 비율입니다. 보통 ~0.9995입니다. |

## 더 읽을거리

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE paper.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram paper.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 라이브러리.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 간결한 reference.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + encoding list.
