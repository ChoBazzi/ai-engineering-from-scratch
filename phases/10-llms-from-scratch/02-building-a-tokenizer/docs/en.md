# 처음부터 Tokenizer 만들기

> Lesson 01은 장난감을 주었습니다. 이 lesson은 실전 도구를 줍니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 minutes

## 학습 목표

- Unicode, whitespace normalization, special token을 처리하는 production-grade BPE tokenizer를 만듭니다
- tokenizer가 emoji, CJK, code를 포함한 모든 입력을 unknown token 없이 encode할 수 있도록 byte-level fallback을 구현합니다
- BPE merge를 적용하기 전에 텍스트를 단어 경계에서 나누는 pre-tokenization regex pattern을 추가합니다
- corpus에서 custom tokenizer를 학습하고 multilingual text에서 tiktoken과 compression ratio를 비교 평가합니다

## 문제

Lesson 01의 BPE tokenizer는 영어 텍스트에서 작동합니다. 이제 일본어를 던져보세요. emoji도요. tab과 space가 섞인 Python code도요.

깨집니다.

BPE가 틀려서가 아닙니다. 구현이 불완전하기 때문입니다. production tokenizer는 어떤 encoding의 raw byte든 처리하고, 분리 전에 Unicode를 정규화하고, 절대 merge되지 않는 special token을 관리하고, pre-tokenization과 subword splitting을 연결하며, 15조 token을 처리하는 training pipeline의 병목이 되지 않을 만큼 빠르게 이 모든 일을 합니다.

GPT-2 tokenizer는 50,257 token을 가집니다. Llama 3는 128,256입니다. GPT-4는 대략 100,000입니다. 장난감 숫자가 아닙니다. 그 vocabulary 뒤의 merge table은 수백 GB 텍스트로 학습되었고, 주변 장치인 normalization, pre-tokenization, special token injection, chat template formatting이 "hello world"만 처리하는 tokenizer와 인터넷 전체를 처리하는 tokenizer를 가릅니다.

이제 그 장치를 만들 것입니다.

## 개념

### 전체 Pipeline

production tokenizer는 하나의 algorithm이 아닙니다. 서로 다른 문제를 해결하는 다섯 단계의 pipeline입니다.

```mermaid
graph LR
    A[Raw Text] --> B[정규화]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

각 단계에는 구체적인 역할이 있습니다.

| Stage | 하는 일 | 중요한 이유 |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, optional lowercase, optional accent stripping | "fi" ligature(U+FB01)가 "fi"(두 문자)가 됩니다. 이 과정이 없으면 같은 단어도 다른 token이 됩니다. |
| Pre-Tokenize | BPE 전에 텍스트를 chunk로 나눔 | BPE가 단어 경계를 가로질러 merge하지 못하게 합니다. "the cat"이 "e c" token을 만들면 안 됩니다. |
| BPE Merge | 학습된 merge rule을 byte sequence에 적용 | 핵심 compression입니다. raw byte를 subword token으로 바꿉니다. |
| Special Tokens | [BOS], [EOS], [PAD], chat template marker 삽입 | 이 token들은 고정 ID를 가집니다. BPE merge에 절대 참여하지 않습니다. 모델은 구조를 위해 이들이 필요합니다. |
| ID Mapping | token string을 integer ID로 변환 | 모델은 string이 아니라 integer를 봅니다. |

### Byte-level BPE

Lesson 01의 tokenizer는 UTF-8 byte에서 작동했습니다. 올바른 선택이었습니다. 하지만 중요한 것을 건너뛰었습니다. 그 byte가 유효한 UTF-8이 아니면 어떻게 될까요?

Byte-level BPE는 가능한 모든 byte 값(0-255)을 유효한 token으로 다뤄 이 문제를 해결합니다. base vocabulary는 정확히 256개 entry입니다. 텍스트, binary, 손상된 파일 등 어떤 파일도 unknown token 없이 tokenization할 수 있습니다.

GPT-2는 한 가지 trick을 추가했습니다. vocabulary가 사람이 읽을 수 있게 각 byte를 printable Unicode 문자에 mapping합니다. byte 0x20(space)은 그들의 mapping에서 문자 "G"가 됩니다. 이는 순전히 표시용입니다. algorithm에는 중요하지 않습니다.

진짜 힘은 byte-level BPE가 지구상의 모든 언어를 처리한다는 점입니다. 중국어 문자는 각각 UTF-8 byte 3개입니다. 일본어는 3-4 byte일 수 있습니다. Arabic, Devanagari, emoji도 모두 byte sequence일 뿐입니다. BPE algorithm은 영어 ASCII byte에서 pattern을 찾는 것과 정확히 같은 방식으로 이 byte sequence에서 pattern을 찾습니다.

### Pre-tokenization

BPE가 텍스트를 만지기 전에 chunk로 나눠야 합니다. 이는 merge algorithm이 단어 경계를 가로지르는 token을 만들지 못하게 합니다.

GPT-2는 regex pattern으로 텍스트를 나눕니다.

```text
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

이 pattern은 contraction("don't"가 "don" + "'t"가 됨), optional leading space가 있는 단어, 숫자, 구두점, 공백을 기준으로 나눕니다. leading space는 단어에 붙은 채 유지됩니다. 그래서 "the cat"은 ["the", " ", "cat"]이 아니라 [" the", " cat"]이 됩니다.

Llama는 regex를 완전히 건너뛰는 SentencePiece를 사용합니다. raw byte stream을 하나의 긴 sequence로 다루고 BPE algorithm이 경계를 알아내게 합니다. 더 단순하지만 BPE가 cross-word token을 만들 자유를 더 많이 줍니다.

이 선택은 중요합니다. GPT-2의 regex는 tokenizer가 한 단어 끝의 "the"와 다음 단어 시작의 "the"를 merge해야 한다고 학습하지 못하게 합니다. SentencePiece는 이를 허용하며, 때로 더 효율적인 compression을 만들지만 token 해석 가능성은 낮아집니다.

### Special token

모든 production tokenizer는 구조적 marker를 위해 token ID를 예약합니다.

| Token | 목적 | 사용 모델 |
|-------|---------|---------|
| `[BOS]` / `<s>` | sequence 시작 | Llama 3, GPT |
| `[EOS]` / `</s>` | sequence 끝 | 모든 모델 |
| `[PAD]` | batch alignment용 padding | BERT, T5 |
| `[UNK]` | unknown token(byte-level BPE는 이를 제거) | BERT, WordPiece |
| `<\|im_start\|>` | chat message boundary 시작 | ChatGPT, Qwen |
| `<\|im_end\|>` | chat message boundary 끝 | ChatGPT, Qwen |
| `<\|user\|>` | user turn marker | Llama 3 |
| `<\|assistant\|>` | assistant turn marker | Llama 3 |

special token은 BPE로 절대 분리되지 않습니다. merge algorithm이 실행되기 전에 정확히 match되어 고정 ID로 대체되고, 주변 텍스트는 일반적으로 tokenization됩니다.

### Chat template

대부분의 사람이 헷갈리고 대부분의 구현이 깨지는 지점입니다.

chat model에 message를 보내면 API는 message list를 받습니다.

```json
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

모델은 JSON을 보지 않습니다. flat token sequence를 봅니다. chat template은 special token을 사용해 message를 그 flat sequence로 변환합니다. 모델마다 방식이 다릅니다.

```text
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

template을 잘못 만들면 모델은 엉망인 출력을 냅니다. 모델은 정확히 하나의 형식으로 학습되었습니다. newline 하나 누락, token 순서 변경, extra space 같은 어떤 이탈도 입력을 training distribution 밖으로 보냅니다.

### Speed

Python은 production tokenization에 너무 느립니다.

tiktoken(OpenAI)은 Rust로 작성되고 Python binding을 제공합니다. HuggingFace tokenizers도 Rust입니다. SentencePiece는 C++입니다. 이들은 순수 Python보다 10-100배 빠릅니다.

감각을 잡아봅시다. Llama 3 pre-training용 15조 token을 초당 1백만 token(빠른 Python)로 tokenization하면 174일이 걸립니다. 초당 1억 token(Rust)이라면 1.7일입니다.

여기서는 algorithm을 이해하기 위해 Python으로 만듭니다. production에서는 compiled implementation을 사용하고 Python wrapper만 다룹니다.

```figure
weight-tying
```

## 직접 만들기

### 1단계: Byte-level encoding

기초입니다. 어떤 문자열이든 byte sequence로 변환하고, 표시를 위해 각 byte를 printable character에 mapping한 뒤, 이 과정을 되돌립니다.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

multilingual text에서 테스트해 byte 수를 확인하세요.

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello"는 5 byte입니다. "你好"는 6 byte입니다(문자당 3개). 불 emoji는 4 byte입니다. byte-level tokenizer는 언어가 무엇인지 신경 쓰지 않습니다. byte는 byte입니다.

### Step 2: Regex 기반 Pre-Tokenizer

GPT-2 regex pattern으로 텍스트를 chunk로 나눕니다. 각 chunk는 BPE로 독립적으로 tokenization됩니다.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` module은 Unicode property escape(문자는 `\p{L}`, 숫자는 `\p{N}`)를 지원합니다. standard library `re` module은 이를 지원하지 않으므로 ASCII character class로 fallback합니다. production multilingual tokenizer에는 `regex`를 설치하세요.

실행해보세요.

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

leading space는 단어에 붙은 채 유지됩니다. contraction은 apostrophe에서 분리됩니다. 구두점은 자기 chunk가 됩니다. BPE는 이 경계를 가로질러 token을 merge하지 않습니다.

### Step 3: Byte Sequence 위의 BPE

Lesson 01의 핵심 algorithm이지만 이제 pre-tokenized chunk 각각에서 독립적으로 작동합니다.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Step 4: Special Token 처리

special token에는 exact matching과 fixed ID가 필요합니다. 이들은 BPE를 완전히 우회합니다.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Step 5: 전체 Tokenizer Class

모든 것을 연결합니다. normalize, special token 기준 분리, pre-tokenize, BPE merge, ID mapping 순서입니다.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 6단계: Multilingual test

진짜 테스트입니다. 영어, 중국어, emoji, code를 던져봅니다.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

중국어 문자는 각각 3 byte를 만듭니다. emoji는 4 byte를 만듭니다. 이 중 어떤 것도 tokenizer를 crash시키지 않습니다. unknown token도 만들지 않습니다. 이것이 byte-level BPE의 힘입니다.

## 활용하기

### 실제 Tokenizer 비교

Llama 3, GPT-4, Mistral의 실제 tokenizer를 load합니다. 각각이 같은 multilingual paragraph를 어떻게 처리하는지 봅니다.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

같은 텍스트에 대해 서로 다른 token count를 보게 될 것입니다. 128K vocabulary를 가진 Llama 3는 흔한 pattern을 더 공격적으로 merge합니다. 100K의 GPT-4는 중간에 있습니다. 32K의 Mistral은 더 많은 token을 만들지만 embedding layer가 더 작습니다.

tradeoff는 언제나 같습니다. vocabulary가 클수록 sequence는 짧아지지만 parameter는 많아집니다.

## 산출물

이 lesson은 production tokenizer를 구축하고 debug하기 위한 prompt를 산출합니다. `outputs/prompt-tokenizer-builder.md`를 참고하세요.

## 연습문제

1. **Easy:** 어떤 token ID에 대해서든 raw byte를 보여주는 `get_token_bytes(id)` method를 추가하세요. 가장 흔한 merged token이 실제로 무엇을 나타내는지 검사하는 데 사용하세요.
2. **Medium:** 공백과 digit을 기준으로 나누되 leading space는 유지하는 Llama-style pre-tokenizer를 구현하세요. 같은 corpus에서 GPT-2 regex 접근의 vocabulary와 비교하세요.
3. **Hard:** `{"role": ..., "content": ...}` message list를 받아 Llama 3 chat format에 맞는 올바른 token sequence를 만드는 chat template method를 추가하세요. HuggingFace implementation과 비교 테스트하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|----------------|----------------------|
| Byte-level BPE | "byte에서 작동하는 tokenizer" | 256개 byte 값을 base vocabulary로 갖는 BPE입니다. unknown token 없이 모든 입력을 처리합니다 |
| Pre-tokenization | "BPE 전 분리" | BPE가 단어 경계를 가로질러 merge하지 못하게 하는 regex 또는 rule-based splitting입니다 |
| NFKC normalization | "Unicode cleanup" | canonical decomposition 뒤 compatibility composition을 적용합니다. "fi" ligature는 "fi"가 되고, fullwidth "A"는 "A"가 됩니다 |
| Chat template | "message가 token이 되는 방식" | role/content message list를 flat token sequence로 변환하는 정확한 형식입니다. 모델별이며 training format과 일치해야 합니다 |
| Special tokens | "control token" | BPE를 우회하는 예약 token ID입니다. [BOS], [EOS], [PAD], chat marker는 merge 전에 정확히 match됩니다 |
| Fertility | "단어당 token 수" | 입력 단어 대비 출력 token 비율입니다. GPT-4의 영어는 1.3, 한국어는 2-3이며, 높을수록 context 낭비를 뜻합니다 |
| tiktoken | "OpenAI tokenizer" | Python binding을 갖춘 Rust BPE implementation입니다. 순수 Python보다 10-100배 빠릅니다 |
| Merge table | "vocabulary" | 학습 중 배운 byte-pair merge의 순서 있는 목록입니다. 이것이 tokenizer의 학습된 지식입니다 |

## 더 읽을거리

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- GPT-3.5/4에서 사용하는 Rust BPE implementation
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- BPE, WordPiece, Unigram을 지원하는 Rust tokenizer library
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- 128K vocabulary와 tokenizer training 세부사항
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- language-agnostic tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- 원래의 byte-to-Unicode mapping
