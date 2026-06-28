# BERT — Masked Language Modeling

> GPT는 다음 단어를 예측한다. BERT는 빠진 단어를 예측한다. 문장 하나의 차이가 embedding 형태를 가진 모든 것의 반 decade를 바꿨다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 minutes

## 문제

2018년에는 sentiment, NER, QA, entailment 같은 모든 NLP task가 각자의 labeled data로 자기 model을 처음부터 학습했다. fine-tune할 수 있는, 미리 학습된 "영어를 이해하는" checkpoint가 없었다. ELMo(2018)는 bidirectional LSTM으로 contextual embedding을 pre-train할 수 있음을 보였다. 도움이 되었지만 일반화는 제한적이었다.

BERT(Devlin et al. 2018)는 이렇게 물었다. transformer encoder를 가져와 인터넷의 모든 문장으로 학습시키고, 양쪽 context를 보고 빠진 단어를 예측하게 하면 어떨까? 그런 다음 downstream task에 head 하나만 fine-tune하면 된다. parameter efficiency는 놀라운 변화였다.

결과적으로 18개월 안에 BERT와 그 변형들(RoBERTa, ALBERT, ELECTRA)이 존재하던 모든 NLP leaderboard를 장악했다. 2020년에는 지구상의 모든 search engine, content moderation pipeline, semantic-search system 안에 BERT가 들어 있었다.

2026년에도 encoder-only model은 classification, retrieval, structured extraction에 적합한 도구다. decoder보다 token당 5-10× 빠르게 실행되며, 그 embedding은 모든 현대 retrieval stack의 backbone이다. ModernBERT(2024년 12월)는 Flash Attention + RoPE + GeGLU로 architecture를 8K context까지 밀어 올렸다.

## 개념

![Masked language modeling: token을 고르고 mask한 뒤 원래 token을 예측](../assets/bert-mlm.svg)

### 학습 신호

문장 하나를 보자: `the quick brown fox jumps over the lazy dog`.

token의 15%를 무작위로 mask한다.

```text
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

model은 masked position의 원래 token을 예측하도록 학습한다. encoder는 양방향이므로 position 1의 `[MASK]`를 예측할 때 position 2 이후의 `brown fox jumps`를 사용할 수 있다. 이것이 GPT가 할 수 없는 일이다.

### BERT mask 규칙

예측 대상으로 선택된 token 15% 중에서:

- 80%는 `[MASK]`로 교체한다.
- 10%는 random token으로 교체한다.
- 10%는 그대로 둔다.

왜 항상 `[MASK]`를 쓰지 않을까? `[MASK]`는 inference time에는 나타나지 않기 때문이다. model이 masked position 100%에서 `[MASK]`를 기대하도록 학습하면 pretraining과 fine-tuning 사이에 distribution shift가 생긴다. 10% random + 10% unchanged는 model이 현실적인 입력에도 대응하게 만든다.

### Next Sentence Prediction (NSP) — 그리고 왜 버려졌는가

원래 BERT는 NSP도 학습했다. sentence A와 B가 주어졌을 때 B가 A 다음에 오는지 예측하는 task다. RoBERTa(2019)는 이를 ablation했고, NSP가 도움이 아니라 해가 된다는 것을 보였다. 현대 encoder는 NSP를 건너뛴다.

### 2026년에 달라진 점: ModernBERT

2024년 ModernBERT 논문은 block을 2026년 primitive로 다시 만들었다.

| Component | Original BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| Positional | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | Alternating local (128) + global |
| Context length | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

또한 2018년 stack과 달리 Flash-Attention-native다. sequence length 8K에서 DeBERTa-v3보다 inference가 2-3× 빠르며 GLUE score도 더 좋다.

### 2026년에도 encoder를 고르는 use case

| 작업 | encoder가 decoder를 이기는 이유 |
|------|---------------------------|
| Retrieval / semantic search embeddings | 양방향 context = token당 더 좋은 embedding quality |
| Classification (sentiment, intent, toxicity) | forward pass 한 번이면 된다. generation overhead가 없다 |
| NER / token labeling | position별 output, 기본적으로 양방향 |
| Zero-shot entailment (NLI) | encoder 위에 classifier head를 얹는다 |
| Reranker for RAG | cross-encoder scoring, LLM reranker보다 10x 빠름 |

```figure
transformer-residual
```

## 직접 만들기

### 1단계: masking logic

`code/main.py`를 보라. `create_mlm_batch` 함수는 token ID list, vocab size, mask probability를 받는다. input ID(mask 적용됨)와 label(masked position에만 있고 나머지는 -100, 즉 PyTorch의 ignore index 관례)을 반환한다.

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 2단계: tiny corpus에서 MLM prediction 실행

20개 단어 vocabulary와 200개 sentence로 2-layer encoder + MLM head를 학습한다. gradient는 쓰지 않는다. forward-pass sanity check다. 전체 training에는 PyTorch가 필요하다.

### 3단계: mask type 비교

세 갈래 규칙이 `[MASK]` 없이도 model을 usable하게 유지하는 방식을 보여 준다. unmasked sentence와 masked sentence에서 예측해 보라. model은 training 중 두 pattern을 모두 봤으므로 둘 다 합리적인 token distribution을 내야 한다.

### 4단계: fine-tune head

MLM head를 toy sentiment dataset의 classification head로 교체한다. head만 학습하고 encoder는 frozen 상태로 둔다. 이것이 모든 BERT application이 따르는 pattern이다.

## 활용하기

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding model은 fine-tuned BERT다.** `all-MiniLM-L6-v2` 같은 `sentence-transformers` model은 contrastive loss로 학습한 BERT다. encoder는 같다. 바뀐 것은 loss다.

**Cross-encoder reranker도 fine-tuned BERT다.** `[CLS] query [SEP] doc [SEP]`에 대한 pair-classification이다. query와 doc 사이의 bidirectional attention이 cross-encoder가 biencoder보다 품질 우위를 갖는 정확한 이유다.

**2026년에 BERT를 고르지 말아야 할 때.** 생성이 필요한 모든 경우다. encoder에는 token을 autoregressive하게 만들어 낼 합리적인 방법이 없다. 또한 1B parameter 미만에서 작은 decoder가 더 높은 유연성으로 품질을 맞출 수 있는 경우(Phi-3-Mini, Qwen2-1.5B)도 해당한다.

## 실전 적용

`outputs/skill-bert-finetuner.md`를 보라. 이 skill은 새로운 classification 또는 extraction task를 위한 BERT fine-tune 범위(backbone choice, head spec, data, eval, stopping)를 잡는다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하고 10,000 token에 대한 mask distribution을 출력하라. 약 15%가 선택되고, 그중 약 80%가 `[MASK]`가 되는지 확인하라.
2. **보통.** whole-word masking을 구현하라. 단어가 subword로 tokenized되면 모든 subword를 함께 mask하거나 모두 mask하지 않는다. 500-sentence corpus에서 이것이 MLM accuracy를 개선하는지 측정하라.
3. **어려움.** public dataset의 10,000 sentence로 tiny(2-layer, d=64) BERT를 학습하라. SST-2 sentiment에 대해 `[CLS]` token을 fine-tune하라. parameter 수를 맞춘 decoder-only baseline과 비교하라. 어느 쪽이 이기는가?

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| MLM | "Masked language modeling" | training signal: token 15%를 무작위로 `[MASK]`로 바꾸고 원래 token을 예측한다. |
| Bidirectional | "양쪽을 본다" | encoder attention에는 causal mask가 없다. 모든 position이 다른 모든 position을 본다. |
| `[CLS]` | "Pooler token" | 모든 sequence 앞에 붙는 special token. 마지막 embedding을 sentence-level representation으로 사용한다. |
| `[SEP]` | "Segment separator" | paired sequence를 분리한다(예: query/doc, sentence A/B). |
| NSP | "Next sentence prediction" | BERT의 두 번째 pretraining task. RoBERTa에서 쓸모없음이 드러났고 2019년 이후 버려졌다. |
| Fine-tuning | "task에 맞게 적응" | encoder는 대부분 frozen 상태로 유지하고 downstream task용 작은 head를 위에 학습한다. |
| Cross-encoder | "Reranker" | query와 doc을 함께 input으로 받아 relevance score를 출력하는 BERT. |
| ModernBERT | "2024 refresh" | RoPE, RMSNorm, GeGLU, alternating local/global attention, 8K context로 다시 만든 encoder. |

## 더 읽을거리

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 원 논문.
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — BERT를 제대로 학습하는 법. NSP를 제거한다.
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — replaced-token detection은 같은 compute에서 MLM보다 낫다.
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 논문.
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — canonical encoder reference.
