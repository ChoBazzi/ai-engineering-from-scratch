# T5, BART — Encoder-Decoder 모델

> encoder는 이해한다. decoder는 생성한다. 둘을 다시 합치면 translate, summarize, rewrite, transcribe 같은 input → output task에 맞게 만들어진 model을 얻는다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 minutes

## 문제

decoder-only GPT와 encoder-only BERT는 각각 다른 목표를 위해 2017년 architecture를 덜어낸 것이다. 하지만 많은 task는 자연스럽게 input-output 형태다.

- Translation: English → French.
- Summarization: 5,000-token article → 200-token summary.
- Speech recognition: audio tokens → text tokens.
- Structured extraction: prose → JSON.

이런 경우에는 encoder-decoder가 가장 깔끔하게 맞는다. encoder는 source의 dense representation을 만든다. decoder는 output을 생성하면서 매 step 그 representation에 cross-attend한다. training은 output 쪽 shift-by-one이다. GPT와 같은 loss지만 encoder output에 condition된다는 점만 다르다.

두 논문이 현대적 playbook을 정의했다.

1. **T5** (Raffel et al. 2019). "Text-to-Text Transfer Transformer." 모든 NLP task를 text-in, text-out으로 재구성했다. 하나의 architecture, 하나의 vocabulary, 하나의 loss. masked span prediction(input의 span을 corrupt하고 output에서 decode)으로 pretrained했다.
2. **BART** (Lewis et al. 2019). "Bidirectional and Auto-Regressive Transformer." denoising autoencoder다. input을 여러 방식으로 corrupt(shuffle, mask, delete, rotate)하고 decoder가 원본을 reconstruct하게 한다.

2026년에도 encoder-decoder format은 input structure가 중요한 곳에서 살아 있다.

- Whisper (speech → text).
- Google's translation stack.
- distinct context-and-edit structure를 가진 일부 code-completion / repair model.
- structured reasoning task용 Flan-T5와 변형들.

decoder-only가 spotlight를 가져갔지만 encoder-decoder는 사라진 적이 없다.

## 개념

![Cross-attention을 가진 encoder-decoder](../assets/encoder-decoder.svg)

### Forward loop 흐름

```text
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

중요한 점은 encoder가 input마다 한 번만 실행된다는 것이다. decoder는 autoregressive하게 실행되지만 매 step *같은* encoder output에 cross-attend한다. encoder output caching은 긴 input에서 공짜 speedup이다.

### T5 pretraining — span corruption

input의 random span을 고른다(평균 길이 3 token, 총 15%). 각 span을 고유 sentinel로 교체한다: `<extra_id_0>`, `<extra_id_1>` 등. decoder는 corrupted span만 sentinel prefix와 함께 output한다.

```text
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

전체 sequence를 예측하는 것보다 저렴한 signal이다. T5 논문의 ablation에서 MLM(BERT)과 prefix-LM(UniLM)에 경쟁력 있는 성능을 냈다.

### BART pretraining — multi-noise denoising

BART는 다섯 가지 noising function을 시도한다.

1. Token masking.
2. Token deletion.
3. Text infilling(span을 mask하고 decoder가 올바른 길이를 삽입).
4. Sentence permutation.
5. Document rotation.

text infilling + sentence permutation 조합이 downstream number에서 가장 좋았다. decoder는 항상 원본을 reconstruct한다. BART의 output은 corrupted span만이 아니라 full sequence다. 그래서 pretraining compute는 T5보다 높다.

### 추론

GPT와 같은 autoregressive generation이다. greedy / beam / top-p sampling이 적용된다. translation과 summarization에서는 output distribution이 chat보다 좁기 때문에 beam search(width 4-5)가 표준이다.

### 2026년에 각 variant를 고르는 기준

| 작업 | Encoder-decoder? | 이유 |
|------|------------------|-----|
| Translation | Yes, usually | 명확한 source sequence, 고정된 output distribution, beam search가 잘 작동 |
| Speech-to-text | Yes (Whisper) | input modality가 output과 다르고, encoder가 audio feature를 shaping |
| Chat / reasoning | No, decoder-only | 지속적인 "input"이 따로 없다. conversation 자체가 sequence |
| Code completion | Usually no | 긴 context의 decoder-only가 우세. Qwen 2.5 Coder 같은 code model은 decoder-only |
| Summarization | Either works | BART, PEGASUS는 초기 decoder-only baseline을 이겼고, 현대 decoder-only LLM은 이를 따라잡음 |
| Structured extraction | Either | T5는 "text → text"가 어떤 output format이든 흡수하므로 깔끔 |

2022년 무렵 이후의 추세는 decoder-only가 encoder-decoder가 맡던 task를 가져가는 것이다. 이유는 (a) instruction-tuned decoder-only LLM이 prompting으로 거의 모든 것에 일반화하고, (b) 하나의 architecture가 둘보다 scale하기 쉬우며, (c) RLHF가 decoder를 전제하기 때문이다. encoder-decoder는 input modality가 다르거나(speech, images) beam search 품질이 중요한 곳에서 버틴다.

## 직접 만들기

`code/main.py`를 보라. toy corpus에 대해 T5-style span corruption을 구현한다. 모든 encoder-decoder pretraining recipe에 계속 등장하기 때문에 이 lesson에서 가장 유용한 단일 조각이다.

### 1단계: span corruption

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

target format은 T5 convention이다: `<sent0> span0 <sent1> span1 ...`. corrupted input은 unchanged token과 span location의 sentinel token을 interleave한다.

### 2단계: round-trip 검증

corrupted input과 target이 주어졌을 때 original sentence를 reconstruct한다. corruption이 reversible이면 forward pass가 잘 정의된다. 이것은 sanity check다. 실제 training에서는 하지 않지만, test가 cheap하고 span bookkeeping의 off-by-one bug를 잡아 준다.

### 3단계: BART noising

다섯 함수: `token_mask`, `token_delete`, `text_infill`, `sentence_permute`, `document_rotate`. 그중 둘을 compose하고 결과를 보여 준다.

## 활용하기

HuggingFace reference:

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 trick은 task name을 input text에 넣는 것이다. 같은 model이 수십 가지 task를 처리한다. 각 task가 text-in, text-out이기 때문이다. 2026년에는 instruction-tuned decoder-only model이 이 pattern을 일반화했지만, 먼저 체계화한 것은 T5다.

## 실전 적용

`outputs/skill-seq2seq-picker.md`를 보라. 이 skill은 새 task의 input-output structure, latency, quality target을 기준으로 encoder-decoder와 decoder-only 중 무엇을 고를지 결정한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행하고 30-token sentence에 span corruption을 적용하라. non-sentinel source token과 decoded target span을 이어 붙이면 original이 재현되는지 확인하라.
2. **보통.** BART의 `text_infill` noise를 구현하라. random span을 single `<mask>` token으로 바꾸고 decoder가 올바른 span length와 contents를 추론해야 한다. 예시 하나를 보여라.
3. **어려움.** tiny English → pig-Latin corpus(200 pairs)에서 `flan-t5-small`을 fine-tune하라. held-out 50-pair set에서 BLEU를 측정하라. 같은 data와 같은 compute로 `Llama-3.2-1B`를 fine-tune한 결과와 비교하라.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Encoder-decoder | "Seq2seq transformer" | 두 stack: input용 bidirectional encoder, output용 cross-attention을 가진 causal decoder. |
| Cross-attention | "source가 target에 말하는 곳" | decoder의 Q × encoder의 K/V. encoder 정보가 decoder로 들어가는 유일한 지점. |
| Span corruption | "T5의 pretraining trick" | random span을 sentinel token으로 바꾸고 decoder가 span을 output한다. |
| Denoising objective | "BART의 game" | input에 noise function을 적용하고 decoder가 clean sequence를 reconstruct하도록 학습한다. |
| Sentinel token | "`<extra_id_N>` placeholder" | source에서 corrupted span을 표시하고 target에서 다시 표시하는 special token. |
| Flan | "Instruction-tuned T5" | 1,800개가 넘는 task로 fine-tune된 T5. encoder-decoder가 instruction-following에서 경쟁력을 갖게 했다. |
| Beam search | "Decoding strategy" | 각 step에서 top-k partial sequence를 유지한다. translation/summarization의 표준. |
| Teacher forcing | "Training-time input" | training 중에는 sampled token이 아니라 true previous output token을 decoder에 넣는다. |

## 더 읽을거리

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART.
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5.
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper, canonical 2026 encoder-decoder.
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — reference implementation.
