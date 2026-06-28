# 기계 번역

> 번역은 30년 동안 NLP 연구비를 만들어 낸 과제였고, 지금도 계속 그 역할을 하고 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 minutes

## 문제

모델은 한 언어의 문장을 읽고 다른 언어의 문장을 생성한다. 길이는 달라진다. 어순도 달라진다. 어떤 원문 단어는 여러 대상 단어로 매핑되고, 그 반대도 생긴다. 관용구는 일대일 매핑을 거부한다. 프랑스어에서 "I miss you"는 "tu me manques"다. 직역하면 "너는 나에게 부족하다"에 가깝다. 이런 경우 단어 수준 정렬은 살아남지 못한다.

기계 번역은 NLP가 encoder-decoder, attention, transformer, 그리고 결국 전체 LLM 패러다임을 발명하도록 밀어붙인 과제다. 번역 품질은 측정 가능했고 인간과 기계 사이의 격차는 끈질겼기 때문에, 모든 진전은 번역에서 비롯되었다.

이 수업은 역사 수업을 건너뛰고 2026년의 실무 파이프라인을 가르친다. pretrained multilingual encoder-decoder(NLLB-200 또는 mBART), subword tokenization, beam search, BLEU와 chrF 평가, 그리고 아직도 프로덕션에 잡히지 않은 채 배포되는 몇 가지 failure mode를 다룬다.

## 개념

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

현대 MT는 parallel text로 학습된 transformer encoder-decoder다. Encoder는 원문을 해당 언어의 tokenization으로 읽는다. Decoder는 encoder의 출력을 cross-attention(10과)을 통해 사용하면서 한 번에 하나의 subword를 생성해 대상을 만든다. Decoding은 greedy decoding 함정을 피하기 위해 beam search를 사용한다. 출력은 detokenize와 detruecase를 거친 뒤 reference와 비교해 점수를 매긴다.

현실 세계의 MT 품질은 세 가지 운영 선택이 좌우한다.

- **Tokenizer.** 혼합 언어 corpus에서 학습한 SentencePiece BPE. 언어 간 shared vocabulary가 NLLB에서 zero-shot pair를 가능하게 한다.
- **Model size.** NLLB-200 distilled 600M은 노트북에서 동작한다. NLLB-200 3.3B는 공개된 production default다. 54.5B는 research ceiling이다.
- **Decoding.** 일반 콘텐츠에는 beam width 4-5. 너무 짧은 출력을 피하려면 length penalty. 용어 일관성이 필요하면 constrained decoding.

```figure
seq2seq-alignment
```

## 직접 만들기

### 1단계: pretrained MT 호출

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

여기서 중요한 것은 세 가지다. `src_lang`은 tokenizer에 어떤 script와 segmentation을 적용할지 알려 준다. `forced_bos_token_id`는 decoder에 어떤 언어를 생성할지 알려 준다. 둘 다 NLLB 전용 기법이다. mBART와 M2M-100은 각자의 규약을 사용하며 서로 바꿔 쓸 수 없다.

### 2단계: BLEU와 chrF

BLEU는 output과 reference 사이의 n-gram overlap을 측정한다. 네 가지 reference n-gram 크기(1-4), precision의 geometric mean, 너무 짧은 출력에 대한 brevity penalty를 쓴다. 점수 범위는 [0, 100]이다. 널리 사용된다. 해석은 까다롭다. BLEU 30은 "사용 가능", 40은 "좋음", 50은 "예외적"이고, 1 BLEU 미만 차이는 noise다.

chrF는 character-level F-score를 측정한다. BLEU가 match를 과소계산하는 morphologically rich language에 더 민감하다. 보통 BLEU와 함께 보고한다.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

항상 `sacrebleu`를 사용하라. 이 도구는 tokenization을 정규화해 논문 간 점수를 비교 가능하게 만든다. 직접 BLEU 계산을 만드는 것은 오해를 부르는 benchmark가 생기는 길이다.

### 3단계 평가 계층(2026)

현대 MT 평가는 서로 보완적인 세 가지 metric family를 사용한다. 최소 두 가지는 함께 배포하라.

- **Heuristic** (BLEU, chrF). 빠르고, reference 기반이며, 해석 가능하지만 paraphrase에 둔감하다. legacy comparison과 regression detection에 사용한다.
- **Learned** (COMET, BLEURT, BERTScore). 인간 판단으로 학습한 neural model이다. 번역과 source/reference 사이의 semantic similarity를 비교한다. COMET은 2023년 이후 MT 연구와의 관련성이 가장 높고, 품질이 중요한 곳에서 2026년 production default다.
- **LLM-as-judge** (reference-free). 큰 모델에 prompt를 주어 fluency, adequacy, tone, cultural appropriateness를 평가하게 한다. Rubric이 잘 설계되면 GPT-4-as-judge는 약 80%의 경우 인간 합의와 맞는다. Reference가 없는 open-ended content에 사용한다.

실용적인 2026 stack은 BLEU와 chrF에는 `sacrebleu`, COMET에는 `unbabel-comet`, 최종 human-facing signal에는 prompted LLM이다. 어떤 metric이든 production data에 신뢰하기 전에 50-100개의 human-labeled example에 맞춰 보정하라.

Reference-free metric(COMET-QE, BLEURT-QE, LLM-as-judge)은 reference 없이 번역을 평가하게 해 준다. Reference translation이 존재하지 않는 long-tail language pair에서 중요하다.

### 3단계: 프로덕션에서 깨지는 것들

위의 실무 파이프라인은 80%의 시간 동안 유창하게 번역하고 나머지 20%는 조용히 실패한다. 이름이 붙은 failure mode는 다음과 같다.

- **Hallucination.** 모델이 source에 없던 내용을 만들어 낸다. 익숙하지 않은 domain vocabulary에서 흔하다. 증상은 출력은 유창하지만 source가 말하지 않은 사실을 주장하는 것이다. 완화책은 domain term에 대한 constrained decoding, regulated content에 대한 human review, input보다 훨씬 긴 output 모니터링이다.
- **Off-target generation.** 모델이 잘못된 언어로 번역한다. NLLB는 rare language pair에서 의외로 이 문제에 취약하다. 완화책은 `forced_bos_token_id`를 확인하고 output에 language-ID model check를 항상 적용해 decoding하는 것이다.
- **Terminology drift.** "Sign up"이 문서 1에서는 "s'inscrire"가 되고 문서 2에서는 "créer un compte"가 된다. UI text와 user-facing string에서는 raw quality보다 consistency가 더 중요하다. 완화책은 glossary-constrained decoding 또는 post-edit dictionary다.
- **Formality mismatch.** 프랑스어 "tu"와 "vous", 일본어의 공손도 수준. 모델은 training에서 더 흔했던 형식을 고른다. Customer-facing content에서는 보통 틀린 선택이다. 완화책은 모델이 지원한다면 formality token이 있는 prompt prefix를 쓰거나, formal-only corpus로 작은 모델을 fine-tune하는 것이다.
- **Length explosion on short input.** 매우 짧은 input sentence는 길이 penalty가 source token 약 5개 아래에서 급격히 무너지기 때문에 지나치게 긴 번역을 자주 만든다. 완화책은 source length에 비례하는 hard max-length cap이다.

### 4단계: domain fine-tuning

Pretrained model은 generalist다. 법률, 의료, 게임 대화 번역은 domain parallel data로 fine-tuning하면 측정 가능한 이득을 얻는다. 방법은 특별하지 않다.

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

고품질 parallel example 몇천 개가 noisy web-scraped example 몇십만 개보다 낫다. Training data 품질은 production에서 가장 큰 lever다.

## 사용하기

2026년 MT production stack은 다음과 같다.

| 사용 사례 | 권장 시작점 |
|---------|---------------------------|
| Any-to-any, 200개 언어 | `facebook/nllb-200-distilled-600M` (laptop) 또는 `nllb-200-3.3B` (production) |
| English-centric, 고품질, 50개 언어 | `facebook/mbart-large-50-many-to-many-mmt` |
| 짧은 실행, 저렴한 inference, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Latency-critical browser-side | ONNX-quantized Marian (~50 MB) |
| 최고 품질, 비용 지불 가능 | GPT-4 / Claude / Gemini with translation prompts |

2026년 기준 LLM은 여러 language pair에서 specialized MT model을 능가한다. 특히 idiomatic content와 long context에서 그렇다. Tradeoff는 per-token cost와 latency다. Context length, stylistic consistency, prompt를 통한 domain adaptation이 throughput보다 중요하면 LLM을 선택하라.

## 배포하기

`outputs/skill-mt-evaluator.md`로 저장하라.

```markdown
---
name: mt-evaluator
description: shipping 가능한 machine translation output인지 평가합니다.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

source text와 candidate translation이 주어지면 다음을 출력하라.

1. 자동 점수 추정. 예상되는 BLEU와 chrF 범위. Reference 사용 가능 여부를 명시한다.
2. 인간이 검증할 수 있는 5-point checklist: (a) content preservation(no hallucinations), (b) correct language, (c) register / formality match, (d) glossary가 제공된 경우 terminology consistency, (e) truncation 또는 length explosion 없음.
3. 조사할 domain-specific issue 하나. 예: legal에서는 named entities와 statute citations. medical에서는 drug names와 dosages. UI에서는 placeholder variables `{name}`.
4. 신뢰도 flag. "Ship" / "Ship with review" / "Do not ship". 2단계에서 발견한 issue의 severity에 연결한다.

Output에 language-ID check가 없으면 번역 배포를 거부하라. 사용자가 reference-free scoring(COMET-QE, BLEURT-QE)을 명시적으로 선택하지 않는 한 reference 없이 평가하지 말라. 1000 token을 넘는 content는 chunked translation이 필요할 가능성이 높다고 표시하라.
```

## 연습문제

1. **쉬움.** `nllb-200-distilled-600M`을 사용해 5문장짜리 영어 문단을 프랑스어로 번역한 뒤 다시 영어로 되돌려라. Round-trip이 원문과 얼마나 가까운지 측정하라. Word-choice drift가 있으면서 semantic preservation이 보일 것이다.
2. **중간.** `fasttext lid.176` 또는 `langdetect`를 사용해 translation output에 대한 language-ID check를 구현하라. Off-target generation이 반환 전에 잡히도록 MT 호출에 통합하라.
3. **어려움.** 원하는 domain corpus 5,000 pair로 `nllb-200-distilled-600M`을 fine-tune하라. Fine-tuning 전후 held-out set에서 BLEU를 측정하라. 어떤 종류의 sentence가 개선되었고 어떤 종류가 퇴보했는지 보고하라.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| BLEU | Translation score | Brevity penalty가 있는 N-gram precision. [0, 100]. |
| chrF | Character F-score | Character-level F-score. Morphologically rich language에 더 민감하다. |
| NMT | Neural MT | Parallel text로 학습한 transformer encoder-decoder. 2017년 이후의 default. |
| NLLB | No Language Left Behind | Meta의 200-language MT model family. |
| Constrained decoding | Controlled output | 특정 token 또는 n-gram이 output에 나타나거나 나타나지 않도록 강제한다. |
| Hallucination | Invented content | Source가 뒷받침하지 않는 model output. |

## 더 읽기

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB paper.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — `sacrebleu`가 BLEU 보고의 유일하게 올바른 방식인 이유.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF paper.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 실용적인 fine-tuning walkthrough.
