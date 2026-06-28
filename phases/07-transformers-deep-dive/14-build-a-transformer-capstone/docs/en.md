# 트랜스포머를 처음부터 만들기 — 캡스톤

> 열세 개의 lesson. 하나의 모델. 지름길은 없다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 minutes

## 문제

당신은 모든 논문을 읽었다. attention, multi-head 분할, positional encoding, encoder와 decoder 블록, BERT와 GPT 손실, MoE, KV cache를 구현했다. 이제 그것들이 실제 작업에서 함께 동작하게 만들어야 한다.

캡스톤: 문자 수준 언어 모델링 작업에서 작은 decoder-only 트랜스포머를 end-to-end로 학습한다. Shakespeare를 읽고, 새로운 Shakespeare를 생성한다. 노트북에서 10분 안에 학습할 만큼 작다. 더 큰 데이터셋과 더 긴 학습으로 바꾸면 실제 LM으로 이어질 만큼 올바르다.

이것은 이 과정의 "nanoGPT"다. 독창적인 것은 아니다. Karpathy의 2023년 nanoGPT 튜토리얼은 모든 학생이 한 번은 작성하는 기준 구현이다. 우리는 그 형태를 가져오되 지금까지 다룬 내용에 맞게 다시 구성한다.

## 개념

![처음부터 만드는 트랜스포머 블록 다이어그램](../assets/capstone.svg)

주석을 단 아키텍처:

```text
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 우리가 제공하는 것

- `GPTConfig` — 모든 하이퍼파라미터를 설정하는 한 곳.
- `MultiHeadAttention` — causal, batched, 선택적 Flash 스타일 경로(PyTorch의 `scaled_dot_product_attention`) 포함.
- `SwiGLUFFN` — 현대적인 FFN.
- `Block` — pre-norm, residual로 감싼 attention + FFN.
- `GPT` — embeddings, 쌓인 블록, LM head, generate().
- AdamW, cosine LR, gradient clipping이 있는 훈련 루프.
- Shakespeare 텍스트용 문자 수준 tokenizer.

### 우리가 제공하지 않는 것

- RoPE — Lesson 04에서 개념적으로 구현했다. 여기서는 단순성을 위해 learned positional embeddings를 쓴다. 연습 문제에서 RoPE로 바꾸게 된다.
- 생성 중 KV cache — 각 생성 단계는 전체 prefix에 대해 attention을 다시 계산한다. 느리지만 단순하다. 연습 문제에서 KV cache를 추가하게 된다.
- Flash Attention — PyTorch 2.0+는 입력이 맞으면 자동 dispatch한다. 우리는 `F.scaled_dot_product_attention`을 쓴다.
- MoE — 블록마다 단일 FFN을 쓴다. Lesson 11에서 MoE를 봤다.

### 목표 지표

Mac M2 노트북에서 4-layer, 4-head, d_model=128 GPT를 `tinyshakespeare.txt`로 2,000스텝 학습하면:

- 훈련 손실이 약 6분 만에 ~4.2(무작위)에서 ~1.5로 수렴한다.
- 샘플 출력은 Shakespeare다운 모양을 보인다. 고어, 줄바꿈, "ROMEO:" 같은 고유명이 나타난다.
- 검증 손실(텍스트의 마지막 10% hold-out)은 훈련 손실을 가깝게 따라간다. 이 크기와 예산에서는 과적합이 없다.

## 직접 만들기

이 lesson은 PyTorch를 사용한다. `torch`를 설치하라(CPU 빌드도 괜찮다). `code/main.py`를 보라. 스크립트는 다음을 처리한다.

- `tinyshakespeare.txt`가 없으면 다운로드한다(또는 로컬 복사본을 읽는다).
- 바이트 수준 문자 tokenizer.
- 90/10 train/val 분할.
- 지원 하드웨어에서 bf16 autocast를 쓰는 훈련 루프.
- 훈련 완료 후 샘플링.

### 1단계: 데이터

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

고유 문자는 65개다. 아주 작은 vocabulary다. 4바이트 vocab_size에 맞는다. BPE도 없고, tokenizer 문제도 없다.

### 2단계: 모델

`code/main.py`를 보라. 블록은 Lesson 05에서 배운 전형적인 형태다. pre-norm, RMSNorm, SwiGLU, causal MHA. 4/4/128의 파라미터 수는 약 800K다.

### 3단계: 훈련 루프

길이 256 토큰 window의 무작위 배치를 얻는다. Forward. Shift-by-one cross-entropy. Backward. AdamW step. Log. 반복.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 4단계: 샘플링

프롬프트가 주어지면 forward를 반복하고, top-p logits에서 샘플링해 붙인 뒤 계속한다. 500토큰 뒤 멈춘다.

### 5단계: 출력 읽기

2,000스텝 뒤:

```text
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Shakespeare는 아니다. 하지만 Shakespeare다운 모양이다. 약 800K 파라미터와 노트북 6분으로 얻은 분명한 성과다.

## 활용하기

이 캡스톤은 기준 아키텍처다. 실제에 가까운 것으로 보내기 위한 세 가지 확장:

1. **Tokenizer를 바꾼다.** BPE를 사용한다(예: `tiktoken.get_encoding("cl100k_base")`). Vocabulary 크기는 65에서 약 50,000으로 뛴다. 이를 보상하려면 모델 용량도 커져야 한다.
2. **더 큰 코퍼스로 학습한다.** `OpenWebText` 또는 `fineweb-edu`(HuggingFace)를 쓴다. 단일 A100에서 125M-param GPT를 10B 토큰으로 학습하는 데 약 24시간이 걸린다.
3. **RoPE + KV cache + Flash Attention을 추가한다.** 아래 연습 문제가 각각을 안내한다.

결과는 유창한 영어를 생성하는 125M-파라미터 GPT가 된다. Frontier 모델은 아니다. 하지만 같은 코드 경로를 더 키운 것이 Karpathy, EleutherAI, Allen Institute가 2026년에 연구 체크포인트를 학습할 때 쓰는 방식이다.

## 출시하기

`outputs/skill-transformer-review.md`를 보라. 이 스킬은 앞선 13개 lesson 전체에 걸쳐 transformer-from-scratch 구현의 정확성을 검토한다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하라. 학습된 모델의 마지막 스텝 검증 손실이 2.0 미만인지 확인하라. `max_steps`를 2,000에서 5,000으로 바꾸면 검증 손실이 계속 개선되는가?
2. **중간.** Learned positional embeddings를 RoPE로 교체하라. `MultiHeadAttention` 내부에서 Q와 K에 회전을 적용하라. 학습한 뒤 검증 손실이 적어도 기존만큼 낮은지 확인하라.
3. **중간.** 샘플링 루프에 KV cache를 구현하라. 캐시가 있을 때와 없을 때 각각 500토큰을 생성하라. 노트북에서 wall-clock이 5-20배 개선되어야 한다.
4. **어려움.** 다음 다음 토큰을 예측하는 두 번째 head를 모델에 추가하라(MTP — DeepSeek-V3의 Multi-Token Prediction). 함께 학습하라. 도움이 되는가?
5. **어려움.** 블록당 단일 FFN을 4-expert MoE로 교체하라. Router + top-2 routing. 활성 파라미터를 맞춘 상태에서 검증 손실이 어떻게 변하는지 보라.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy의 튜토리얼 repo" | 최소 decoder-only 트랜스포머 훈련 코드, 약 300 LOC. 표준 기준 구현. |
| tinyshakespeare | "표준 장난감 코퍼스" | 약 1.1MB 텍스트. 2015년 이후 모든 character-LM 튜토리얼이 쓴다. |
| Tied embeddings | "입출력 행렬 공유" | LM head 가중치 = token embedding 행렬의 전치. 파라미터를 아끼고 품질을 높인다. |
| bf16 autocast | "훈련 정밀도 트릭" | forward/back은 bf16으로 실행하고 optimizer state는 fp32로 유지한다. 2021년 이후 표준. |
| Gradient clipping | "스파이크 방지" | 전역 grad norm을 1.0으로 제한한다. 훈련 폭주를 막는다. |
| Cosine LR schedule | "2020년대 이후 기본값" | LR을 선형으로 올린 뒤(warmup) peak의 10%까지 cosine 모양으로 낮춘다. |
| MFU | "Model FLOP Utilization" | 달성 FLOPs / 이론 peak. 2026년 기준 dense 40%, MoE 30%면 강하다. |
| Val loss | "Hold-out 손실" | 모델이 보지 않은 데이터의 교차 엔트로피. 과적합 탐지기. |

## 더 읽을거리

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 고전적인 주석 달린 구현.
