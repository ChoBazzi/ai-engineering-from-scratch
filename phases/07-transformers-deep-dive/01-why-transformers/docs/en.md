# 왜 Transformer인가 — RNN의 문제들

> RNN은 token을 한 번에 하나씩 처리합니다. Transformer는 모든 token을 한꺼번에 처리합니다. 이 하나의 아키텍처적 선택이 2017년 이후 deep learning의 모든 scaling curve를 바꿨습니다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 minutes

## 문제

2017년 전에는 지구상의 모든 최첨단 sequence model, 즉 언어, 번역, 음성 모델이 recurrent neural network였습니다. LSTM과 GRU는 반 decade 동안 ImageNet급 번역 benchmark에서 승리했습니다. 누구에게나 그것이 유일한 도구였습니다.

하지만 세 가지 치명적인 약점이 있었습니다. Sequential computation 때문에 시간 축을 따라 병렬화할 수 없었습니다. token `t+1`은 token `t`의 hidden state가 필요합니다. 1,024-token sequence는 cycle당 1,000,000개의 floating-point op를 처리할 수 있는 GPU 위에서도 1,024개의 직렬 step을 뜻했습니다. 병렬 처리를 위해 설계된 hardware에서 training wall-clock time은 sequence length에 선형으로 증가했습니다.

Vanishing gradient는 50 token 전의 정보가 이미 50개의 non-linearity를 거쳐 압축되었다는 뜻이었습니다. Gated recurrent unit(LSTM, GRU)은 이 압착을 완화했지만 없애지는 못했습니다. "the book I read last summer on a plane to Kyoto was..." 같은 long-range dependency는 자주 실패했습니다.

Fixed-width hidden state는 decoder가 무엇을 보기 전에 encoder가 전체 source sequence를 단일 vector로 밀어 넣어야 한다는 뜻이었습니다. source가 5 token이든 500 token이든 상관없습니다. bottleneck은 같은 shape입니다.

2017년 논문 "Attention Is All You Need"는 급진적인 제안을 했습니다. recurrence를 완전히 버리자는 것이었습니다. 모든 position이 다른 모든 position에 parallel하게 attend하도록 만들었습니다. 1,024개의 sequential matmul 대신 하나의 큰 matrix multiplication으로 train합니다.

그 결과는 2026년 기준 모든 modality를 지배합니다. Language(GPT-5, Claude 4, Llama 4), vision(ViT, DINOv2, SAM 3), audio(Whisper), biology(AlphaFold 3), robotics(RT-2). 같은 block, 다른 input입니다.

## 개념

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Bottleneck으로서의 recurrence.** RNN은 `h_t = f(h_{t-1}, x_t)`를 계산합니다. 각 step은 이전 step에 의존합니다. `h_4` 전에 `h_5`를 계산할 수 없습니다. 10,000개 이상의 parallel core가 있는 현대 GPU에서 긴 sequence에 이 방식은 silicon의 99%를 낭비합니다.

**Broadcast로서의 attention.** Self-attention은 모든 pair `(i, j)`에 대해 `output_i = sum_j(a_ij * v_j)`를 동시에 계산합니다. 전체 N×N attention matrix가 하나의 batched matmul로 채워집니다. 어떤 step도 다른 step에 의존하지 않습니다. GPU가 좋아하는 형태입니다.

**Speedup은 상수가 아닙니다.** 이는 `O(N)` serial depth와 `O(1)` serial depth의 차이입니다. 실제로 transformer는 N=512에서 같은 hardware 기준 epoch당 5-10배 빠르게 train되며, sequence length가 길어질수록 attention의 `O(N²)` memory wall에 닿기 전까지 격차가 커집니다. 이후 Flash Attention이 이 문제를 완화합니다. Lesson 12를 보세요.

**Transformer의 비용.** Attention memory는 `O(N²)`로 증가합니다. 2K context에서는 괜찮습니다. 128K context에서는 sliding window, RoPE extrapolation, Flash Attention tiling, 또는 linear attention variant가 필요합니다. Recurrence는 time과 memory 모두 `O(N)`였습니다. Transformer는 memory를 내주고 time을 얻은 다음, parallelism으로 time을 다시 크게 회수합니다.

**Inductive bias의 변화.** RNN은 locality와 recency를 가정합니다. Transformer는 아무것도 가정하지 않습니다. 모든 pair가 attention 후보입니다. 그래서 transformer는 잘 train되려면 더 많은 data가 필요하지만, data가 충분해지면 더 멀리 scale됩니다. Chinchilla(2022)는 이를 공식화했습니다. token이 충분하면 transformer는 같은 parameter count의 RNN을 항상 이깁니다.

## 직접 만들기

여기서는 neural network를 만들지 않습니다. laptop에서 bottleneck의 차이를 체감하도록 핵심 bottleneck을 수치적으로 simulate합니다.

### 1단계: serial depth 측정하기

`code/main.py`를 보세요. 두 function을 만듭니다. 하나는 sequence를 addition chain으로 encode합니다(serial, RNN과 유사). 다른 하나는 parallel reduction으로 encode합니다(broadcast, attention과 유사). 같은 수학, 다른 dependency graph입니다.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

최대 100,000개 element의 sequence에서 둘의 시간을 잽니다. RNN version은 O(N)이고 단일 CPU pipeline을 탑니다. 순수 Python에서도 length가 1,000 이상이면 attention-style reduction이 앞섭니다. Python의 `sum()`은 C로 구현되어 있고 step마다 interpreter overhead 없이 iterate하기 때문입니다.

### 2단계: theoretical operation 세기

두 algorithm 모두 N번 add합니다. 차이는 *dependency depth*입니다. 다음 operation이 시작되기 전에 순차적으로 일어나야 하는 operation의 수입니다. RNN depth = N입니다. Attention depth는 tree reduction이면 log(N), parallel scan이면 1입니다. GPU time을 결정하는 것은 op count가 아니라 depth입니다.

### 3단계: 긴 sequence에서 empirical scaling 확인하기

O(N) 격차가 보이는 timing table을 출력합니다. 2026년형 Mac laptop에서 1,000 element 미만 sequence는 너무 빨라서 측정하기 어렵습니다. 100,000 길이 sequence는 깔끔한 linear scan을 보여 줍니다. 이를 12-layer LSTM equivalent와 16,384-token transformer로 scale하면, 2016년에 training wall-clock이 왜 blocker였는지 보입니다.

## 활용하기

2026년에 여전히 RNN을 선택할 때:

| 상황 | 선택 |
|-----------|------|
| 한 token씩 처리하는 streaming inference, constant memory | RNN 또는 state-space model(Mamba, RWKV) |
| Attention memory가 폭발하는 매우 긴 sequence(>1M tokens) | Linear attention, Mamba 2, Hyena |
| matmul accelerator가 없는 edge device | Depthwise-separable RNN이 여전히 FLOPs/watt에서 유리 |
| 그 외 대부분(training, batched inference, 128K까지의 context) | Transformer |

Mamba 같은 state-space model(SSM)은 structured parameterization을 가진 RNN에 가깝습니다. 그래서 `O(N)` scan memory와 selective scan을 통한 parallel training이라는 양쪽의 장점을 얻습니다. 더 나은 long-context scaling으로 transformer quality의 90%를 회복합니다. 2026년 대부분의 frontier lab은 hybrid SSM+transformer model(예: Jamba, Samba)을 train합니다. recurrence는 죽은 것이 아니라 component가 되었습니다.

## 산출물

`outputs/skill-architecture-picker.md`를 보세요. 이 skill은 length, throughput, training-budget constraint가 주어졌을 때 새 sequence problem에 맞는 architecture를 고릅니다. 1B token을 넘는 training run에 pure RNN을 추천하려면 반드시 trade-off를 명시해야 합니다.

## 연습문제

1. **쉬움.** `code/main.py`의 `rnn_style`을 가져와 scalar hidden state를 길이 64짜리 hidden state vector로 바꾸세요. 다시 측정하세요. hidden-state dimension이 커질수록 serial overhead는 얼마나 증가하나요?
2. **보통.** 순수 Python으로 parallel prefix-sum(Hillis-Steele scan)을 구현하세요. length 1024의 serial scan과 같은 numerical output을 내는지 확인하세요. depth를 세어 보세요.
3. **어려움.** Attention-style reduction을 GPU 위 PyTorch로 port하세요. sequence length를 64부터 65,536까지 sweep하면서 둘의 시간을 재세요. curve shape을 plot하고 설명하세요.

## 핵심 용어

| 용어 | 흔히 하는 말 | 실제 의미 |
|------|-----------------|-----------------------|
| Recurrence | "RNN은 sequential이다" | step `t`가 step `t-1`에 의존하여 time axis를 따라 serial execution을 강제하는 computation. |
| Serial depth | "Graph가 얼마나 깊은가" | dependent op의 가장 긴 chain. hardware가 무한해도 wall-clock의 하한을 결정합니다. |
| Attention | "Token들이 서로를 보게 한다" | position i와 j 사이 similarity score에서 나온 `a_ij`로 계산하는 weighted sum `sum_j a_ij v_j`. |
| Context window | "Model이 얼마나 많이 보는가" | attention layer가 input으로 받을 수 있는 position 수. quadratic memory cost가 여기서 증가합니다. |
| Inductive bias | "Architecture에 내장된 가정" | data가 어떤 모습일지에 대한 prior. CNN은 translation invariance를, RNN은 recency를 가정합니다. |
| State-space model | "대수가 붙은 RNN" | structured state-space matrix를 통해 parallel training이 가능하도록 parameterize된 recurrence. |
| Quadratic bottleneck | "Context가 왜 비싼가" | Attention memory = sequence length에 대해 `O(N²)`. Flash Attention은 상수를 숨기지만 scaling 자체를 없애지는 않습니다. |

## 더 읽을거리

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — mainstream NLP에서 recurrence를 밀어낸 논문.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention이 RNN 위에 붙은 형태로 탄생한 곳.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 기록을 위한 원래 LSTM 논문.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — transformer에 대한 현대적인 recurrent 답변.
