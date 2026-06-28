# Jamba — Hybrid SSM-Transformer

> State space model(SSM)과 Transformer는 원하는 것이 다릅니다. Transformer는 attention으로 품질을 얻지만 quadratic cost를 냅니다. SSM은 recurrence로 linear-time inference와 constant memory를 얻지만 품질은 뒤처집니다. AI21의 Jamba(2024년 3월)와 Jamba 1.5(2024년 8월)는 둘을 같은 모델 안에 넣었습니다. Mamba layer 7개마다 Transformer layer 1개, 격 block마다 MoE, 그리고 단일 80GB GPU에 들어가는 256k context window입니다. Mamba-3(ICLR 2026)는 complex-valued state space와 MIMO projection으로 SSM 쪽을 조입니다. 이 lesson은 두 architecture를 end to end로 읽고, pure-SSM과 pure-Transformer long-context 시도들이 살아남지 못한 곳에서 hybrid recipe가 왜 3년 동안 scaling을 견뎠는지 설명합니다.

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 minutes

## 학습 목표

- Jamba block의 세 primitive인 Transformer layer, Mamba layer, MoE와 1:7:even interleaving recipe를 설명할 수 있습니다.
- SSM의 recurrence가 큰 틀에서 어떤 모습인지, 왜 constant-memory inference를 가능하게 하는지 말할 수 있습니다.
- 256k context에서 Jamba model의 KV cache footprint를 계산하고 pure-Transformer model이 필요로 할 비용과 비교할 수 있습니다.
- 세 가지 Mamba-3 innovation(exponential-trapezoidal discretization, complex-valued state update, MIMO)과 각각이 겨냥하는 문제를 이름 붙일 수 있습니다.

## 문제

Attention은 sequence length에 대해 quadratic입니다. State space model은 linear입니다. 그 차이는 누적됩니다. 256k token에서는 Transformer attention map이 head당 65B entries입니다. SSM의 recurrent state는 sequence length와 무관하게 fixed-size입니다.

Pure-SSM model(Mamba, Mamba-2)은 작은 scale에서 Transformer perplexity와 맞먹지만 state-tracking task에서는 뒤처지고, 일부 in-context retrieval 범주에서 실패합니다. 직관은 이렇습니다. SSM은 history를 fixed state로 압축하고, history가 길면 정보가 새어 나갑니다. Attention은 모든 것을 정확히 기억하지만 quadratic cost를 냅니다.

명백한 해법은 둘 다 쓰는 것입니다. exact recall이 중요한 곳에는 Transformer layer를 놓고, 다른 곳에는 SSM layer를 씁니다. 비율을 조정합니다. Jamba는 이 hybrid recipe를 scale에서 출시한 첫 production-grade model입니다(52B total, 12B active, 256k context, single 80GB GPU). Jamba 1.5는 family를 398B total / 94B active까지 확장합니다. Mamba-3(ICLR 2026)는 hybrid를 다시 구성할 때 기준이 되는 현재 최고의 pure-SSM baseline입니다.

이 lesson은 세 논문을 모두 읽고 "올바른 ratio를 고르는" mental model을 만듭니다.

## 개념

### 한 페이지로 보는 SSM

State space model은 fixed-size state `h`를 통해 sequence `x_1, ..., x_N`을 처리합니다:

```text
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

각 step에서 state는 linear dynamics `A`로 evolve하고, input `B x_t`를 받고, output `C h_t`를 냅니다. `A, B, C`는 학습될 수 있습니다. 핵심 속성에 주목하세요. `y_t`를 계산하려면 `h_{t-1}`과 `x_t`만 필요하고, 더 이전의 어떤 `x`도 필요하지 않습니다. memory는 constant입니다. inference는 token당 O(1)입니다.

모델링 품질의 trick은 `A`의 구조입니다. S4(Gu 2021)는 training 중 긴 convolution으로 효율적으로 평가할 수 있는 highly structured matrix를 사용했습니다. Mamba(Gu, Dao 2023)는 고정된 `A, B, C`를 data-dependent한 것으로 바꿨습니다("selective" 부분). Mamba-2(2024)는 구조를 더 단순화했습니다. Mamba-3(2026)는 특정 위치에 complexity를 다시 추가합니다.

핵심 속성: decoder LLM에서 SSM layer는 growing KV cache 대신 fixed-size per-layer state를 갖는 attention layer의 drop-in replacement입니다.

### Jamba block

Jamba block은 두 숫자에 따라 layer를 interleave합니다:

- `l`: attention-to-Mamba ratio. Jamba는 `l = 8`을 씁니다. 즉 Mamba layer 7개마다 Transformer layer 1개입니다(7 Mamba + 1 Attention = group당 8 layers).
- `e`: MoE frequency. Jamba는 `e = 2`를 씁니다. 즉 격 layer마다 MoE를 적용합니다.

block 안의 layer sequence:

```text
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

각 Jamba block은 8 layers입니다. 4 block deep(총 32 layers)이면 28 Mamba와 4 Attention layers를 얻습니다. 그중 16개가 MoE를 사용합니다.

### 1:7 비율을 쓰는 이유

AI21은 ablation을 돌렸습니다. attention-to-Mamba의 어떤 ratio가 perplexity-per-parameter와 long-context eval의 in-context recall에서 가장 좋은가?

- attention이 너무 많음(1:1): quality는 올라가지만 memory와 speed가 나빠집니다.
- attention이 너무 적음(1:15): memory는 훌륭하지만 in-context retrieval이 실패합니다.
- Sweet spot: 1:7 또는 1:8.

직관: Transformer layer는 exact recall과 state tracking을 처리합니다. Mamba layer는 처리의 저렴한 bulk를 담당합니다.

### Positional encoding

Mamba layer는 recurrence를 통해 자체적으로 position-aware합니다. 원래 Mamba 기반 hybrid의 attention layer는 RoPE를 쓰지 않았습니다. SSM layer가 position info를 제공했습니다. Jamba 1.5는 empirical long-context evaluation을 바탕으로 한 post-hoc refinement로, 더 긴 context generalization을 위해 attention layer에 RoPE를 추가합니다.

### Memory budget

Jamba-1 shape(32 layers: 28 Mamba + 4 Attention, hidden 4096, 32 attention heads)의 경우:

- KV cache(attention layer만): 256k BF16에서 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`. 4개 attention layer만 기여합니다.
- SSM state: `28 * hidden * state_size` per token prefix이지만, 이것은 sequence length에 따라 scaling하지 않는 fixed-size per layer입니다. 일반적인 Mamba state는 feature당 16, hidden 4096입니다. 총 `28 * 4096 * 16 * 2 = 3.7 MB`.

같은 hidden을 가진 32-layer pure Transformer, full MHA 32 heads와 비교하면 256k BF16에서 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`입니다. KV cache가 8x 감소합니다. 2024년 대부분 모델이 쓰는 GQA(8) baseline(`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`)과 비교해도, Jamba의 1:7 hybrid는 16 GB로 여전히 2x 작습니다.

이것이 AI21이 "single 80GB GPU에서 256k context"라고 말하는 의미입니다. full-MHA pure Transformer의 KV cache는 들어가지 않습니다. GQA baseline조차 weights와 activations를 위한 공간을 거의 남기지 않습니다. Jamba는 들어갑니다.

### Mamba-3: 2026년 pure-SSM baseline

Mamba-3(ICLR 2026, arXiv:2603.15569)는 pure-SSM 쪽에 세 innovation을 도입합니다:

1. **Exponential-trapezoidal discretization.** Mamba-2의 Euler-method discretization을 더 표현력 있는 recurrence로 대체합니다. `x_t`에 대한 외부 convolution이 아니라 core recurrence 안에서 state-input에 convolution-like operation을 적용합니다.

2. **Complex-valued state update.** 이전 Mamba들은 state matrix를 complex(S4)에서 real diagonal(Mamba), scaled identity(Mamba-2)로 줄였습니다. Mamba-3는 complex value를 다시 추가합니다. 이는 state에 대한 data-dependent rotary embedding과 동등합니다. 이전 real-valued simplification이 잃어버린 state-tracking capability를 복원합니다.

3. **Multi-input multi-output (MIMO) projections.** feature별 scalar projection 대신 matrix-valued projection을 씁니다. decode latency를 늘리지 않으면서 modeling power와 inference-time hardware utilization을 개선합니다.

1.5B parameters에서 Mamba-3는 Gated DeltaNet 대비 average downstream accuracy를 0.6 point 개선합니다. MIMO variant는 여기에 1.2를 더해 총 1.8 point gain을 만듭니다. 같은 state size에서 Mamba-3는 절반의 state로 Mamba-2와 맞먹습니다.

Mamba-3는 아직 production hybrid에서 scale로 출시되지는 않았지만, 다음 Jamba-class model의 SSM side에 들어갈 명백한 후보입니다.

### Hybrid를 사용할 때

hybrid가 이기는 경우:

- context가 pure Transformer KV cache를 고통스럽게 만들 만큼 김(64k+).
- task가 short-range structure(SSM에 좋음)와 long-range recall(Transformer 필요)을 섞음.
- Transformer KV cache만으로도 들어가지 않는 single-GPU memory budget에 배포하고 싶음.

hybrid가 지는 경우:

- context가 짧음(16k 미만). SSM overhead가 낭비되고 pure Transformer로 충분합니다.
- task가 everywhere-to-everywhere attention을 필요로 함(deep reasoning, multi-document cross-reference). hybrid에서 attention layer가 sparse한 점이 해칩니다.
- trillion-parameter frontier model로 scaling 중임. pure-Transformer + MLA + MoE(DeepSeek-V3 style)가 현재 capability race에서 이기고 있습니다.

### 경쟁 구도

| Model | Family | Scale | 고유 주장 |
|-------|--------|------|-------------|
| Mamba-2 | pure SSM | 3B | linear time, constant memory |
| Jamba | hybrid | 52B/12B | 80GB에서 256k |
| Jamba 1.5 Large | hybrid | 398B/94B | enterprise-grade long-context |
| Mamba-3 | pure SSM | 1.5B (paper) | state-tracking restored |
| DeepSeek-V3 | pure Transformer + MoE | 671B/37B | frontier capability |

2026 landscape: pure-Transformer MoE가 frontier를 지배하지만, hybrid는 256k-plus context niche를 차지합니다. Mamba-3의 state-tracking 개선은 다음 generation에서 hybrid ratio를 더 낮추는 방향(SSM은 더 많고 attention은 더 적게)으로 밀 수 있습니다.

```figure
swiglu-ffn
```

## 활용하기

`code/main.py`는 hybrid architecture용 memory calculator입니다. SSM-Transformer ratio와 hidden-size / layer-count config가 주어지면 다음을 계산합니다:

- 목표 context에서 KV cache.
- SSM state memory.
- 여러 model shape에 대해 context N에서 total memory.

calculator가 지원하는 것:

- Pure-Transformer baseline(KV cache가 N에 따라 증가).
- Jamba-style 1:7 hybrid.
- Pure-SSM(KV cache 없음).

숫자는 공개 shape에 대해서는 Jamba-1과 Jamba-1.5 paper에서 직접 가져왔고, hypothetical variant에는 extrapolation했습니다.

실제 deployment를 위한 integration considerations:

- 대부분의 production inference server(vLLM, SGLang)는 Jamba와 Mamba를 지원합니다. 특정 version을 확인하세요.
- 256k context에서는 Jamba의 memory advantage가 concurrent-request throughput에서 나타납니다. 같은 VRAM에서 Transformer sequence보다 더 많은 Jamba sequence를 넣을 수 있습니다.
- Mamba-3 standalone model은 아직 production에 출시되지 않았습니다. 1.5B research preview입니다.

## 산출물

이 lesson은 `outputs/skill-hybrid-picker.md`를 산출합니다. workload specification(context length profile, task mix, memory budget)이 주어지면 memory와 quality tradeoff에 대한 명시적 reasoning으로 pure Transformer, Jamba-style hybrid, pure SSM 중 하나를 추천합니다.

## 연습문제

1. `code/main.py`를 실행해 같은 shape의 32-layer pure Transformer(hidden 4096, 32 heads)와 Jamba-1 hybrid에 대해 256k context의 KV cache를 계산하세요. AI21 paper가 주장한 ~8x memory reduction을 확인하세요.

2. calculator를 수정해 1:3 hybrid(4 Mamba : 1 Attention)와 1:15 hybrid(14 Mamba : 1 Attention)를 모델링하세요. ratio 대비 KV cache를 plot하세요. 어떤 ratio에서 KV cache가 SSM state memory와 같아지나요?

3. Jamba paper(arXiv:2403.19887)의 Section 3을 읽으세요. AI21이 Mamba-2가 더 빠른데도 Mamba-1을 쓰는 이유를 설명하세요. 힌트: hybrid ablation section이 이를 문서화합니다.

4. Jamba 1.5 Large(398B total, 94B active)의 MoE-every-other-layer parameter overhead를 계산하세요. active ratio를 DeepSeek-V3(37B/671B)와 비교하고 Jamba architecture가 왜 active ratio를 더 높이는지 설명하세요.

5. Mamba-3 paper(arXiv:2603.15569)의 Section 3을 읽으세요. complex-valued state update가 data-dependent rotary embedding과 동등한 이유를 세 문장으로 설명하세요. 답을 Phase 7 · Lesson 04의 RoPE derivation에 연결하세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| State space model (SSM) | "Fixed state를 가진 recurrence" | 학습된 recurrence `h_t = A h_{t-1} + B x_t`를 가진 layer; token당 constant memory |
| Selective SSM | "Mamba's trick" | linear time에서 model에 gating-like selectivity를 주는 data-dependent A, B, C parameters |
| Attention-to-Mamba ratio | "Attention layer가 몇 개인가" | Jamba에서 `l = 8`은 Mamba layer 7개당 attention layer 1개를 의미 |
| Jamba block | "8-layer group" | attention 하나 + Mamba 일곱 개 + alternate position의 MoE |
| SSM state | "Hidden buffer" | Mamba layer에서 KV cache를 대체하는 fixed-size per-layer state |
| 256k context | "Jamba's flagship number" | Jamba-1이 single 80GB GPU에 넣는 sequence length; pure Transformer는 이 크기에서 불가능 |
| Mamba-3 | "2026 pure SSM" | complex state + MIMO를 갖춘 현재 최고의 pure-SSM architecture; hybrid가 다시 구성될 baseline |
| MIMO | "Multi-input multi-output" | feature별 scalar 대신 matrix-valued projection을 사용하는 Mamba-3 innovation |
| Exponential-trapezoidal discretization | "Mamba-3's recurrence" | Mamba-2의 Euler-method discretization을 포괄하는 더 표현력 있는 recurrence |
| Hybrid architecture | "Attention과 SSM 섞기" | Transformer와 SSM layer를 interleave하는 모든 model; Jamba가 production archetype |

## 더 읽을거리

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) — original Jamba paper, ratio ablations, 256k context claim
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) — scaled-up family, 398B/94B와 12B/52B public releases
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) — Jamba가 기반으로 삼은 selective SSM paper
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) — simplified structured-state-space successor
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) — complex-valued state, MIMO, 2026 pure-SSM frontier
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) — S4 paper, LLM을 위한 SSM genealogy의 시작점
