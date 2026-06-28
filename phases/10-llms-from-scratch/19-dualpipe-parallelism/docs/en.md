# DualPipe 병렬화

> DeepSeek-V3는 2,048개의 H800 GPU에서 학습되었고, MoE expert는 노드 전반에 흩어져 있었습니다. 노드 간 expert all-to-all 통신은 compute 1 GPU-hour마다 comm 1 GPU-hour를 요구했습니다. GPU는 시간의 절반 동안 놀고 있었습니다. DualPipe(DeepSeek, 2024년 12월)는 forward와 backward 계산을, 그것들이 유발하는 all-to-all comms와 겹치는 양방향 파이프라인입니다. 버블은 줄고 처리량은 올라가며, 두 개의 모델 파라미터 복사본(이름의 유래인 "dual")을 유지하는 비용은 Expert Parallelism이 이미 expert를 rank 전반에 펼치고 있다면 작습니다. 이 lesson은 DualPipe가 실제로 무엇을 하는지, 그리고 Sea AI Lab의 DualPipeV 개선판이 어떻게 약간 더 빡빡한 버블을 감수하는 대신 2x 파라미터 비용을 없애는지 살펴보는 Learn 타입 walkthrough입니다.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 minutes

## 학습 목표

- DualPipe forward-backward chunk의 네 구성요소와 각 요소가 왜 자기만의 overlap window를 갖는지 말할 수 있습니다.
- 규모가 커졌을 때의 pipeline bubble 문제와 "bubble-free"가 마케팅이 아니라 실제로 무엇을 뜻하는지 설명할 수 있습니다.
- 8개 PP rank와 16개 micro-batch에 대한 DualPipe schedule을 손으로 추적하고, forward stream과 reverse stream이 서로의 idle slot을 채우는지 확인할 수 있습니다.
- DualPipeV(Sea AI Lab, 2025)의 tradeoff를 말할 수 있습니다. Expert Parallelism이 비활성일 때 약간 더 큰 bubble을 감수하고 2x parameter replication을 제거합니다.

## 문제

2k H800 GPU에서 671B MoE 모델을 학습하면 세 병목이 함께 커집니다:

1. **메모리 압박.** 각 GPU는 모델의 조각을 들고 있습니다. 128개 head, 61개 layer, sequence 8k에서의 activation memory는 엄청납니다.
2. **Pipeline bubbles.** 전통적인 pipeline parallelism(GPipe, 1F1B)은 GPU가 자기 stage의 입력이나 gradient를 기다리는 동안 idle 상태로 둡니다. 8 stage에서는 1F1B scheduling을 써도 GPU 시간의 약 12%가 bubble일 수 있습니다.
3. **노드 간 all-to-all.** expert parallelism이 있는 MoE는 expert를 노드 전반에 흩어 놓습니다. 모든 forward pass는 토큰을 담당 expert로 보내는 all-to-all과, 결과를 다시 합치는 또 하나의 all-to-all을 유발합니다. 2k GPU에서는 이것이 쉽게 compute-to-comm 1:1 비율이 됩니다.

각 문제에는 별도 해법이 있습니다. 메모리에는 gradient checkpointing, pipeline bubble에는 Zero Bubble(Sea AI Lab, 2023), all-to-all에는 expert-parallel comm kernel이 있습니다. DualPipe가 하는 일은 이 해법들이 함께 작동하게 만드는 것입니다. 스케줄은 하나의 forward-backward chunk 안에서 compute와 comm을 겹치고, pipeline 양끝에서 동시에 micro-batch를 주입하며, 그 결과 만들어진 schedule로 compute window 안에 all-to-all을 숨깁니다.

보고된 결과: DeepSeek-V3의 14.8T-token training run에서 pipeline bubble을 거의 제거하고 GPU utilization 95% 이상을 달성했습니다.

## 개념

### Pipeline parallelism 복습

N-layer 모델을 P개 device로 나눕니다. device `i`는 layer `i * N/P .. (i+1) * N/P - 1`을 보유합니다. micro-batch는 device 0에서 P-1까지 forward로 흐른 뒤, P-1에서 0까지 backward로 돌아옵니다. 각 device는 이전 device가 출력을 보낸 뒤에야 자기 forward stage를 시작할 수 있고, downstream device가 upstream gradient를 보낸 뒤에야 backward를 시작할 수 있습니다.

GPipe(Huang et al., 2019)는 한 번에 하나의 micro-batch를 스케줄하므로 GPU 시간 대부분을 낭비합니다. 1F1B(Narayanan et al., 2021)는 여러 micro-batch의 forward와 backward pass를 interleave합니다. Zero Bubble(Qi et al., 2023)은 backward pass를 두 부분, 즉 backward-for-input(B)과 backward-for-weights(W)로 나누고 bubble을 채우도록 스케줄합니다. Zero Bubble 이후에는 pipeline이 거의 빈틈없이 tight해집니다.

DualPipe는 그 위에 두 아이디어를 더합니다:

### 아이디어 1: chunk decomposition

각 forward chunk는 네 구성요소로 나뉩니다:

- **Attention.** Q/K/V projection, attention, output projection으로 나눕니다.
- **All-to-all dispatch.** 토큰을 expert로 보내는 노드 간 통신.
- **MLP.** MoE expert 계산.
- **All-to-all combine.** expert 출력을 되가져오는 노드 간 통신.

backward chunk에는 이 각각의 gradient 버전이 추가됩니다. DualPipe는 all-to-all dispatch가 다음 chunk의 attention compute와 병렬로 일어나고, all-to-all combine이 이어지는 chunk의 MLP compute와 병렬로 일어나도록 스케줄합니다.

### 아이디어 2: bidirectional scheduling

대부분의 pipeline schedule은 stage 0에서 micro-batch를 주입하고 stage P-1 방향으로 흐르게 합니다. DualPipe는 양끝 모두에서 micro-batch를 주입합니다. stage 0은 그곳에서 시작한 forward micro-batch를 보고, stage P-1도 그곳에서 시작한 forward micro-batch를 봅니다. 두 stream은 가운데에서 만납니다.

이것이 작동하려면 device `i`가 early-pipeline layer `i`와 late-pipeline layer `P - 1 - i`를 모두 가지고 있어야 합니다. 이것이 DualPipe의 "dual" 부분입니다. 각 device는 양방향 각각에 서비스를 제공하는 데 필요한 모델 layer의 두 복사본을 유지합니다. DeepSeek-V3 규모에서는 이것이 2x parameter replication 비용입니다. Expert Parallelism이 이미 MoE expert를 매우 얇게 분산하고 있어 non-expert layer를 두 번 복제하는 비용은 상대적으로 작기 때문에 감당할 수 있습니다.

핵심은 한 방향의 forward stream과 다른 방향의 backward stream이 단방향 schedule에서 bubble이 생길 정확한 위치에서 겹친다는 점입니다. bubble이 사라집니다.

### 손으로 추적한 schedule

P = 4 ranks, 8 micro-batches를 생각해 봅시다. 4개는 forward, 4개는 reverse로 나뉩니다. 시간은 왼쪽에서 오른쪽으로 흐르고, 행은 device rank입니다.

```text
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

"F4/F5R" 표기를 읽으면, rank 1은 같은 time slot에서 micro-batch 4의 forward(파이프라인에서 왼쪽에서 오른쪽으로 이동)와 micro-batch 5의 forward(오른쪽에서 왼쪽으로 이동)를 모두 실행하고 있습니다. 이것이 운영 관점에서 "bidirectional"의 의미입니다.

rank 2에서는 cross stream이 더 빨리 겹치고, rank 0과 P-1에서는 가장 늦게 겹칩니다. schedule의 안정적인 중간 구간에서는 모든 rank가 한 방향의 forward와 다른 방향의 backward를 겹쳐 실행합니다. compute는 바쁩니다. forward pass의 all-to-all dispatch는 backward compute 안에 숨습니다. all-to-all combine은 forward compute 안에 숨습니다. bubble이 밀려납니다.

### Bubble accounting

표준 1F1B pipeline bubble(rank당 낭비 시간):

```text
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 개선은 이를 줄이지만 0으로 만들지는 않습니다. DualPipe는 안정 구간에서 micro-batch 수가 pipeline depth의 2배로 나누어떨어지면 bubble이 0입니다. 안정 구간 밖(warmup과 cooldown)에는 일부 bubble이 있지만 micro-batch 수와 함께 증가하지 않습니다. 논문이 강조하는 핵심 속성입니다.

마케팅 용어로는 "bubble-free"입니다. 기술 용어로는 bubble이 micro-batch 수와 함께 증가하지 않는다는 뜻입니다. Sea AI Lab의 후속 분석(DualPipeV / Cut-in-half)은 Expert Parallelism이 병목이 아닐 때만 완전한 zero-bubble이 가능함을 보입니다. EP 기반 all-to-all이 있으면 어떤 scheduling compromise는 항상 존재합니다.

### DualPipeV refinement

Sea AI Lab(2025)은 EP comm overlap이 핵심이 아닐 때 2x parameter replication이 낭비라고 관찰했습니다. DualPipeV schedule은 bidirectional injection을 하나의 parameter copy에서 실행되는 "V-shape" schedule로 접습니다. bubble은 DualPipe보다 약간 크지만 메모리 절감은 큽니다. DeepSeek은 open-source DualPipe 구현에서 EP-off mode로 DualPipeV를 채택했습니다.

tradeoff:

| Feature | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Device당 param copy | 2 | 1 | 1 | 1 |
| Micro-batch 대비 bubble | constant | small growth | grows | grows |
| Compute-comm overlap | full | partial | minimal | partial |
| 사용 시점 | EP-heavy MoE | dense 또는 EP-light | baseline | any pipeline |

### 14.8T-token run에서의 의미

DeepSeek-V3의 pre-training은 2,048개 H800 GPU에서 약 2.8M GPU-hour 동안 14.8T token을 소비했습니다. naive 1F1B를 썼다면 pipeline bubble로 12-15%를 잃었을 것입니다. 이는 340-420K GPU-hour로, 70B 모델 하나를 완전히 학습하기에 충분합니다. DualPipe는 그 대부분을 회수했습니다. 내부 로그 없이는 기여분을 직접 정량화하기 어렵지만, 논문은 학습 전체 평균 GPU utilization이 95% 이상이라고 주장합니다.

작은 실행(1k GPU 미만)에서는 DualPipe가 과합니다. pipeline bubble은 전체 비용 대비 작고, dense-model training은 all-to-all bottleneck에 거의 걸리지 않습니다. 수천 GPU 규모의 frontier MoE 학습에서는 사실상 필수입니다.

### Stack에서의 위치

- **FSDP**(Phase 10 · 05)와 상호보완적입니다. FSDP는 model parameter를 rank 전반에 shard하고, DualPipe는 rank 전반의 compute를 schedule합니다. 둘은 결합됩니다.
- **ZeRO-3** gradient sharding과 호환됩니다. 두 copy replication의 bookkeeping은 ZeRO의 sharded gradient와 협력해야 합니다.
- 특정 cluster topology에 맞춰 튜닝된 **custom all-to-all kernels**가 필요합니다. DeepSeek의 open-source kernel이 reference implementation입니다.

```figure
expert-capacity
```

## 활용하기

`code/main.py`는 pipeline schedule simulator입니다. `(P, n_micro_batches, schedule)`을 받아 1F1B, Zero Bubble, DualPipe, DualPipeV 각각의 안정 구간 utilization을 출력합니다. 이것은 teaching tool입니다. 숫자는 논문의 정성적 주장과 맞지만, production measured speedup에 대한 주장은 아닙니다.

시뮬레이터의 가치는 P와 micro-batch 수를 바꾸어 실행하면서 1F1B에서는 bubble fraction이 커지지만 DualPipe에서는 그렇지 않음을 보는 데 있습니다.

실제 training run을 위한 integration considerations:

- micro-batch count를 깔끔하게 나눌 수 있는 pipeline-parallel depth를 고르세요.
- expert-parallel mesh가 bidirectional all-to-all을 지원하는지 확인하세요. DeepSeek의 kernel이 reference입니다.
- 처음에는 schedule 자체를 디버깅하는 데 일주일쯤 쓸 각오를 하세요. bookkeeping이 까다롭습니다.
- aggregate만 보지 말고 rank별 GPU utilization을 모니터링하세요. DualPipe의 이점은 straggler를 조이는 데서 옵니다.

## 산출물

이 lesson은 `outputs/skill-dualpipe-planner.md`를 산출합니다. 학습 클러스터 명세(GPU 수, 토폴로지, interconnect, model shape)가 주어지면 pipeline parallelism 전략, 사용할 scheduling algorithm, 목표 규모에서의 예상 bubble fraction을 추천합니다.

## 연습문제

1. `(P=8, micro_batches=16, schedule=dualpipe)`와 `(P=8, micro_batches=16, schedule=1f1b)`로 `code/main.py`를 실행하세요. GPU utilization 차이를 계산하고, 이를 학습 100만 토큰당 회수된 GPU-hour로 표현하세요.

2. `(P=4, micro_batches=8, schedule=dualpipe)`의 schedule table을 손으로 그리세요. 각 time slot에 micro-batch ID와 방향을 표시하세요. bubble이 사라지는 첫 time slot을 찾으세요.

3. DeepSeek-V3 technical report(arXiv:2412.19437)의 Figure 5를 읽으세요. DualPipe forward chunk 안에서 all-to-all dispatch의 overlap window를 찾으세요. compute schedule이 이를 어떻게 숨기는지 설명하세요.

4. P=8 pipeline stage인 70B dense model과 P=16 pipeline stage인 671B MoE model에서 DualPipe의 2x parameter overhead를 계산하세요. MoE case의 overhead가 비율상 더 작은 이유를 보이세요(대부분의 parameter가 expert이고, 큰 EP group 전반에 sharded되어 있음).

5. DualPipe를 Chimera(2021년의 경쟁 bidirectional scheduler)와 비교하세요. 논문의 Section 3.4를 reference로 삼아 Chimera에는 없고 DualPipe가 추가한 두 가지 구체적 속성을 찾으세요.

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| Pipeline bubble | "Rank당 idle time" | pipeline stage가 입력이나 gradient를 기다리느라 낭비되는 GPU cycle |
| 1F1B | "Default pipeline schedule" | one forward / one backward interleaved scheduling; DualPipe가 이기는 baseline |
| Zero Bubble | "Sea AI Lab 2023" | backward를 B(input gradient)와 W(weight gradient)로 나눔; pipeline을 거의 완전히 tight하게 만듦 |
| DualPipe | "DeepSeek-V3 schedule" | bidirectional pipeline + compute-comm overlap; bubble이 micro-batch 수와 함께 증가하지 않음 |
| DualPipeV | "Cut-in-half" | 약간 더 큰 bubble을 감수하고 2x parameter replication을 없애는 V-shape 개선판 |
| Chunk | "Pipeline work unit" | 하나의 micro-batch가 하나의 pipeline stage를 통과하는 forward 또는 backward pass |
| All-to-all dispatch | "토큰을 expert로 보냄" | 토큰을 할당된 MoE expert로 route하는 노드 간 comm |
| All-to-all combine | "Expert output을 되가져옴" | MLP 이후 expert output을 gather하는 노드 간 comm |
| Expert Parallelism (EP) | "GPU 전반의 experts" | 서로 다른 GPU가 서로 다른 expert를 보유하도록 MoE expert를 rank 전반에 shard함 |
| Pipeline Parallelism (PP) | "GPU 전반의 layers" | model layer를 rank 전반에 shard함; DualPipe가 schedule하는 차원 |
| Bubble fraction | "낭비된 GPU 시간" | (bubble_time / total_time); DualPipe가 0에 가깝게 밀어내는 비율 |

## 더 읽을거리

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) — primary DualPipe reference
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) — DualPipeV(Cut-in-half) mode를 포함한 open-source reference implementation
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — Zero Bubble predecessor
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) — DeepSeek의 EP-off mode에 영향을 준 DualPipeV analysis
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — DualPipe가 비교하는 1F1B schedule
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — original pipeline parallelism paper와 bubble problem
