# Async와 Hogwild! Inference

> Speculative decoding(Phase 10 · 15)은 한 sequence 안의 token을 병렬화합니다. Multi-agent framework는 전체 sequence 사이를 병렬화하지만 명시적 coordination(voting, sub-task splitting)을 강제합니다. Hogwild! Inference(Rodionov et al., arXiv:2504.06261)는 다른 일을 합니다. 같은 LLM의 N개 instance를 하나의 공유 key-value cache를 상대로 병렬 실행합니다. 각 worker는 다른 모든 worker가 생성한 token을 즉시 봅니다. QwQ, DeepSeek-R1 같은 modern reasoning model은 fine-tuning 없이 그 shared cache를 통해 self-coordinate할 수 있습니다. 이 접근법은 실험적이지만, spec decode와 직교하는 inference parallelism의 완전히 새로운 축을 엽니다. 이 lesson은 stdlib Python으로 two-worker Hogwild! simulator를 구현하고, shared-cache collaboration이 기존 모델의 reasoning ability에서 왜 emergent하게 나타나는지 설명합니다.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## 학습 목표

- 세 가지 일반적인 parallel-LLM topology(voting, sub-task, Hogwild!)를 설명하고, 각각이 겨냥하는 문제를 말할 수 있습니다.
- Hogwild!의 핵심 setup을 말할 수 있습니다. multiple workers, one shared KV cache, self-prompting을 통한 emergent coordination.
- worker count `N`, task-level parallelism `p`, coordination overhead `c`의 함수로 Hogwild!의 wall-time speedup을 계산할 수 있습니다.
- toy problem에서 two-worker Hogwild! simulator를 구현하고 emergent task division을 관찰할 수 있습니다.

## 문제

현대 LLM은 긴 reasoning chain을 생성해 어려운 문제를 풉니다. 5000 token의 step-by-step logic은 흔하고, 깊은 수학 문제에서는 수만 token도 발생합니다. 70B model에서 decode가 35 tokens/sec라면 50k token은 24분입니다. 대화형이라고 보기 어렵습니다.

Speculative decoding(Phase 10 · 15)은 한 sequence 안을 병렬화해 3-5x speedup을 줍니다. 그 이상에서는 autoregressive decoding의 sequential dependency가 단단한 상한입니다. 각 새 token은 모든 이전 token에 의존합니다.

명백한 질문: sequence 사이를 병렬화할 수 있을까요? 같은 문제에 같은 모델의 여러 copy를 돌리고, 협력하게 하고, 작업을 나누게 할 수 있을까요?

기존 작업: voting ensemble(N개 모델을 실행해 majority answer 선택), tree-of-thought(reasoning path를 branch하고 recombine), multi-agent framework(각 agent에 sub-task를 할당하고 coordinator 사용). 이것들은 특정 task domain에서 도움이 됩니다. 하지만 모두 명시적 coordination machinery를 도입합니다. voting rules, branch-and-prune logic, agent-to-agent messaging protocol입니다.

Hogwild! Inference는 다른 접근을 취합니다. N worker가 하나의 KV cache를 공유합니다. 각 worker는 다른 모든 worker의 generated token을 즉시 봅니다. 마치 자기 context인 것처럼 말입니다. worker들은 training이나 fine-tuning 없이 작업을 나누는 법을 알아냅니다. Reasoning model(QwQ, DeepSeek-R1, Claude-family reasoning mode)은 shared cache를 읽고 "worker 2가 이미 base case를 처리했으니 나는 inductive step을 하겠다" 같은 말을 할 수 있습니다.

speedup은 workload-dependent이며 2026년 4월 기준 실험적입니다. 하지만 inference parallelism의 새 축을 열기 때문에 알 가치가 있습니다.

## 개념

### Setup

N worker process를 초기화하고 모두 같은 LLM을 실행합니다. worker별 KV cache 대신 하나의 shared cache를 유지합니다. worker `i`가 token `t_j`를 생성하면, 그 token은 shared cache의 다음 position에 기록됩니다. worker `k`가 다음 step을 수행할 때는 현재 cache state를 읽습니다. 여기에는 모든 N worker가 지금까지 생성한 모든 것이 포함됩니다.

step time에서 worker들은 token write를 두고 race합니다. worker별 position index는 없습니다. cache는 하나의 growing sequence입니다. 순서는 write arrival time이 결정합니다.

### Coordination이 나타나는 이유

worker들은 prompt를 공유합니다. 보통 "You are one of N instances working together on this problem. Each instance reads the shared memory and can see what other instances have written. Avoid redundant work." 같은 형태입니다. prompt와 shared cache만으로 충분합니다. Reasoning model은 cache를 읽고, 문제의 어느 부분이 이미 시도되었는지 알아차리며, 종종 미탐색 부분으로 pivot합니다.

Hogwild! paper(Rodionov et al., 2025)는 다음과 같은 관찰을 보고합니다:

- worker들이 plan을 만들고 cache를 통해 다른 worker에게 전달합니다.
- worker들이 다른 worker의 reasoning error를 알아차리고 지적합니다.
- plan이 실패하면 worker들이 적응하고 alternative를 제안합니다.
- redundancy를 확인하라고 prompt하면 worker들이 중복을 감지하고 pivot합니다.

이 중 어느 것도 fine-tuning을 요구하지 않습니다. emergent behavior는 모델이 이미 가진 reasoning capability에서 나옵니다.

### 이름의 유래

논문의 이름은 asynchronous-update optimizer인 Hogwild! SGD(Recht et al., 2011)를 비튼 것입니다. analogy는 이렇습니다. SGD의 asynchronous worker들은 모두 shared parameter vector에 write합니다. Hogwild! Inference의 worker들은 모두 shared KV cache에 write합니다. 둘 다 synchronization guarantee보다 empirical convergence에 의존합니다.

### RoPE가 이를 다룰 수 있게 하는 이유

Rotary Position Embeddings(RoPE, Su et al. 2021)는 Q와 K vector의 rotation으로 position information을 encode합니다. position이 rotation이고 baked-in offset이 아니므로 token의 position이 이동해도 KV cache entry를 재계산하지 않아도 됩니다. worker `i`가 shared cache의 position `p`에 write하면, 그 position을 읽는 다른 worker는 cached entry를 직접 사용할 수 있습니다. re-rotation이 필요 없습니다.

learned-position 또는 absolute-position model에서는 Hogwild!가 concurrent write마다 cache invalidation을 필요로 했을 것입니다. RoPE는 cache를 안정적으로 유지하게 해줍니다.

### Wall-time 계산

`T_serial`을 한 worker가 혼자 문제를 푸는 시간이라고 합시다. `p`는 task-level parallelizable fraction입니다. `c`는 step당 coordination overhead(extended cache를 읽고 무엇을 쓸지 결정하는 비용)입니다.

Single-worker time: `T_serial`.
coordination이 공짜일 때 N-worker Hogwild! time: `T_serial * ((1 - p) + p / N)`. 고전적인 Amdahl입니다.
coordination overhead 포함: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

worker가 생산적이려면 `c`가 step당 decode time에 비해 작아야 합니다. 5k+ token을 생성하는 reasoning model에서는 worker들이 수백 token의 coordination overhead를 감당하고도 앞설 수 있습니다. short chat task에서는 coordination이 지배하고 Hogwild!가 serial보다 나쁩니다.

### 구체적 예시

Reasoning problem: chain-of-thought 10k tokens. 문제에 `p = 0.7` parallelizable content(서로 다른 proof strategy, case analysis)가 있고, worker당 `c = 200` token의 coordination overhead가 있다고 합시다. `N = 4` workers:

- Serial time: 10000 decode steps.
- Hogwild! time: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 decode steps.
- Speedup: 10000 / 5550 = 1.8x.

이는 완만합니다. 하지만 더 긴 reasoning problem(50k tokens)에서는 coordination overhead가 amortize되고 speedup이 2.5-3x로 밀립니다. Hogwild!는 multi-threaded code를 자연스럽게 쓸 수 있게 해주는 language의 thread-level parallelism에 해당하는 inference 기법입니다.

### Hogwild!를 사용할 때

- task가 독립 sub-goal로 parallelize될 수 있는 긴 reasoning problem(수천 token).
- step by step으로 생각하도록 학습된 reasoning model. non-reasoning model은 self-coordinate을 잘하지 못합니다.
- shared cache와 N worker process를 담을 충분한 VRAM이 있는 single-node deployment. cache는 공유되지만 각 worker는 자기 activation memory를 갖습니다.

### 사용하지 않을 때

- 짧은 interactive chat. coordination overhead가 지배합니다.
- parallelize되지 않는 task(single linear proof, single compilation). N=1이 최대입니다.
- non-reasoning model. coordination이 emergent하지 않습니다.
- multi-node deployment. shared cache는 매우 빠른 cross-worker synchronization을 필요로 합니다. intra-node는 괜찮지만 cross-node는 latency disaster입니다.

### 실험적 상태

2026년 4월 기준 Hogwild!는 open-source PyTorch implementation이 있는 research method입니다. production adoption은 아직 없습니다. 세 blocker가 있습니다:

1. concurrent process 전반에서 shared KV cache를 관리하는 것은 non-trivial engineering입니다.
2. Emergent coordination은 task-dependent입니다. benchmark는 아직 구축 중입니다.
3. speedup은 speculative decoding이 이미 제공하는 것과 비교하면 modest하고, 둘은 결합 가능하지만 combined engineering은 또 다른 layer입니다.

알 가치가 있습니다. 실험할 가치가 있습니다. 아직 product에 베팅할 가치는 없습니다.

```figure
continuous-batching
```

## 직접 만들기

`code/main.py`는 toy Hogwild! simulator를 구현합니다:

- 두 worker process. 각각은 알려진 확률로 여러 token category(work-token, observe-token, coordinate-token) 중 하나를 생성하는 deterministic "LLM"입니다.
- 두 worker가 모두 읽고 쓰는 shared cache(그냥 token list).
- 단순 coordination logic: worker가 다른 worker가 이미 어떤 category에서 충분한 work token을 생성한 것을 보면 다른 category를 고릅니다.

simulator는 fixed step budget으로 실행되고 다음을 보고합니다:

- 생성된 total work-tokens.
- Total wall time(worker step 수).
- single worker 대비 effective speedup.
- 어떤 worker가 어떤 token을 썼는지 trace.

### 1단계: shared cache

두 worker가 append하는 list입니다. 실제 구현에서는 간단한 locking(Python `threading.Lock`)을 쓰고, 여기서는 counter로 simulate합니다.

### 2단계: worker loop

각 worker는 각 step에서:

- 현재 shared cache를 읽습니다.
- 이미 있는 내용에 따라 쓸 token category를 결정합니다.
- token 하나를 씁니다.

### 3단계: coordination heuristic

category X가 이미 cache에 K token을 가지고 있고 worker의 intended category가 X라면, worker는 category Y로 바꿉니다. 이것은 reasoning model behavior, 즉 "이 부분은 이미 다뤄졌으니 다른 것을 하자"를 toy로 대체한 것입니다.

### 4단계: 측정된 speedup

N=1 worker와 N=2 workers로 같은 total step budget에서 simulator를 실행하세요. work-token 수를 세세요. N=2는 coordination-driven task division 덕분에 대략 1.5-1.8x 더 많은 work-token을 생성해야 합니다.

### 5단계: coordination stress test

coordination heuristic의 sensitivity를 줄이세요. 다시 실행하세요. 좋은 coordination이 없으면 N=2가 같은 token을 중복 생성하고 speedup이 1 아래로 떨어지는 것을 관찰하세요. 이는 논문의 관찰과 맞습니다. worker들이 self-coordinate할 reasoning capacity를 가질 때만 trick이 작동합니다.

## 활용하기

2026년 4월 기준 production에서 Hogwild! integration은 research-grade입니다. Yandex/HSE/IST의 reference implementation은 PyTorch 기반이며 DeepSeek-R1과 QwQ model의 single-node multi-process setup을 겨냥합니다.

실용적 adoption path:

1. reasoning-task workload를 profile하세요. exploratory token(여러 strategy, case analysis, search)과 linear token의 비율을 측정하세요.
2. exploration이 지배하면 two-worker Hogwild! experiment를 실행하세요. wall-time improvement를 측정하세요.
3. improvement가 1.3x 미만이면 coordination-dominated regime입니다. single-worker로 되돌리세요.
4. improvement가 1.5x 이상이면 N=4로 늘려 다시 측정하세요. diminishing returns는 보통 N=4-8 근처에서 옵니다.

speculative decoding과 결합: 각 Hogwild! worker는 독립적으로 spec decode를 사용할 수 있습니다. 두 speedup은 대략 곱해져, 3x spec decode와 1.8x Hogwild!가 naive single-worker decoding 대비 effective 5.4x를 만듭니다.

## 산출물

이 lesson은 `outputs/skill-parallel-inference-router.md`를 산출합니다. reasoning workload profile(token budget, task parallelism profile, model family, deployment target)이 주어지면 voting, tree-of-thought, multi-agent, Hogwild!, speculative decoding strategy 사이에서 route합니다.

## 연습문제

1. default setting으로 `code/main.py`를 실행하세요. N=2 Hogwild! configuration이 같은 wall time에서 N=1 baseline보다 더 많은 work-token을 생성하는지 확인하세요.

2. coordination heuristic의 strength를 줄이세요(`coordination_weight=0.1` 설정). 다시 실행하세요. speedup이 무너지는 것을 보이세요. 이유를 설명하세요. worker들이 coordinate하지 못하면 effort를 duplicate합니다.

3. `p=0.8, c=500`, N=4 workers인 50k-token reasoning task의 예상 Hogwild! speedup을 계산하세요. `p=0.3, c=200`, N=4인 1k-token chat task도 똑같이 계산하세요. 왜 하나는 이득이고 다른 하나는 손실인가요?

4. Hogwild! paper의 Section 4(preliminary evaluation)를 읽으세요. 저자들이 보고한 두 failure mode를 찾으세요. 더 나은 coordination prompt가 각각을 어떻게 완화할 수 있을지 설명하세요.

5. toy에서 Hogwild!와 speculative decoding을 결합하세요. 각 worker가 내부적으로 2-token spec-decode를 사용합니다. multiplicative speedup을 보고하세요. 두 worker가 모두 같은 shared-cache prefix를 extend하고 싶어 할 때 어떤 bookkeeping problem이 발생하나요?

## 핵심 용어

| 용어 | 사람들이 흔히 말하는 뜻 | 실제 의미 |
|------|----------------|------------------------|
| Hogwild! | "Parallel workers, shared cache" | 하나의 shared KV cache를 가지고 동시에 실행되는 같은 LLM의 N instances; self-prompting을 통한 emergent coordination |
| Shared KV cache | "Coordination medium" | 모든 worker가 읽고 쓰는 하나의 growing KV buffer; worker 간 instant token visibility를 가능하게 함 |
| Emergent coordination | "No training needed" | reasoning-capable LLM은 fine-tuning이나 explicit protocol 없이 shared cache를 읽고 작업을 나눌 수 있음 |
| Coordination overhead (c) | "Orienting에 쓰는 tokens" | extended cache를 읽고 무엇을 할지 결정하는 worker당 비용; total decode time 대비 작아야 함 |
| Parallelizable fraction (p) | "병렬로 실행 가능한 것" | task-level parallelism: total work 중 본질적으로 sequential하지 않은 부분의 비율 |
| RoPE enables Hogwild! | "Rotary positions are shift-invariant" | position이 rotation이므로 shared cache에 write해도 prior token을 재계산할 필요가 없음 |
| Voting ensemble | "N개 실행, majority 선택" | 가장 단순한 parallel inference topology; classification에는 유용하지만 long-form reasoning에는 덜 적합 |
| Tree of thought | "Branch and prune" | 여러 branch를 탐색하고 prune하는 reasoning strategy; explicit coordination logic |
| Multi-agent framework | "Sub-task 할당" | 각 agent가 role을 받고 coordinator가 orchestrate함; protocol overhead가 큼 |

## 더 읽을거리

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — Hogwild! paper, QwQ와 DeepSeek-R1 preliminary evaluation
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — original Hogwild!, naming origin
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, shared-cache inference를 tractable하게 만드는 속성
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — Hogwild!와 직교하는 tree-of-thought reasoning strategy
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding, Hogwild!와 compose되는 within-sequence parallelism
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) — paper experiment의 single source of truth
