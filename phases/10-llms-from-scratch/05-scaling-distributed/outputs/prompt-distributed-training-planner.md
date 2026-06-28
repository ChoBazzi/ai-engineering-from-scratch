---
name: prompt-distributed-training-planner
description: 모델 크기와 가용 hardware를 바탕으로 distributed training run 계획
version: 1.0.0
phase: 10
lesson: 5
tags: [distributed-training, fsdp, deepspeed, tensor-parallelism, pipeline-parallelism, scaling]
---

# Distributed Training Planner

대규모 언어 모델의 distributed training run을 계획할 때 이 프레임워크로 parallelism strategy, memory budget, communication overhead, expected throughput을 결정하세요.

## 입력 요구사항

다음을 제공하세요.
- **Model size** (십억 단위 parameter)
- **Target training tokens** (조 단위)
- **Available GPUs** (type: A100/H100/H200, count, interconnect: NVLink/InfiniBand)
- **GPU memory** (A100/H100은 80GB, H200은 141GB)
- **Nodes** (node당 GPU 수, node 수)
- **Budget constraints** (최대 비용 달러, 최대 wall-clock time)

## 1단계: Memory Budget

각 component의 per-GPU memory를 계산하세요.

| Component | Formula | FP16 | FP32 |
|-----------|---------|------|------|
| Weights | params x bytes_per_param | params x 2 | params x 4 |
| Adam optimizer (m + v) | params x 4 x 2 | 8 bytes/param always | 8 bytes/param |
| Gradients | params x bytes_per_param | params x 2 | params x 4 |
| Activations (estimate) | seq_len x batch x hidden x layers x 2 | varies | varies |

총합이 GPU memory를 초과하면 sharding이 필요합니다. 다음 순서로 시도하세요.
1. ZeRO-1 (optimizer만 shard) -- communication이 가장 저렴함
2. ZeRO-2 (+ gradients) -- 중간 수준 communication
3. FSDP/ZeRO-3 (+ weights) -- communication은 가장 크지만 memory 절감 최대
4. activation이 여전히 너무 크면 activation checkpointing 추가
5. 단일 layer가 하나의 GPU에 맞지 않으면 tensor parallelism 추가

## 2단계: Parallelism Strategy

### 의사결정 트리

1. **하나의 layer가 하나의 GPU에 들어가나요?**
   - 아니요: tensor parallelism이 필요합니다. TP = 2, 4, 또는 8(node 내부)로 설정하세요.
   - 예: tensor parallelism을 건너뜁니다.

2. **전체 모델(sharding 포함)이 하나의 node 안 GPU들에 들어가나요?**
   - 아니요: pipeline parallelism이 필요합니다. PP = node 수 / group으로 설정하세요.
   - 예: pipeline parallelism을 건너뜁니다.

3. **data parallelism에 남는 GPU는 몇 개인가요?**
   - DP = total_gpus / (TP x PP)

4. **data parallel group 안에서 어떤 sharding level을 사용할까요?**
   - FSDP(ZeRO-3)에서 시작하세요. communication이 병목이면 ZeRO-2 또는 ZeRO-1로 낮추세요.

### 일반적인 구성

| Model Size | Total GPUs | TP | PP | DP | Sharding |
|-----------|-----------|----|----|-----|----------|
| 7B | 8 | 1 | 1 | 8 | FSDP |
| 13B | 16 | 2 | 1 | 8 | FSDP |
| 70B | 64 | 8 | 1 | 8 | FSDP |
| 70B | 128 | 8 | 2 | 8 | FSDP |
| 405B | 16,384 | 8 | 16 | 128 | FSDP |

## Step 3: Communication 분석

training step당 communication volume을 추정하세요.

- **Data parallel (all-reduce)**: 2 x gradient_size x (N-1)/N per step
- **FSDP (all-gather + reduce-scatter)**: ~3 x weight_size x (N-1)/N per step (higher than DP)
- **Tensor parallel (all-reduce per layer)**: 2 x activation_size x num_layers per step (needs NVLink)
- **Pipeline parallel (point-to-point)**: activation_size per stage boundary (minimal)

communication 시간이 compute time의 20%를 넘으면 strategy가 communication-bound입니다. 해결책:
- gradient accumulation(all-reduce 빈도 감소)
- communication과 computation overlap(FSDP는 기본적으로 수행)
- micro-batch size 증가(더 나은 compute-to-communication ratio)
- communication 부담이 더 작은 sharding stage로 전환

## Step 4: Throughput과 비용 추정

**FLOPS per training step:**
- Forward: ~2 x params x tokens_per_batch
- Backward: ~4 x params x tokens_per_batch (2x forward)
- Total: ~6 x params x tokens_per_batch

**Training time:**
- total_flops = 6 x params x total_tokens
- time_seconds = total_flops / (num_gpus x gpu_tflops x 1e12 x utilization)
- 일반적인 utilization: 35-45% (communication, pipeline bubble, memory overhead 고려)

**Cost:**
- total_gpu_hours = num_gpus x time_seconds / 3600
- cost = total_gpu_hours x cost_per_gpu_hour

## Step 5: 검증 체크리스트

실행 전에:

1. per-GPU memory가 hardware limit 안에 들어갑니다(10% headroom 포함)
2. effective batch size가 목표와 일치합니다(per_gpu_batch x DP x gradient_accumulation_steps)
3. communication-to-compute ratio가 20% 미만입니다
4. pipeline bubble fraction이 15% 미만입니다(충분한 micro-batch)
5. learning rate가 effective batch size에 맞게 scaling되었습니다
6. checkpointing frequency가 failure probability를 고려합니다(대규모 run은 1-2시간마다 저장)
7. gradient clipping이 설정되었습니다(대형 모델은 보통 1.0)
8. warmup step이 total step에 비례합니다(보통 전체의 0.1-1%)

## 위험 신호

- **TP > 8**: node 간 tensor parallelism(InfiniBand 경유)은 거의 항상 pipeline parallelism보다 느립니다
- **Pipeline stages > 32**: micro-batch가 많아도 bubble overhead가 커집니다
- **Effective batch size > 10M tokens**: 수익 체감이 발생하며 convergence를 해칠 수 있습니다
- **Utilization below 30%**: communication-bound입니다. parallelism strategy를 재평가하세요
- **13B 초과에서 activation checkpointing 없음**: backward pass 중 메모리가 부족해집니다
- **작은 per-GPU batch에서 gradient accumulation 없음**: gradient noise가 증가합니다. 256+ sample의 effective batch로 accumulate하세요
