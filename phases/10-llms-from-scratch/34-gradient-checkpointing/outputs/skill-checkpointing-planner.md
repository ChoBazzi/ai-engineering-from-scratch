---
name: checkpointing-planner
description: training config와 HBM budget이 주어졌을 때 layer별 activation recomputation policy(none / selective / full / offload)를 선택합니다.
version: 1.0.0
phase: 10
lesson: 34
tags: [gradient-checkpointing, activation-recomputation, selective-checkpoint, fsdp-offload, training-memory]
---

training config(layer count L, hidden size d, sequence length S, microbatch B, dtype bytes per value, attention kernel, tensor-parallel degree TP, pipeline-parallel degree PP, MoE인 경우 expert-parallel degree EP)와 weights 및 optimizer state 이후의 per-rank HBM budget이 주어지면 다음을 출력하세요:

1. Per-layer policy. stack의 각 layer family(embedding, attention, FFN, MoE expert, norm, output head)에 대해 none, selective, full, offload 중 하나를 고르세요. S가 4_096을 넘으면 attention은 기본 selective, residual stream과 norm은 기본 none, FFN은 해당 layer의 activation에 대한 measured PCIe transfer time이 measured recompute time보다 작을 때만 offload로 기본 설정합니다.
2. Segment size k. full checkpointing이 켜져 있으면 uniform layer cost에서는 k를 round(sqrt(L))로 고르고, activation memory가 budget을 지배하면 더 작은 k를 고르세요. extra FLOP percentage를 forward FLOPs의 (1/k)로 보고하세요.
3. FlashAttention interaction. attention kernel이 이미 softmax를 recompute하는지 확인하세요. 그렇다면 selective attention checkpointing이 얻는 이득은 작습니다. none으로 낮추세요. kernel 이름(FlashAttention-2/3, xFormers memory-efficient, vanilla)을 명시하세요.
4. TP / PP plan. TP에서는 recompute 시 gather 또는 rescatter가 필요한 activation과 추가되는 per-step communication bytes를 이름 붙이세요. PP에서는 reverse microbatch가 되돌아오기 전에 activation memory를 free하도록 어떤 pipeline stage가 end-to-end checkpoint되는지 확인하세요.
5. Budget math. policy 적용 전후의 activation memory를 예측하세요(MB per rank). FLOP overhead를 fwd+bwd 대비 percent로 예측하세요. 10 percent headroom을 두고 HBM budget에 들어가지 않는 plan은 거부하세요.

selective on attention alone으로 budget을 맞출 수 있는데 every layer full checkpointing을 쓰는 것은 거부하세요. profile상 같은 memory savings에 대해 FLOP overhead가 selective보다 여러 배 높고, 정확한 비율은 workload-specific입니다. target PCIe link에서 layer의 measured activation transfer time이 measured recompute time을 초과하면 offload를 거부하세요. recompute가 이깁니다. 선택한 framework가 amax history를 snapshot하지 않는 FP8 training에서 "checkpoint everywhere"를 거부하세요. recompute가 scale을 drift시켜 gradient를 조용히 corrupt합니다.

Example input: "L=64, d=8192, S=8192, B=1, bf16, FlashAttention-3, TP=8, PP=4, HBM budget per rank 32 GB after weights, MoE with 8 experts and EP=8."

Example output:
- Layer별 policy: attention selective, FFN none, MoE expert full, embedding none, output head offload.
- Segment size: full은 MoE에만 k=8로 적용; FLOP overhead는 expert path에서 12 percent, 나머지는 0.
- FlashAttention interaction: FA-3가 이미 softmax를 recompute합니다. selective는 kernel 내부가 아니라 layer wrapper에서 적용합니다.
- TP / PP plan: recompute 시 attention input의 TP gather, step당 0.3 GB extra comms; PP stage마다 full forward를 checkpoint; PP stage 3은 final backward를 위해 activation을 유지.
- Budget math: policy 없이 activations 38 GB, policy 적용 후 11 GB. total FLOP overhead 7.5 percent fwd+bwd.
