---
name: sequence-architecture-picker
description: 길이, 처리량, training budget이 주어졌을 때 sequence architecture(RNN, transformer, SSM, hybrid)를 고릅니다.
version: 1.0.0
phase: 7
lesson: 1
tags: [transformers, architecture, rnn, ssm]
---

Sequence problem(max length, batch shape, budgeted training tokens, inference latency target, device class)이 주어지면 다음을 출력하세요.

1. 주요 architecture. transformer, state-space model(Mamba/RWKV), hybrid SSM+attention, RNN 중 하나. dominant constraint와 연결된 한 문장 이유를 덧붙입니다.
2. context 길이 전략. Transformer라면 full attention cutoff, sliding window size, RoPE scaling factor를 제시합니다. SSM이라면 scan chunk size를 제시합니다. RNN이라면 hidden width를 제시합니다.
3. training FLOP profile. architecture와 context에서 token당 FLOPs를 근사하고, spec이 compute budget에 맞는지 적습니다.
4. inference memory profile. Transformer는 KV cache, SSM은 state size, RNN은 per-token memory를 산정합니다. target device가 batch size 1을 담을 수 없으면 표시합니다.
5. risk note. 이 선택이 해당 spec scale에서 겪는 것으로 알려진 구체적인 failure mode 하나를 적습니다. 예: Flash Attention 없이 24GB GPU에서 64K context transformer OOM.

1B token을 넘는 training run에는 gradient-flow와 parallelism penalty를 명시하지 않는 한 pure RNN을 추천하지 마세요. `O(N^2)` memory cost를 명시하지 않는 한 >64K context에 full-attention transformer를 추천하지 마세요. 이름 있는 fallback 없이 published <12 months ago인 brand-new architecture를 production 용도로 추천하지 마세요.
