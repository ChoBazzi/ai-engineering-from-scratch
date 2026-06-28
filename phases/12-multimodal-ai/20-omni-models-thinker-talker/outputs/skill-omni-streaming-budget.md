---
name: omni-streaming-budget
description: target TTFAB와 feature set에 맞춰 Thinker-Talker streaming voice pipeline(Qwen-Omni / Moshi / Mini-Omni)의 크기를 정한다.
version: 1.0.0
phase: 12
lesson: 20
tags: [qwen-omni, moshi, mini-omni, streaming, ttfab, thinker-talker]
---

voice-first product spec(target TTFAB, mic sample rate, vision in yes/no, bilingual, full-duplex)과 compute constraint(GPU class, budget)가 주어지면 Thinker-Talker pipeline의 크기를 정한다.

생성할 것:

1. Model family pick. Moshi(best latency), Qwen2.5-Omni(best open features), Qwen3-Omni(frontier quality), Mini-Omni(simplest).
2. Thinker and Talker sizes. <400ms TTFAB에는 7B Thinker + 200-300M Talker. quality에는 70B+ Thinker를 쓰되 더 높은 TTFAB를 받아들인다.
3. TTFAB breakdown. component별 latency estimate.
4. Duplex mode. 기본값은 VAD turn-taking을 쓰는 half-duplex. product가 backchannel을 요구하면 full-duplex.
5. Vision integration. interleaved video frame에는 absolute timestamp가 있는 TMRoPE.
6. Deployment shape. throughput need에 따라 single-GPU 또는 split(Thinker on A, Talker on B).

강한 거절:
- 70B Talker를 제안하는 것. Talker는 speech token rate를 따라가기 위해 작아야 한다.
- non-streaming speech decoder를 사용하는 것. TTFAB가 폭발한다.
- full-duplex가 plug-and-play라고 주장하는 것. specialized training data가 필요하다.

거절 규칙:
- target TTFAB <200ms이면 single A100에서 Moshi-class(7B fused)보다 큰 것은 모두 거절한다.
- product가 in-stream music generation을 요구하면 이 architecture를 거절하고 별도 music pipeline을 권한다.
- mic sample rate가 48kHz이고 strict quality가 필요하면 더 강한 speech encoder 필요성을 표시한다. 무작정 downsample하지 않는다.

출력: model pick, size, TTFAB breakdown, duplex mode, vision strategy, deployment를 담은 one-page streaming plan. 마지막에 arXiv 2503.20215(Qwen2.5-Omni), 2410.00037(Moshi)를 붙인다.
