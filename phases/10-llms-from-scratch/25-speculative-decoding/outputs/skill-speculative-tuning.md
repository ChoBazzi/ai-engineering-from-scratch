---
name: speculative-tuning
description: decode workload를 profile하고 speculative decoding을 위한 draft model, draft length K, temperature gate, fallback policy를 선택합니다.
version: 1.0.0
phase: 10
lesson: 25
tags: [speculative-decoding, draft-model, alpha, throughput, inference, decode-latency]
---

target model(size, family, tokenizer), workload telemetry(task mix, prompt-vs-decode token ratio, p50/p99 decode latency, accelerator and HBM headroom, average batch size, sampling temperature distribution), available draft checkpoints가 주어지면 다음을 출력하세요:

1. Draft choice. same-family small(Llama-3.2-1B for Llama-70B), distilled draft(Qwen3-0.6B-spec), target에 붙인 Medusa heads, 또는 30 percent FLOP cost ratio보다 가까운 draft가 없을 때 "no spec decode" 중에서 고르세요. tokenizer가 target과 byte-for-byte로 일치하는지 확인하고, mismatched tokenizer는 거부하세요.
2. Draft length K. E[tokens] / (1 + K x c)의 argmax입니다. 여기서 c는 draft-to-target cost ratio입니다. in-distribution data 5_000 token calibration run에서 측정한 alpha로 K=2, 3, 4, 5, 6에 대해 계산 과정을 보이세요. default는 chat K=4, code K=6, high-temperature creative writing K=2입니다.
3. Temperature gate. 이 값보다 높으면 spec decode를 비활성화하는 temperature threshold를 설정하세요. default는 0.8입니다. calibration에서 alpha가 더 일찍 collapse하면 0.6으로 낮추세요. request별 inspection에 50 microseconds를 넘게 추가하는 temperature gate는 거부하세요.
4. Tree budget. serving stack이 tree drafting을 지원하면 batch 8 미만에서는 작은 fixed tree(depth 2, branch 3-2)를 고르고, batch 32 초과에서는 flat chain을 고르세요. verifier의 KV scratch size를 bytes로 말하고 HBM headroom에 들어가는지 확인하세요.
5. Fallback policy. 서버가 해당 request stream에 대해 plain autoregressive decode로 돌아가는 metric(last 1_000 verifies에 대한 sliding-window measured alpha)과 threshold(alpha 0.4 미만)를 명명하세요. fallback decision의 per-request lifetime을 포함하세요.

verifier가 compute-bound가 되는 지점보다 batch size가 크면 spec decode를 거부하세요. 그 지점 위에서는 speculator가 흡수하려던 unused FLOPs가 더 이상 존재하지 않으며 throughput이 떨어집니다. measured alpha가 0.4 미만인 task family에는 spec decode를 거부하세요. draft overhead가 지배하고 wall-clock latency가 악화됩니다. target 대비 held-out 1_000-token sample에서 검증되지 않은 draft는 거부하세요. 검증되지 않은 draft는 silent KL drift입니다.

Example input: "Llama-3.3-70B on 8xH100, chat workload, batch 16, p50 decode 28 ms, p99 60 ms, temperature distribution mean 0.4 / max 1.2, calibration shows alpha 0.78 on chat, 0.61 on code."

Example output:
- Draft: Llama-3.2-1B-Instruct-spec. Same tokenizer, same family, ratio c approx 0.03.
- K: 4. E[tokens/verify] = 3.4 chat, 2.5 code. K=5는 chat에서 0.1 token만 얻고 0.03 extra c를 지불하므로 reject.
- Temperature gate: 0.8. 0.8보다 높으면 calibration set에서 alpha가 0.45 아래로 떨어집니다.
- Tree budget: depth 2 branch (3, 2). batch 16에서 KV scratch 480 MB는 들어갑니다.
- Fallback: last 1_000 verifies에 대한 sliding-window alpha가 0.40 미만이면 해당 stream에서 30 s 동안 spec decode를 비활성화한 뒤 다시 probe합니다.
