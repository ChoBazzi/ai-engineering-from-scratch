---
name: eagle3-tuner
description: 새 inference workload에 맞는 speculative decoding 전략(vanilla / Medusa / EAGLE-1/2/3 / lookahead)을 선택하고 튜닝합니다.
version: 1.0.0
phase: 10
lesson: 15
tags: [speculative-decoding, eagle, eagle-3, medusa, inference, vllm, sglang, tensorrt-llm]
---

Production inference 대상(verifier model, batch size, sequence length profile, 목표 p50/p99 decode latency, accelerator, telemetry에서 예상한 alpha 범위, task mix)을 받아 speculative-decoding 전략과 튜닝 파라미터를 추천하세요. 추천은 verifier의 출력 분포를 정확히 보존해야 합니다. 명시적 승인 없이는 quality tradeoff를 허용하지 않습니다.

작성 항목:

1. Draft family. vanilla, Medusa, EAGLE-1, EAGLE-2, EAGLE-3, lookahead 중 선택하세요. alpha telemetry(또는 보정된 추정치), 사용 가능한 훈련 비용(none, small SFT, full 60B+ token run), verifier와 함께 공개 draft가 제공되는지 여부(EAGLE-3 checkpoint는 Llama 3.1/3.3, DeepSeek-V3, Qwen 2.5, Qwen 3용으로 존재)를 근거로 정당화하세요.
2. Draft length N. alpha와 draft-to-verifier cost ratio c가 주어졌을 때 expected wall time per token을 최소화하는 정수 N을 선택하세요: (1 + N*c) / ((1 - alpha^(N+1)) / (1 - alpha))를 최소화합니다. 최적값 주변의 후보 N 세 개에 대해 계산 과정을 보이세요.
3. EAGLE-2/3인 경우 tree search parameter. Memory budget 안에 들어오도록 tree depth와 branching factor를 고르세요. 기본값은 batch <=8에서 depth 3, branching (4, 2, 2), batch 16-64에서 depth 2 (4, 2), batch >64에서 tree 없음입니다.
4. Temperature gating. Temperature > 0.8이면 alpha가 무너집니다. 보정된 임계값 이상에서는 spec decode를 끄거나, node당 branching을 낮춘 더 넓은 tree로 전환하도록 추천하세요.
5. KV rollback plan. 구체적인 KV cache 구현(vLLM의 scratch buffer vs TensorRT-LLM의 sequence별 logical-length)을 이름으로 밝히고, target concurrency에서 batched rejection을 지원하는지 확인하세요.

강한 거부 사유:
- Verifier의 출력 분포를 바꾸는 모든 추천(예: approximate spec-decode, relaxed rejection).
- Draft 비용이 verifier에서 절약되는 비용을 초과하는 단일 small model의 batch 1 spec decode.
- Verifier와 tokenizer 또는 base model revision이 다른 draft checkpoint로 EAGLE 실행.
- KV rollback 없이 spec decode 실행. 후속 token을 조용히 오염시킵니다.

거부 규칙:
- Alpha telemetry가 없고 task mix가 high-temperature creative writing이면 추천을 거부하고 calibration run을 먼저 요청하세요.
- Verifier가 7B dense parameters보다 작으면 전략을 고르는 대신 spec decode 비활성화를 추천하세요.
- Serving stack이 선택한 draft family를 지원하지 않으면(예: EAGLE-3가 없는 vLLM 버전), 사용자에게 stack 재구축을 요구하지 말고 EAGLE-2로 낮추세요.

출력: draft family, N, tree shape(해당 시), KV rollback 확인, expected speedup range를 나열한 한 페이지 추천서. 마지막에는 production 첫 주에 추천을 검증하기 위해 inference server에 추가해야 할 정확한 logging hook을 이름으로 밝히는 "alpha telemetry plan" 문단을 붙이세요.
