---
name: open-model-picker
description: 주어진 배포 대상에 맞는 open LLM 계열, quantization, inference stack을 선택합니다.
version: 1.0.0
phase: 10
lesson: 14
tags: [open-models, llama, deepseek, mixtral, qwen, gemma, moe, gqa, mla, quantization]
---

배포 대상(GPU 종류, GPU당 VRAM, GPU 개수, 목표 context length, 목표 p50/p99 latency, 최대 동시 요청 수)과 작업 프로파일(chat, code, reasoning, long-context retrieval, tool use)을 받아 Lesson 14의 여섯 가지 architecture knob 각각에 대한 명시적 근거와 함께 open model 및 serving stack을 추천하세요.

작성 항목:

1. 모델 shortlist. 후보 세 개를 제시하고, 각 후보에 total params, active params(MoE 고려), architecture flags(norm / activation / position / attention / MoE / context), shortlist에 들어간 단 하나의 이유를 포함하세요.
2. 메모리 예산 확인. 최상위 후보에 대해 BF16 및 선택한 quantization에서의 weight memory, target context와 target batch size에서의 KV cache, activation headroom을 계산하세요. weights + KV cache + activations가 가용 VRAM을 넘으면 추천을 중단하세요.
3. Quantization 선택. GPTQ-4bit, AWQ-4bit, FP8, BF16 중 고르세요. 작업의 accuracy 민감도에 비추어 정당화하세요(code / math / reasoning 작업은 chat이나 retrieval보다 공격적 quantization의 손실이 큽니다).
4. Inference stack. vLLM, TensorRT-LLM, SGLang, llama.cpp 중 고르세요. continuous batching 필요성, speculative decoding 지원, quantization format 호환성, single-node vs multi-node topology를 기준으로 정당화하세요.
5. Throughput sanity check. GPU memory bandwidth(decode)와 TFLOPs(prefill)를 기준으로 prefill tokens/sec 및 decode tokens/sec를 추정하세요. decode throughput이 대상의 concurrent-user floor보다 낮으면 추천을 거부하세요.
6. Fallback. 최상위 후보가 VRAM 또는 throughput budget을 초과할 때의 두 번째 선택지를 제시하세요. 항상 하나는 이름으로 명시하세요.

강한 거부 사유:
- Offloading 또는 공격적 quantization 없이 단일 24GB 소비자 GPU에서 30B를 넘는 dense model.
- Expert-parallel 지원이 없는 serving stack 위의 MoE model.
- GQA 또는 MLA가 없는 architecture에서 long-context(128k+) 사용(KV cache가 폭증합니다).
- 구체적인 model revision을 명시하지 않는 추천(예: "Llama 3"가 아니라 "Llama 3 8B Instruct v3.1").

출력: model, quantization, stack을 나열하고 각 결정에 번호가 붙은 근거를 제공하는 한 페이지 추천서. 마지막에는 선택을 뒤집을 구체적 capability 또는 deployment parameter를 이름으로 밝히는 "worth reconsidering if..." 문단을 붙이세요.
