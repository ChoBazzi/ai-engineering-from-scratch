---
name: skill-quantization
description: hardware, quality, latency constraint에 따라 LLM 배포에 맞는 quantization strategy를 고릅니다.
version: 1.0.0
phase: 10
lesson: 11
tags: [quantization, inference, deployment, optimization, fp8, int4, int8, gptq, awq, gguf]
---

# Quantization 의사결정 Framework

language model을 배포할 때 이 framework를 사용해 올바른 number format, quantization method, quality validation strategy를 선택하세요.

## 입력 요구사항

다음을 제공하세요.

- **Model**(이름, parameter count, original precision)
- **Target hardware**(GPU model/VRAM, CPU, Apple Silicon, edge device)
- **Latency target**(tokens/second, time to first token)
- **Quality floor**(허용 가능한 최대 perplexity increase, benchmark delta)
- **Serving pattern**(batch size, max context length, concurrent users)

## 빠른 선택

| Your Situation | Format | Method | Expected Quality Loss |
|---------------|--------|--------|----------------------|
| H100 GPU, maximum throughput | FP8 E4M3 | Native H100 casting | < 0.1% |
| A100/A10, need 2x throughput | INT8 | LLM.int8() or SmoothQuant | < 0.5% |
| Single 24GB GPU, 70B model | INT4 | AWQ or GPTQ | 1-3% |
| MacBook / Apple Silicon | INT4 GGUF | Q4_K_M via llama.cpp | 1-2% |
| Mobile / edge device | INT4 or INT3 | QAT + device-specific | 2-5% |
| Maximum compression, some loss OK | INT2 | QuIP# or AQLM | 5-15% |
| Training (mixed precision) | BF16 + FP32 accum | Native framework support | 0% |

## Component별 precision 선택

모든 tensor에 같은 처리를 하면 안 됩니다.

| Component | Safe Minimum | Recommended | Avoid |
|-----------|-------------|-------------|-------|
| FFN weights | INT4 | INT4 (AWQ/GPTQ) | INT2 without QAT |
| Attention weights | INT4 | INT8 or FP8 | INT2 |
| Embedding layer | INT8 | FP16 (keep original) | INT4 |
| Output head | INT8 | FP16 (keep original) | INT4 |
| KV cache | FP8 | FP8 or INT8 | INT4 at long context |
| Attention logits | FP16 | FP16 or BF16 | INT8 |
| Activations (inference) | INT8 | FP8 or INT8 | INT4 |

## 방법 비교

### GPTQ

- **When:** GPU inference이고 Hugging Face-compatible model이 필요할 때
- **Calibration data:** 128 examples, 각 2048 tokens
- **Time:** A100에서 70B 기준 30-60분
- **Tooling:** `auto-gptq`, `exllama`, `exllamav2`
- **Strength:** 검증이 잘 되어 있고 Hugging Face에 huge model zoo가 있음
- **Weakness:** AWQ보다 적용이 느리고 일부 model에서 quality가 조금 낮음

### AWQ

- **When:** GPU inference이고 best quality-per-bit를 원할 때
- **Calibration data:** 128 examples
- **Time:** A100에서 70B 기준 15-30분
- **Tooling:** `autoawq`, `vLLM`(native support)
- **Strength:** 최고의 INT4 quality, 빠른 적용, vLLM integration
- **Weakness:** GPTQ보다 model zoo가 작음

### GGUF

- **When:** CPU inference, Apple Silicon, llama.cpp ecosystem
- **Variants:** Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0, F16
- **Recommended default:** Q4_K_M(best quality/size balance)
- **Tooling:** `llama.cpp`, `ollama`, `LM Studio`
- **Strength:** self-contained file, mixed precision, 거대한 ecosystem
- **Weakness:** GPU에는 최적이 아님(CPU/Metal용으로 설계)

### SmoothQuant

- **When:** GPU의 INT8에서 weight와 activation quantization이 모두 필요할 때
- **Key idea:** per-channel scaling으로 quantization difficulty를 activation에서 weight로 옮김
- **Tooling:** `smoothquant`, `TensorRT-LLM`
- **Strength:** weight와 activation이 모두 INT8인 W8A8을 가능하게 해 2x speedup
- **Weakness:** INT8 전용이며 INT4로 확장되지 않음

## 품질 검증 프로토콜

quantize한 뒤 배포 전에 검증하세요.

1. **Perplexity test.** WikiText-2 또는 domain corpus에서 계산합니다. Delta < 0.5는 excellent, 0.5-1.0은 good, > 2.0은 problem입니다.
2. **Benchmark sweep.** MMLU(general), GSM8K(math), HumanEval(code)를 실행합니다. math와 code가 precision loss에 가장 민감합니다.
3. **Output comparison.** original model과 quantized model에서 각각 100개 response를 생성합니다. LLM-as-judge로 win rate를 계산합니다. Target: quantized model이 prompt의 > 90%에서 win 또는 tie.
4. **Latency measurement.** batch size 1과 target batch size에서 tokens/second를 측정합니다. speedup이 quality cost를 정당화하는지 확인합니다.
5. **Long-context test.** long context(> 4K tokens)를 serving한다면 maximum context length에서 test하세요. KV cache quantization error는 sequence length와 함께 누적됩니다.

## Memory Budget 계산기

```text
Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB (relatively constant, depends on batch size)
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)
```

Llama 3 70B, INT4, 32K context 예:

- Weights: 70B * 4 / 8 / 1.07 = 32.6 GB
- KV cache (FP16): 2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 = ~40 GB
- KV cache (FP8): ~20 GB
- Total with FP8 KV: ~55 GB(80GB A100 한 장에 들어감)

## 흔한 실수

| Mistake | Why It Fails | Fix |
|---------|-------------|-----|
| embedding layer를 INT4로 quantize | 첫 layer의 error가 전체 model로 증폭됨 | embedding은 FP16 또는 INT8 유지 |
| INT4에 per-tensor scale 사용 | outlier row 하나가 모든 row의 precision을 망침 | per-channel 또는 per-group scale 사용 |
| GPTQ/AWQ를 calibration하지 않음 | representative data 없이는 scale factor가 틀림 | domain에서 128 examples 사용 |
| 모든 layer에 같은 bit-width 사용 | first/last layer가 더 민감함 | mixed precision: first/last는 더 높은 bit |
| 매우 긴 context에서 KV cache quantize | error가 sequence length와 함께 누적됨 | KV cache는 INT4가 아니라 FP8 사용 |
| quality validation 생략 | 일부 model은 특히 boundary에서 quantize가 나쁨 | 항상 perplexity + task eval 실행 |

## 배포 Recipe

### Recipe 1: AWQ를 쓰는 vLLM(GPU server)

```text
pip install vllm autoawq
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

### Recipe 2: GGUF를 쓰는 llama.cpp(MacBook)

```text
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99
```

### Recipe 3: FP8을 쓰는 TensorRT-LLM(H100)

```text
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
```
