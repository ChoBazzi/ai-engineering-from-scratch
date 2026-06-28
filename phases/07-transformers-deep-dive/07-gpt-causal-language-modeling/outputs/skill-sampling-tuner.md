---
name: sampling-tuner
description: 주어진 generation task에 맞는 decoding strategy(greedy / temperature / top-k / top-p / min-p / speculative)를 고릅니다.
version: 1.0.0
phase: 7
lesson: 7
tags: [gpt, sampling, decoding, inference]
---

generation task(code, creative writing, reasoning, dialogue, structured output)와 latency/quality target이 주어지면 다음을 출력하세요.

1. sampling method. greedy, temperature-only, top-k, top-p, min-p, beam-k, speculative 중 하나를 고릅니다. 한 문장 이유를 제시합니다.
2. parameter values. temperature, top-k, top-p, min-p, repetition penalty를 task type에 맞춘 구체적 숫자로 제시합니다. 예: code에는 temperature 0.2 + top-p 1.0, chat에는 min-p 0.1 + temperature 0.7.
3. stop conditions. `max_new_tokens`, stop token list, pattern-based stop(예: 닫는 `</tool_call>`).
4. determinism toggle. 재현성을 위한 fixed seed를 제시하고, 해당 use case(eval, legal 등)에 필요한지 표시합니다.
5. quality check. task objective에 대한 한 줄 test를 제시합니다(compile/pass unit tests, factuality, format validity 등).

structured output이나 code completion에 temperature > 1.0을 권장하지 마세요. hallucination risk가 급격히 올라갑니다. open-ended dialogue에 pure greedy를 권장하지 마세요. model이 loop에 빠집니다. model이 template/tool을 생성할 수 있는데 stop-token list가 지정되지 않은 sampling config는 배포하지 마세요.
