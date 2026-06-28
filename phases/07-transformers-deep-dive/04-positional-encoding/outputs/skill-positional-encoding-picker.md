---
name: positional-encoding-picker
description: Context length와 training budget이 주어졌을 때 positional encoding(RoPE, ALiBi, sinusoidal)과 scaling strategy를 고릅니다.
version: 1.0.0
phase: 7
lesson: 4
tags: [transformers, positional-encoding, rope, alibi]
---

Transformer spec(target context length at inference, trained context length, extrapolation requirement, fine-tune budget in tokens)이 주어지면 다음을 출력하세요.

1. 기본 encoding. RoPE, ALiBi, sinusoidal, learned-absolute 중 하나. 한 문장 이유를 덧붙입니다.
2. hyperparameter. RoPE라면 `base` value와 even split을 위한 `d_head` requirement를 제시합니다. ALiBi라면 slope formula를 제시합니다. Sinusoidal이라면 `max_len`을 제시합니다.
3. 확장 전략. target > trained라면 NTK-aware scaling factor, YaRN config, LongRoPE spec, 또는 position-interpolation ratio를 제시합니다. Fine-tune token budget을 명시합니다.
4. 테스트 계획. max context에서 NIAH(needle-in-a-haystack) pass rate target, trained-length baseline 대비 perplexity가 X 이내라는 기준을 제시합니다.
5. fallback. Long-context eval이 실패하면 무엇을 할지 적습니다. 더 큰 `base`로 retrain, ALiBi로 전환, 또는 deployed context length 제한 중 하나입니다.

2026년에 새 model에는 sinusoidal 또는 learned-absolute를 추천하지 마세요. Extrapolate하지 못하며 모든 현대 stack은 RoPE 또는 ALiBi를 가정합니다. Fine-tune stage 없이 RoPE를 trained length의 8×를 넘어 scale하지 마세요. Full deployed length에서 NIAH run 없이 long-context config를 ship하지 마세요.
