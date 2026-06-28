---
name: eval-report
description: 샘플 품질, adherence, preference, 실패 감사를 포함하는 전체 생성 모델 평가를 계획합니다.
version: 1.0.0
phase: 8
lesson: 14
tags: [evaluation, fid, clip, elo]
---

새 generative-model checkpoint, reference baseline, modality(image / video / audio / 3D)가 주어지면 전체 eval plan을 출력하세요.

1. 샘플 품질. Held-out real set 대비 10-30k samples에서 FID / FD-DINO / CMMD를 측정합니다. 해상도는 맞춥니다. 3-seed mean +/- std를 보고합니다.
2. 준수도. Prompt-image pairs에서 CLIP score / CMMD를 측정합니다. Text-to-image에는 HPSv2 + ImageReward + PickScore를 포함합니다. Video에는 vision-language metrics(V-Eval)를 추가합니다. Audio에는 CLAP + MOS를 추가합니다.
3. 쌍대 선호도. Baseline 대비 200-2000 prompts에서 blinded A/B를 수행합니다. Human + LLM-judge + PartiPrompts coverage를 포함합니다.
4. Category breakdown. Prompt category(people, animals, text rendering, composition, style)별 성능을 봅니다. Global metrics가 개선되어도 category별 regression은 표시합니다.
5. Safety / misuse. Top-K generations에 대해 NSFW classifier, deepfake detector, watermark check, copyright similarity scan을 수행합니다.
6. Sign-off. 명시적 gate: FID가 baseline의 +5% 이내이거나, human win rate가 >55%이거나, 문서화된 qualitative advantage가 있어야 합니다. 단일 metric claim은 허용하지 않습니다.

N < 5000에서 FID 보고를 거부하세요. 모델이 학습 중 봤을 수 있는 prompt로 계산한 benchmark 출시는 거부하세요. Human cross-check 없이 LLM-judge 결과만 보고하는 것을 거부하세요. Metric이 "20% 올랐다"고 주장하면서 absolute base value를 보고하지 않거나 single seed만 보고하는 claim은 모두 표시하세요.
