---
name: editing-pipeline
description: 원본과 편집 설명에서 출시 가능한 출력까지 이어지는 이미지 편집 파이프라인을 계획한다.
version: 1.0.0
phase: 8
lesson: 09
tags: [inpaint, outpaint, edit, sam]
---

원본 이미지, 목표 편집(remove X, replace Y with Z, extend canvas, restyle region, change season / time-of-day), 품질 기준(draft / portfolio / print)이 주어지면 다음을 출력한다.

1. 마스크 전략. 명시적 brush mask, SAM 2 click / box prompt, text phrase에 대한 Grounded-SAM, 또는 RMBG(background removal용). 한 문장으로 이유를 설명한다.
2. Base model + mode. instruction edits에는 SD-Inpaint / SDXL-Inpaint / Flux-Fill / Flux-Kontext, 마스크가 없으면 SDEdit noise-level(0.3 / 0.6 / 0.9).
3. Prompt scaffolding. 새 content만이 아니라 편집 후 전체 이미지를 설명한다. negative prompt를 포함한다.
4. CFG + strength + feather. Mask feather 8-16 px, SDXL-inpaint의 CFG는 약 5-7, Flux는 3-4. Strength는 full regenerate에는 0.8-1.0, preserve에는 0.3-0.5.
5. 가드레일. NSFW / deepfake / trademark detection hook, face-swap policy gate, reversibility(mask + seed 저장).

명시적 policy check 없이 알아볼 수 있는 public figure에 대한 identity edit는 출시를 거부한다. 원본 canvas의 최소 30%가 anchor로 남아 있지 않은 이미지는 outpaint를 거부한다(맥락이 너무 적으면 모델이 hallucinate한다). t/T &gt; 0.7인 SDEdit run과 fidelity target "preserve subject"의 조합은 likely mismatch로 flag한다.
