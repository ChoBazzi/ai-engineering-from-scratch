---
name: video-brief
description: 비디오 brief를 2026년 비디오 generator용 model + prompt + shot plan으로 번역한다.
version: 1.0.0
phase: 8
lesson: 10
tags: [video, diffusion, sora, veo, kling]
---

비디오 brief(duration, aspect ratio, style, subject, camera plan, audio needs, fidelity bar, budget)가 주어지면 다음을 출력한다.

1. 모델 + hosting. Sora, Veo 3, Kling 2.1, Runway Gen-3, Pika 2.0, CogVideoX, HunyuanVideo, WAN 2.2, 또는 Mochi-1. duration / quality / license와 연결된 한 문장 이유.
2. Prompt scaffolding. (a) camera language(establishing, tracking, dolly, crane, handheld), (b) subject + action, (c) lighting + style, (d) negative prompt 또는 style toggles. Sora는 50-150 tokens, Runway는 20-60 tokens를 목표로 한다.
3. Shot plan. Single-clip vs stitched multi-shot, keyframe 또는 first-frame anchors, shot별 I2V vs T2V.
4. Seed + 재현성. Shot별 seed, version pin, tooling repo.
5. QA checklist. Flicker, identity consistency, physics violations, watermark compliance를 frame-by-frame으로 점검한다.
6. 오디오. Veo 3에서는 native, 그 외에는 bolt-on(ElevenLabs, Suno 또는 licensed stems + lip-sync pass).

free tier에서 1080p로 10초를 넘는 continuous motion을 약속하는 것은 거부한다(Pika / Kling / Runway는 10s가 cap이며, 더 긴 run은 stitched된다). release 없이 real people의 likeness 생성을 거부한다. 2026년에 real-time 4K generation을 암시하는 brief는 flag한다. 현재 최고 수준은 hosted endpoint에서 1080p 6초 clip당 생성 약 30초다.
