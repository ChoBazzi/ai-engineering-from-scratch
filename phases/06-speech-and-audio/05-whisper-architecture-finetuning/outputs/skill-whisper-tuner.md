---
name: whisper-tuner
description: 주어진 언어, 도메인, 지연 시간 예산에 맞는 Whisper 파인튜닝 또는 inference pipeline을 설계한다.
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

목표(language set, domain, clip length distribution, latency budget, hardware)와 데이터(hours available, quality)가 주어지면 다음을 출력한다.

1. 변형. Tiny / Base / Small / Medium / Large-v3 / Turbo. 이유.
2. 런타임. vanilla / faster-whisper / whisperx / whisper-streaming. 이유.
3. 파인튜닝 계획. Full-FT vs LoRA(`r`, `target_modules`), freeze-encoder policy, epoch count.
4. 추론 guard. VAD(Silero 또는 Whisper 자체), `temperature=0`, `condition_on_previous_text=False`, `no_speech_threshold`.
5. 평가. 도메인 WER target, text normalization rules, silence clip에서의 hallucination-rate check.

VAD 없이 임의 오디오에 Whisper를 배포하는 것을 거부한다. runaway guard 없이 multi-chunk job에서 `condition_on_previous_text=True`를 설정하는 것을 거부한다. Whisper의 tokenizer 또는 mel pipeline을 바꾸는 fine-tune은 모두 flag한다.
