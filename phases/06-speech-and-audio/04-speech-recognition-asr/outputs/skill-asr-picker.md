---
name: asr-picker
description: 주어진 배포 목표에 맞는 ASR model, decoding strategy, chunking, LM fusion을 고릅니다.
version: 1.0.0
phase: 6
lesson: 04
tags: [audio, asr, speech-recognition]
---

배포 목표(language list, domain, latency budget, hardware, offline / streaming, clip duration)가 주어지면 다음을 출력하세요.

1. 모델. Whisper-large-v3-turbo / Parakeet-TDT / Canary-Flash / wav2vec 2.0 / Moonshine. 한 문장 이유.
2. 디코딩. Greedy / beam width / temperature fallback / LM fusion weight. Quality budget에 연결된 이유.
3. 청크 처리와 VAD. Chunk length, stride, Silero-VAD 또는 Whisper 자체 gate 사용 여부.
4. 언어 정책. Force language vs auto-LID; cross-lingual frame 처리 방법.
5. 평가 계획. Domain test set의 WER, coverage-per-speaker, silence clip에서 hallucination rate.

VAD gating이 없는 long-form Whisper 배포는 거부하세요(침묵에서 hallucination이 잘 생김). Text normalization(lower, punct strip) 없는 WER 보고도 거부하세요. LM 없이 beam-width > 16을 쓰면 표시하세요. Blank 위의 raw beam은 도움이 되지 않습니다.
