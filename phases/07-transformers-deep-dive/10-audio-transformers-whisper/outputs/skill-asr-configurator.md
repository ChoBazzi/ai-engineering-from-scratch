---
name: asr-configurator
description: 새 음성 파이프라인에 맞는 ASR 모델(Whisper 변형 / Moonshine / faster-whisper)과 디코딩 파라미터를 고릅니다.
version: 1.0.0
phase: 7
lesson: 10
tags: [transformers, whisper, asr, speech]
---

음성 작업(transcription / translation / streaming / on-device), 언어, 오디오 특성(잡음, 억양, 길이), 지연 시간/품질 목표를 입력받아 다음을 출력하세요.

1. 모델 선택. 다음 중 하나입니다. faster-whisper large-v3-turbo(프로덕션 기본값), whisper large-v3(최고 품질, 다국어), whisper medium(중간급), Moonshine base(edge), distil-whisper(영어에서 2× 빠름). 한 문장 이유를 덧붙이세요.
2. 양자화. int8_float16(CPU 기본값), float16(GPU 기본값), fp32(research). VRAM 영향을 표시하세요.
3. 디코딩. Beam width(보통 5, streaming은 1), temperature fallback schedule, log-prob threshold, no-speech threshold, VAD gate on/off를 제시하세요.
4. 청킹. 30 s fixed window와 streaming chunks(보통 2 s overlap이 있는 10 s) + VAD-based segmentation을 비교하세요. overlap에 대한 post-merge strategy를 문서화하세요.
5. 후처리. Timestamp alignment(WhisperX forced alignment), punctuation restoration, diarization(pyannote)을 제시하세요. 작업에 필요한 항목을 표시하세요.

프로덕션에는 plain OpenAI Whisper(reference implementation)를 추천하지 마세요. `faster-whisper`가 동일 출력으로 4× 빠릅니다. 문서화된 이유가 없다면 VAD 없는 streaming ASR 배포를 거부하세요. 입력이 multi-speaker일 가능성이 있으면 single-speaker 가정을 반드시 표시하세요.
