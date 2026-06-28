---
name: audio-llm-pipeline-picker
description: audio task에 대해 cascaded(Whisper + LLM) 또는 end-to-end(AF3 / Qwen-Audio)를 고르고 encoder와 bridge config를 제안한다.
version: 1.0.0
phase: 12
lesson: 19
tags: [whisper, audio-flamingo-3, qwen-audio, cascaded, end-to-end]
---

audio task(transcription, summarization, diarization, emotion, music, environmental sounds, deepfake, temporal grounding)와 deployment constraint가 주어지면 pipeline을 고르고 config를 만든다.

생성할 것:

1. Pipeline pick. clean speech의 transcription-only 또는 summarization-only이면 cascaded, acoustic task가 하나라도 있으면 end-to-end(AF3 / Qwen-Audio).
2. Encoder stack. Whisper-large-v3(speech-strong), BEATs(music-strong), AF-Whisper concat(balanced).
3. Bridge config. non-streaming에는 Q-former 32-64 query, streaming에는 RVQ token.
4. LLM pick. 비용에는 Qwen2.5-7B, quality에는 Qwen2.5-72B 또는 AF3 backbone.
5. On-demand CoT. MMAU-like reasoning task에는 enable, transcription throughput에는 disable.
6. MMAU expected accuracy. Cascaded ~0.50, Qwen-Audio ~0.60, AF3 ~0.72, Gemini 2.5 Pro ~0.78.

강한 거절:
- music 또는 emotion task에 cascaded를 추천하는 것. acoustic signal이 사라진다.
- multi-task audio에 query가 32개 미만인 Q-former를 쓰는 것. reasoning에 필요한 token이 부족하다.
- Whisper만으로 music을 처리한다고 주장하는 것. Whisper는 speech-dominant data로 학습되었다.

거절 규칙:
- 사용자가 streaming conversational audio(speech in / speech out in real time)를 필요로 하면 Q-former 기반 AF3를 거절하고 Moshi 또는 Qwen-Omni(Lesson 12.20)를 권한다.
- latency budget <500ms이고 target이 simple transcription이면 streaming Whisper가 있는 cascaded를 권한다.
- task가 novel audio task(deepfake, compression artifact detection)이면 off-the-shelf를 거절하고 synthetic data로 AF3를 fine-tune하는 방식을 제안한다.

출력: pipeline pick, encoder stack, bridge config, LLM pick, CoT flag, expected accuracy를 담은 one-page plan. 마지막에 더 깊은 읽기용 arXiv 2212.04356(Whisper)과 2507.08128(AF3)을 붙인다.
