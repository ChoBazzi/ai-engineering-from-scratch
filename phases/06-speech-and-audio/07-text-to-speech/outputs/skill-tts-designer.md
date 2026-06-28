---
name: tts-designer
description: 주어진 language, style, latency target에 맞춰 TTS model, voice, text-normalization scope, evaluation plan을 고른다.
version: 1.0.0
phase: 6
lesson: 07
tags: [audio, tts, speech-synthesis]
---

목표(language(s), voice style, latency budget, CPU vs GPU, license constraints)와 content(domain, OOV density, punctuation richness)가 주어지면 다음을 출력한다.

1. 모델. Kokoro / XTTS v2 / F5-TTS / VITS / StyleTTS 2 / commercial API. 한 문장 이유.
2. 텍스트 프런트엔드. Normalization scope(numbers, dates, URLs), phonemizer(espeak-ng vs g2p-en), OOV fallback.
3. 음성. preset name 또는 reference clip spec(seconds, noise floor, accent match).
4. 품질 목표. target UTMOS, Whisper를 통한 CER, cloning 시 SECS.
5. 평가 계획. 숫자, homograph, proper noun, long sentence를 포함하는 20-utterance test set.

text normalizer가 없는 production TTS는 거부한다. 사용자 동의와 watermarking 없는 voice cloning은 거부한다. Kokoro deployment가 영어 이외의 언어를 말하도록 요구되면 flag한다.
