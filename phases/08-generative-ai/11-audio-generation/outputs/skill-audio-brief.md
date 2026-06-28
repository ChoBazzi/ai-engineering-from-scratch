---
name: audio-brief
description: 오디오 brief를 TTS, music, SFX 전반의 model + prompt + eval plan으로 번역한다.
version: 1.0.0
phase: 8
lesson: 11
tags: [audio, tts, music, sfx, codec]
---

오디오 brief(task: TTS / music / SFX / voice clone, duration, style, voice or genre, license constraints, real-time or offline, quality bar)가 주어지면 다음을 출력한다.

1. 모델 + hosting. ElevenLabs V3, OpenAI TTS, XTTS v2, Suno v4, Udio, Stable Audio 2.5, MusicGen 3.3B, AudioCraft 2, 또는 GPT-4o realtime. 한 문장 이유.
2. Prompt format. TTS: text + voice prompt(3-10 s sample 또는 voice ID) + emotion / pace tags. Music: genre + instrumentation + mood + BPM + structural markers. SFX: onomatopoeia + source + duration hint.
3. Codec + generator + vocoder chain. 특정 codec(Encodec 32 kHz, DAC 44 kHz, custom)과 generator 선택(token-AR vs flow-matching)을 명명한다.
4. Seed + 재현성. Seed pin, version pin, prompt hash.
5. 평가. TTS에는 MOS(mean opinion score) 또는 A/B, music에는 CLAP score, TTS transcription에는 CER, SFX에는 user listening test.
6. 가드레일. Voice-clone consent + watermark(PerTh / SynthID-audio), music output의 copyright scan, training-data policy check.

소유자의 verified consent 없이 어떤 voice도 clone하지 않는다(Cassette 시대의 "3-second prompt"는 동의가 아니다). unlicensed reference material이 포함된 music 출시는 거부한다. streaming token-AR model을 사용하지 않으면서 real-time target &lt; 200 ms를 요구하는 경우 flag한다. diffusion-based audio는 2026년에 sub-300 ms TTFB를 충족할 수 없다.
