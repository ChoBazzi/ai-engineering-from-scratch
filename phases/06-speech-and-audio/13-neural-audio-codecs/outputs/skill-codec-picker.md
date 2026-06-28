---
name: codec-picker
description: 주어진 generative 또는 compression task에 맞는 neural audio codec(EnCodec / DAC / SNAC / Mimi)을 선택한다.
version: 1.0.0
phase: 6
lesson: 13
tags: [codec, encodec, dac, snac, mimi, rvq, semantic-tokens]
---

task(generative LM, compression, full-duplex dialogue, music editing, fidelity target)가 주어지면 다음을 출력하라.

1. 코덱. EnCodec-24k · EnCodec-48k · DAC-44.1k · SNAC-24k · Mimi · (fallback: non-neural compression용 Opus). 한 문장 이유.
2. Frame rate + codebooks. Bitrate budget, codebook count(보통 4-12), target clip duration에 대한 sequence length.
3. 토큰화 방식. Flat vs hierarchical(SNAC) vs semantic+acoustic(Mimi). LM이 token을 consume하는 방식.
4. 디코더. In-codec decoder · external vocoder(HiFi-GAN) · LM-only(vocoder 없음, codec token 직접 예측). 이유를 설명한다.
5. 학습 영향. encoder/decoder를 학습해야 하는가? domain audio에서 fine-tune해야 하는가(speech-only → domain-specific music)? off-the-shelf로 freeze하는가?

tight latency budget의 AR-LM workload에는 DAC를 거부하라. 86 Hz frame rate × 8 codebooks = 10 s당 5,504 token이라 fast generation에는 너무 길다. music에는 Mimi를 거부하라. speech-tuned이기 때문이다. semantic-conditional generation에는 EnCodec을 거부하라. semantic codebook이 없어 text에서 흐릿한 speech가 나온다.

예시 입력: "Build an AR LM for text-to-speech TTS. Target TTFA 200 ms. English only."

예시 출력:
- 코덱: Mimi. Semantic+acoustic split은 text → codebook 0 → codebooks 1-7 factorization을 가능하게 하며, 빠르고 voice cloning도 지원한다.
- Frame rate + codebooks: 12.5 Hz · 8 codebooks · 4.4 kbps. 10 s = 1,000 tokens.
- 토큰화: text + speaker reference에서 codebook 0을 먼저 예측하고, codebook 0 + speaker reference가 주어졌을 때 codebooks 1-7을 예측한다(depth-transformer pattern).
- 디코더: Mimi의 built-in decoder. external vocoder는 필요 없다.
- 학습: text-to-codec LM을 학습하고 Mimi는 freeze한다.
