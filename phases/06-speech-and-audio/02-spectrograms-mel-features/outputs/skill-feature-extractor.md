---
name: feature-extractor
description: 다운스트림 오디오 모델에 맞는 feature type, mel count, frame/hop, normalization을 고릅니다.
version: 1.0.0
phase: 6
lesson: 02
tags: [audio, features, spectrogram, mel]
---

대상 모델(ASR / TTS / classifier / speaker / music)과 입력 오디오(sample rate, domain)가 주어지면 다음을 출력하세요.

1. 특징 유형. Log-mel, mel, MFCC, raw waveform, 또는 discrete codec(EnCodec, SoundStream). 한 문장 이유.
2. Mel 개수와 주파수 범위. `n_mels`, `fmin`, `fmax`. Domain(speech vs music)과 model target에 연결된 이유.
3. 프레임과 hop. `frame_len`, `hop_len`, window type. 필요한 temporal resolution에 연결된 이유.
4. 정규화. Per-utterance mean/var, global stats, 또는 fixed reference를 쓰는 dB; featurization 전 또는 후.
5. 검증 snippet. 1초 reference clip에서 결과 shape, min/max, mean/std를 출력하고 학습 설정과 일치한다고 assert하는 Python.

Frame/hop/mel count가 대상 모델의 공개 학습 config와 다른 feature pipeline은 배포를 거부하세요. Whisper 또는 Parakeet에 MFCC 기반 설정을 쓰면 잘못된 것으로 표시하세요. 이 모델들은 log-mel을 소비합니다. Normalization assertion이 없는 feature extractor도 표시하세요.
