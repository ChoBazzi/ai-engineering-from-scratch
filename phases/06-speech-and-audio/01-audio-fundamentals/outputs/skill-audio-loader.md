---
name: audio-loader
description: 원시 오디오 파일이 대상 모델의 기대와 맞는지 검증하고 안전하게 리샘플링합니다.
version: 1.0.0
phase: 6
lesson: 01
tags: [audio, speech, preprocessing]
---

오디오 파일(path, channels, sample rate, bit depth, codec)과 대상 모델(필수 sample rate와 channel count가 있는 ASR / TTS / classifier)이 주어지면 다음을 출력하세요.

1. 불일치. 파일이 대상과 맞지 않는 모든 차원(sr, channels, duration floor, clipping check)을 나열합니다.
2. 리샘플링 계획. Source sr, target sr, resampling library(`torchaudio.transforms.Resample` 또는 `librosa.resample`), anti-aliasing filter type.
3. 채널 계획. Mono fold 전략(mean vs left-only), 또는 모델이 지원할 때 multichannel pass-through.
4. 정규화. Peak vs RMS normalization, dBFS target, clipping guard.
5. 검증 snippet. 파일을 로드하고 transform을 실행한 뒤 최종 배열이 `(target_sr, dtype, channel_count, range)`와 일치한다고 assert하는 Python.

Anti-aliasing filter 없는 다운샘플링은 거부하세요. Reconstruction filter 없이 2x를 넘는 업샘플링도 거부하세요. Clipping peak가 ±0.999를 넘거나 DC offset이 ±0.01을 넘는 입력 파일은 표시하세요.
