---
name: speaker-verifier
description: model choice, enrollment protocol, threshold tuning을 포함해 speaker verification 또는 diarization pipeline을 설계한다.
version: 1.0.0
phase: 6
lesson: 06
tags: [audio, speaker, verification, diarization]
---

목표(verification vs identification vs diarization, domain, channel, threat model)와 데이터(threshold tuning을 위한 hours, number of speakers, enrollment clip budget)가 주어지면 다음을 출력한다.

1. 임베더. ECAPA-TDNN / WavLM-SV / ReDimNet / x-vector. 이유.
2. 등록 프로토콜. clip 수, 최소 duration, noise gate, channel match.
3. 점수화. Cosine / PLDA, AS-norm 사용 여부, cohort size.
4. 임계값. 목표 FAR(fraud risk) 또는 EER, tuning set size.
5. Spoof 방어. Anti-spoof model(AASIST, RawNet2), liveness challenge, replay detection 중 하나.

anti-spoof front-end가 없는 fraud-grade deployment는 거부한다. evaluation set, channel, clip length distribution을 보고하지 않는 EER 공개는 거부한다. domain별 재튜닝 없이 고정된 cosine threshold를 쓰면 flag한다.
