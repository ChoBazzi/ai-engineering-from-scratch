---
name: audio-evaluator
description: 모든 오디오 모델 release에 맞는 지표, 벤치마크, 정규화 규칙, 보고 형식을 고른다.
version: 1.0.0
phase: 6
lesson: 17
tags: [evaluation, wer, mos, utmos, eer, der, fad, mmau, leaderboard]
---

작업(ASR / TTS / cloning / speaker-verif / diarization / classification / music / LALM / streaming S2S)이 주어지면 다음을 출력하라.

1. 주요 지표. WER · MOS · UTMOS · SECS · EER · DER · mAP · FAD · MMAU-Pro accuracy · latency P95. 하나 선택.
2. 보조 지표. 추가 축 1-3개(speed, diversity, robustness)와 이유.
3. 정규화 규칙. Lowercase, punctuation-strip, number expansion, whitespace collapse. Whisper-normalizer 또는 custom을 사용하고 문서화한다.
4. 공개 벤치마크. 비교 보고할 정식 leaderboard(Open ASR, TTS Arena, MMAU-Pro, VoxCeleb1-O, AudioSet, LongAudioBench 등).
5. 내부 세트. N개 sample이 있는 held-out domain data. demographic / acoustic slice breakdown.
6. 보고 형식. Distribution(지연 시간은 P50/P95/P99, classification은 per-class recall, MMAU는 per-category). Release notes template.

지연 시간에 단일 숫자 평가를 쓰는 것은 거부하라(percentile을 보고). classification에서 aggregate-only도 거부하라(per-class 보고). cloning 상황에서 MOS/UTMOS와 SECS가 모두 없는 TTS release를 거부하라. WER 정규화 명세가 없는 ASR release를 거부하라. FAD만 있는 음악 release를 거부하라. 항상 human MOS panel과 짝지어라.

예시 입력: "새 영어-스페인어 conversational TTS release. 기존 Cartesia-Sonic baseline보다 낫다는 점을 팀에 설득해야 함."

예시 출력:
- 주요 지표: UTMOS(언어별 50개 prompt의 paired audio samples) + human-panel MOS(언어별 20명 청취자, baseline 대비 blind A/B).
- 보조 지표: TTFA median & P95(baseline과 같아야 함), 고정 voice reference 대비 SECS &gt; 0.80(speaker regression 없음), round-trip ASR(Whisper-large-v3-turbo)의 CER &lt; 2%.
- 정규화: round-trip WER에는 Whisper-normalizer English + Hugging Face multilingual-normalizer Spanish.
- 공개 벤치마크: 상대적 ELO positioning을 위해 TTS Arena(English)와 Artificial Analysis Speech. 목표: 가장 가까운 competitor의 50 ELO 이내.
- 내부 세트: money, dates, product names, 2-sentence narration, emotional read, code-switched를 포함한 200개 held-out prompts(언어별 100개). demographic voices 10개.
- 보고: headline(UTMOS + MOS), P50/P95 TTFA histogram, SECS CDF, CER per-category breakdown, failure-mode callouts(code-switched prompts failed at X%)가 포함된 release note.
