---
name: alm-picker
description: audio-understanding task에 맞는 audio-language model, benchmark subset, output modality(text vs speech), guardrail을 선택한다.
version: 1.0.0
phase: 6
lesson: 10
tags: [alm, lalm, qwen-omni, audio-flamingo, gemini-audio, mmau]
---

task(speech / sound / music / multi-audio / long-audio, output modality, latency, license)가 주어지면 다음을 출력하라.

1. 모델. Qwen2.5-Omni-7B · Qwen3-Omni · SALMONN · Audio Flamingo 3 · AF-Next · LTU · GAMA · Gemini 2.5 Pro(API) · GPT-4o Audio(API). 한 문장 이유.
2. 검증할 benchmark subset. MMAU-Pro speech / sound / music / multi-audio · LongAudioBench · AudioCaps · ClothoAQA. 사용자 task와 맞는 축을 선택한다.
3. 출력 방식. Text-only · text + speech(Qwen-Omni, GPT-4o Audio). 필요하다면 추가 speech decoder budget을 잡는다.
4. 가드레일. 모델의 multi-audio score가 &lt; 30%(near-random)이면 multi-audio comparison이 필요한 prompt를 거부한다. &gt; 10분 input은 LALM 전에 diarize한다.
5. 에스컬레이션. 이 task가 언제 specialized model로 fallback해야 하는지 정한다. transcription에는 Whisper, classification에는 BEATs, diarization에는 pyannote. LALM은 각 영역의 최고 모델이 아니다.

MMAU-Pro multi-audio subset에서 모델 점수가 &gt; 40%임을 검증하지 않고 multi-audio comparison task를 출시하지 마라. upstream diarization 없이 long-audio(&gt; 10 min)를 거부하라. vendor-reported number를 독립 재검증 없이 사용하는 deployment는 flag하라.

예시 입력: "Compliance audit: transcribe 10-minute bank-call recordings + detect if the agent read the mandatory disclosure."

예시 출력:
- 모델: transcription에는 Whisper-large-v3-turbo + transcript 위 disclosure-check QA에는 Gemini 2.5 Pro(via API). raw audio에 LALM을 직접 쓰고 싶겠지만 long-audio LALM accuracy는 10분 이후 떨어진다.
- 벤치마크 subset: MMAU-Pro speech subset(Gemini 2.5 Pro = 73.4%). speech-reasoning axis를 다룬다. 자체 50-call gold set에서도 spot-check한다.
- 출력 방식: text-only. audit report에는 speech output이 필요 없다.
- 가드레일: 먼저 pyannote 3.1로 diarize한다. per-speaker segment를 따로 보낸다. call별 confidence score를 log한다.
- 에스컬레이션: call이 disclosure check에 실패하면 autonomous flag 대신 human reviewer로 route한다.
