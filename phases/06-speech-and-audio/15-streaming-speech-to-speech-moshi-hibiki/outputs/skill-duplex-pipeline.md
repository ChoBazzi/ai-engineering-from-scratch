---
name: duplex-pipeline
description: 음성 에이전트 워크로드에 맞는 full-duplex(Moshi)와 pipeline(VAD + STT + LLM + TTS) 아키텍처 중 하나를 고른다.
version: 1.0.0
phase: 6
lesson: 15
tags: [moshi, hibiki, full-duplex, voice-agent, streaming]
---

워크로드(지연 시간 목표, 도구 호출 필요성, 언어 범위, 하드웨어 예산, cloud vs edge)가 주어지면 다음을 출력하라.

1. 아키텍처. Full-duplex(Moshi / GPT-4o Realtime / Gemini Live) vs pipeline(LiveKit + STT + LLM + TTS, Lesson 12). 한 문장 이유.
2. 모델. Moshi · Hibiki · Hibiki-Zero · Sesame CSM · GPT-4o Realtime · Gemini 2.5 Live · traditional pipeline. 이유.
3. 규모. 세션당 GPU 비용(Moshi는 slot을 점유), 최대 동시 세션, cold-start 영향.
4. 도구 호출 경로. 필요하다면 hybrid pipeline(duplex + 도구 호출용 외부 LLM) 또는 pure pipeline. trade-off 설명.
5. 언어 범위. Full-duplex 모델은 언어 지원이 좁다. pipeline은 LLM의 다국어 능력을 물려받는다.

도구 호출/retrieval이 필요한 엔터프라이즈 에이전트에 full-duplex-only 아키텍처를 쓰는 것은 거부하라. Moshi는 대화 모델이지 에이전트 프레임워크가 아니다. 250ms 미만 대화형 에이전트에 pipeline-only를 쓰는 것도 거부하라. 단계별 지연이 누적된다. GPU 하나에서 동시 세션 &gt; 4개에 Moshi를 쓰는 것도 거부하라. contention에 걸린다.

예시 입력: "언어 학습용 voice companion — 회화 유창성 연습. 영어 + 프랑스어. &lt; 250 ms responsiveness. 일간 활성 사용자 1만 명."

예시 출력:
- 아키텍처: full-duplex(Moshi). 250ms 미만 지연 요구사항 + 회화 유창성 연습은 Moshi의 강점에 맞는다.
- 모델: Moshi. EN + FR 모두 잘 지원된다. CC-BY 4.0 license.
- 규모: 동시 세션 4-6개당 L4 GPU 하나 → 10% concurrency인 10k DAU의 peak에는 약 1500 GPUs. 조용한 경로에는 Kyutai Pocket TTS + local Whisper를 쓰는 on-device light mode를 계획한다.
- 도구 호출: 최소화. "문법 힌트 보여주기"와 "이 문구 번역하기"는 작은 LLM sidecar로 라우팅할 수 있다. 대부분의 상호작용은 Moshi가 강한 open-ended dialogue다.
- 언어 범위: EN + FR(native). ES / DE / JP는 Hibiki-Zero adaptation으로 지원(새 언어마다 1000h 오디오 필요).
