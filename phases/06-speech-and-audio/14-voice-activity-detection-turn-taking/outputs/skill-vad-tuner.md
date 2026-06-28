---
name: vad-tuner
description: 음성 에이전트에 맞는 VAD 모델, 임계값, 침묵 행오버, 프리롤, 턴 감지 전략을 고른다.
version: 1.0.0
phase: 6
lesson: 14
tags: [vad, silero, cobra, turn-detection, flush-trick]
---

워크로드(consumer / call-center / edge / accessibility, 잡음 프로필, 언어 조합, 지연 시간)가 주어지면 다음을 출력하라.

1. VAD. Silero VAD(기본값) · Cobra(상용 정확도) · pyannote segmentation(화자 분리 수준) · WebRTC VAD(legacy / tiny). 한 문장 이유.
2. 매개변수. Threshold(0.3-0.5), min speech(200-300 ms), silence hangover(400-800 ms), pre-roll(250-500 ms).
3. 의미 기반 턴 감지. 활성화(LiveKit turn-detector 또는 custom MLP) 여부. 예상 사용자 발화 패턴과 연결된 이유.
4. Flush trick. 활성화(STT가 지원하는 경우 — Kyutai / Deepgram) 여부. 예상 지연 시간 절감.
5. Guard. 최소 길이보다 짧은 음성 거부, pre-roll 항상 유지, 사용자별 silence-hangover override 상한 설정, VAD 서비스가 down이면 fail-open(모든 것을 음성으로 취급).

프로덕션에서 에너지만 쓰는 VAD는 거부하라. 잡음에 너무 취약하다. silence-hangover 0은 거부하라. 사용자를 방해한다. 전용 Silero를 사용할 수 있는데 Whisper 기반 VAD를 쓰는 것도 거부하라(더 느리고 덜 정확하다).

예시 입력: "항공권 재예약용 콜센터 IVR. 시끄러운 배경(공항). 영어 + 스페인어. &lt; 500 ms turn detection."

예시 출력:
- VAD: 잡음 내성 이점 때문에 Cobra(commercial). 비용이 지나치면 Silero로 fallback.
- 매개변수: threshold 0.4(공항 noise floor가 높음), min speech 300 ms, silence hangover 600 ms(사용자가 IVR 중 항공편 번호를 읽느라 자주 멈춤), pre-roll 400 ms.
- 의미 기반 턴: LiveKit turn-detector 활성화. 문장 중간 멈춤이 흔함("내 항공편을 바꾸고 싶어요... 내일로요").
- Flush trick: Deepgram streaming에서 활성화. 예상 절감: turn-end latency 400 ms → 150 ms.
- Guard: Cobra/Deepgram에 연결할 수 없으면 fail-open. 튜닝을 위해 모든 VAD-fire event를 audit log에 남김.
