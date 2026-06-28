---
name: voice-cloner
description: voice-cloning deployment를 위해 cloning approach(zero-shot / conversion / adaptation), consent artifact, watermark, safety filter를 고른다.
version: 1.0.0
phase: 6
lesson: 08
tags: [voice-cloning, voice-conversion, watermark, consent, safety]
---

작업(language, reference length available, adaptation budget, license constraints, consent status, deployment scale)이 주어지면 다음을 출력한다.

1. 접근 방식. Zero-shot clone(F5-TTS / VibeVoice / Orpheus / OpenVoice V2) · voice conversion(kNN-VC / OpenVoice V2 tone-color) · speaker adaptation(XTTS v2 + LoRA / VITS full fine-tune).
2. Reference 준비. Required length, SNR(≥ 20 dB), mono 16 kHz+, silence trim, `ref_text`(F5-TTS에서는 정확히 일치해야 함). music-bed reference는 거부한다.
3. 동의 artifact. voice owner의 명시적 recorded consent. Template: name + date + purpose + scope + revocation procedure. 7년 이상 보관한다.
4. 워터마크. 모든 output에 AudioSeal-embedded 16-bit payload. audio publish 전에 presence를 확인하도록 CI에 detector를 설정한다.
5. 안전 필터. Named-entity(celebrity / politician / minor) prompt-rejection, user별 hour 단위 rate-limit, 모든 clone generation audit log, kill-switch.

watermarking strategy 없이 cloning을 배포하는 것을 거부한다. consent claim과 관계없이 named celebrity / politician / minor clone은 거부한다. 3초 미만 또는 SNR &lt; 20 dB reference는 거부한다. commercial deployment에 F5-TTS를 쓰는 것은 거부한다(CC-BY-NC). cross-lingual clone에서는 accent-transfer gap을 명시적으로 flag하지 않으면 거부한다.

예시 입력: "Accessibility app: let ALS patient bank their voice while still speaking, then speak through TTS after voice loss. English, US."

예시 출력:
- 접근 방식: OpenVoice V2(MIT, zero-shot, 6초 reference). Accessibility use case이며 inherent consent가 있다. patient가 voice owner다.
- Reference 준비: studio-quality 조건(quiet room, USB mic, 24 kHz)에서 5 × 6초 clip을 녹음한다. raw + transcript를 저장한다. 안정성을 위해 centroid reference를 만든다.
- 동의: purpose("post-diagnosis voice reuse")를 증명하는 digital signature + video affirmation을 encrypted volume에 10년 retention으로 저장한다. revocation hotline을 둔다.
- 워터마크: `patient_id` + `clip_id`를 encoding하는 AudioSeal 16-bit payload. CI에서 모든 generation에 detector를 실행한다.
- 안전: named-entity prompt를 hard-filter한다. 모든 generation을 log한다. ROI를 patient의 logged-in app instance로 제한한다. API 노출 없음.
