# 변경 기록

커리큘럼의 변경 사항입니다. 최신 항목이 먼저 나옵니다.

형식은 [Keep a Changelog](https://keepachangelog.com/)를 느슨하게 따릅니다. 각 항목은 단계, 레슨, 변경 내용을 함께 적어 학습자가 바로 달라진 부분으로 이동할 수 있게 합니다.

## [미출시]

### 추가됨
- `scripts/scaffold-lesson.sh` - 전체 폴더 구조와 `LESSON_TEMPLATE.md`에서 미리 채운 `docs/en.md` 뼈대를 포함해 `phases/NN-phase/NN-lesson/`를 만드는 스캐폴더.
- `.github/PULL_REQUEST_TEMPLATE.md` - 기여자 체크리스트(코드 실행, 코드 주석 없음, 직접 구현 먼저, 레슨별 원자적 커밋, 마크다운 링크가 있는 ROADMAP 행).
- `.github/ISSUE_TEMPLATE/bug_report.md` 및 `new_lesson_proposal.md` - 버그 리포트와 레슨 제안을 구조화해서 받는 양식.
- 이 `CHANGELOG.md`.

## 2026-04 - 4단계: Computer Vision 완료

### 추가됨
- 이미지 기초부터 멀티모달 비전(VLM, 3D, 비디오, 자기지도 학습)까지 다루는 4단계 레슨 28개 전체.
- 웹사이트가 표시할 수 있도록 `ROADMAP.md`의 4단계 행을 레슨 폴더로 향하는 마크다운 링크로 연결.

### 수정됨
- 15개 이상 레슨에 걸친 4단계 정밀 검토:
  - `phase-4/02`: shape calculator가 adaptive pool, flatten, linear의 RF/stride 처리를 명시.
  - `phase-4/03`: backbone selector 설명이 다루는 모든 계열을 나열하고 OCR, 의료, 산업용 head 가이드를 추가.
  - `phase-4/04`: 분류 진단이 실패 모드별 정량 임계값을 사용하고, 정의되지 않은 metric은 `n/a`로 표시하며, 클래스가 3개 미만일 때의 guard를 추가.
  - `phase-4/06`: detection metric reader가 `mAP@0.5`가 아니라 `AP@0.5`를 사용하고, 클래스별 recall을 선택 사항으로 명시하며, anchor designer가 stride truncation과 level당 single-anchor 경로를 명확히 함.
  - `phase-4/10`: sampler picker가 `unet_forward_ms`를 입력으로 선언하고, ControlNet guard를 rule 0으로 승격.
  - `phase-4/14`: ViT inspector를 거부 규칙에 맞춤. 포팅 시도는 권장하지 않고 감사 대상으로 처리.
  - `phase-4/24`: open-vocab stack picker에 명시적 규칙 우선순위와 license-filter 의미를 추가하고, concept designer가 step-5/rule-80 충돌을 해소.
  - `phase-4/25`: VLM 문서의 `_merge`가 placeholder 불일치 시 설명적인 `ValueError`를 발생시키고, CMER가 내부에서 정규화.
  - `phase-4/27`: `synthetic_frames`가 GT box를 frame H/W에 맞게 clip.
  - `phase-4/28`: `rope_3d`가 dim split을 검증하고, DiT block 예제에서 사용하지 않는 `F` import 제거.

## 2026년 1분기 및 이전

### 추가됨
- 0단계(Setup & Tooling): 레슨 12개 전체.
- 1단계(Math Foundations): 레슨 22개 전체.
- 2단계(ML Fundamentals): 레슨 18개 전체.
- 3단계(Deep Learning Core): 퍼셉트론, backprop, optimizer까지의 핵심 레슨.
- 내장 Claude Code skill: `find-your-level`(배치 퀴즈) 및 `check-understanding`(단계별 퀴즈).
- `aiengineeringfromscratch.com` 웹사이트: 카탈로그, 레슨별 페이지, 로드맵, 277개 용어 glossary.
- 전체 20단계의 초기 스캐폴딩(`phases/00-*`부터 `phases/19-*`까지).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Unreleased]: https://github.com/rohitg00/ai-engineering-from-scratch/compare/HEAD...HEAD
