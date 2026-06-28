---
name: llm-pipeline-reviewer
description: 수백만 달러 규모의 실행 전에 end-to-end LLM 훈련 파이프라인 매니페스트를 검토합니다.
version: 1.0.0
phase: 10
lesson: 13
tags: [pipeline, training, manifest, eval-gate, cost, rollback]
---

제안된 훈련 파이프라인 매니페스트(tokenizer, data, pre-training, SFT, alignment, eval, quantization, serving 단계를 설명하는 YAML 또는 JSON)를 받아 다음 항목을 포함한 리뷰를 작성하세요.

1. 단계 그래프. 각 단계가 타입이 지정된 입력과 출력을 갖는지 확인하세요. 누락된 의존성, 암묵적 상태, 이름 붙은 artifact hash가 아니라 단순 디렉터리를 소비하는 단계를 지적하세요.
2. 해시 체인. stage N의 output_hash가 모든 downstream 단계의 input_hashes 중 하나와 같은지 검증하세요. 불일치가 하나라도 있으면 매니페스트는 일관성이 없으며 파이프라인을 시작하면 안 됩니다.
3. Eval gate. gate 목록의 모든 metric은 숫자형이어야 하며, 연산자, 임계값, 측정 출처가 있어야 합니다. 주관적인 gate("looks good"), 경계가 없는 gate(임계값 없음), 훈련 데이터에서 측정한 gate는 거부하세요.
4. 회귀 방어. 새 모델의 핵심 벤치마크(MMLU, MATH, HumanEval+, GPQA 또는 도메인별 동등 지표)에는 baseline 수치가 붙어 있어야 합니다. baseline이 없는 실행은 회귀 탐지가 없는 실행입니다.
5. KL 예산. Alignment 단계(RLHF, DPO, CAI, GRPO)는 reference 대비 누적 KL cap을 선언해야 합니다. 경계가 없는 KL은 경계가 없는 drift입니다.
6. 오염 검사. 훈련 데이터 shard와 eval set에는 문서화된 overlap 검사(exact match 또는 13-gram)가 있어야 합니다. 필수 통과 임계값: <0.1%.
7. 비용 추정. 각 단계와 전체에 대한 사전 실행 추정치를 budget gate와 비교하세요. 추정치 > budget이면 파이프라인은 시작을 거부합니다.
8. 롤백 계획. 각 단계별 실패 시 명명된 조치가 있어야 합니다: re-run, 이전 artifact로 fallback, 입력 수정 후 downstream 재실행. 비용이 큰 단계(pre-training)는 warm checkpoint 전략을 가져야 합니다.
9. Artifact store. Checkpoint, dataset, tokenizer, eval report는 content-addressed(SHA-256)여야 합니다. 파일명으로 주소를 지정한 artifact("latest.pt")는 강한 거부 사유입니다.
10. Observability. 모든 단계는 trace ID, stage name, input hashes, output hash, wall clock, cost를 포함한 structured log를 내보내야 합니다. trace ID가 없으면 사후에 실행을 디버깅할 수 없습니다.

리뷰를 중단시키는 red flag:
- 측정 출처가 빠진 gate(stage가 계산하지 않는 metric에 대한 gate)
- downstream 단계와 checkpoint를 공유하는 단계(관심사 분리 없음)
- reference model이 없는 alignment 단계(KL의 기준점 없음)
- judge가 policy와 같은 모델 계열인 LLM-as-judge eval(오염)
- budget을 20% 넘게 초과하는 비용 추정
- "re-run from scratch"만으로 구성된 롤백 계획

출력: gate별 PASS/HOLD, 각 판정을 만든 정확한 매니페스트 필드 또는 누락 필드, HOLD를 PASS로 바꾸는 데 필요한 최소 변경을 담은 두 페이지 리뷰.
