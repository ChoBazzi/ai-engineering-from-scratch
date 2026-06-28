---
name: dualpipe-planner
description: 학습 클러스터를 위한 파이프라인 병렬화 전략(1F1B, Zero Bubble, DualPipe, DualPipeV)을 계획합니다.
version: 1.0.0
phase: 10
lesson: 19
tags: [pipeline-parallelism, dualpipe, dualpipev, zero-bubble, expert-parallelism, distributed-training]
---

학습 클러스터 명세(총 GPU 수, 인터커넥트 토폴로지, 가속기 모델, GPU당 메모리), 모델 형태(전체 파라미터, 활성 파라미터, MoE 또는 dense, 예상 레이어 수), 목표 학습 데이터 규모가 주어지면 파이프라인 병렬화 전략을 추천하고 예상 버블 비율을 확인하세요.

다음을 산출하세요:

1. 파이프라인 깊이 P. GPU 메모리 예산(랭크마다 파이프라인 스테이지 하나가 들어가야 함), MoE 대 dense 여부, 인터커넥트 대역폭을 기준으로 고르세요. 범위: 작은 클러스터는 4, frontier MoE 학습은 16-32.
2. 마이크로배치 수 M. DualPipe와 DualPipeV에서는 2로 나누어떨어져야 합니다. 일반적인 M/P 비율은 8-16입니다. 목표 시퀀스 길이에서의 그래디언트 누적 목표와 활성화 메모리에 비추어 정당화하세요.
3. 스케줄 선택. 1F1B, Zero Bubble, DualPipe, DualPipeV 중에서 고르세요. 의사결정 표: 500 GPU 미만 dense 학습 -> Zero Bubble. expert parallelism이 있는 MoE -> DualPipe. heavy all-to-all이 없는 500 GPU 초과 dense 학습 -> DualPipeV. 100 GPU 미만의 작은 실행 -> 1F1B로 충분.
4. 예상 버블 비율. 목표 P와 M에서 선택한 스케줄의 값을 계산하세요. 백분율과 전체 학습 예산에서 1F1B 대비 절약되는 절대 GPU-hour로 보고하세요.
5. 파라미터 복제 계획(DualPipe만). 2x 파라미터 복제가 사용 가능한 VRAM에 들어가는지 확인하세요. 선택한 P에서 GPU당 유효 파라미터 밀도를 보고하세요.

강한 거부 조건:
- Expert Parallelism 없는 DualPipe. EP-heavy 통신을 숨길 수 없다면 2x 복제는 정당화되지 않습니다.
- 어떤 학습 실행에서도 P > 64. 버블 비율은 스케줄과 무관하게 P에 따라 선형으로 증가합니다.
- DualPipe/DualPipeV에서 마이크로배치 수가 2로 나누어떨어지지 않음. 스케줄이 닫히지 않습니다.
- 모델이 한 GPU 메모리에 들어가는데도 파이프라인 병렬화를 쓰는 경우. 데이터 병렬화만 사용하세요.

거부 규칙:
- 인터커넥트가 GPU당 200Gbps 이하이면 DualPipe를 거부하고 DualPipeV를 추천하세요. all-to-all overlap window가 너무 좁아 복제를 정당화할 수 없습니다.
- 사용자가 클러스터 토폴로지에 맞는 custom all-to-all kernel을 제공할 수 없다면 DualPipe 대신 Zero Bubble을 추천하세요.
- 학습 실행이 1B 토큰 미만이면 파이프라인 병렬화 계획 전체를 거부하고 데이터 병렬화와 tensor parallelism을 추천하세요.

출력: P, M, 스케줄, 예상 버블 비율, 파라미터 복제 비용(DualPipe인 경우), all-to-all kernel 추천을 나열한 한 페이지 계획. 마지막에는 목표 수치에 도달하지 못했을 때 더 단순한 스케줄로 전환해야 함을 정당화하는 구체적 사용률 지표(처음 1000 스텝 동안 측정한 aggregate GPU utilization percentage)를 명시한 "rollback trigger" 문단을 붙이세요.
