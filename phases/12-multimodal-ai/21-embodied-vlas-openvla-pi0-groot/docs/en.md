# 체화형 VLA: RT-2, OpenVLA, π0, GR00T

> 모델이 웹사이트의 레시피를 읽고 주방 로봇에서 실행한 첫 사례는 RT-2(Google DeepMind, 2023년 7월)였다. RT-2는 행동을 텍스트 토큰으로 이산화하고, 웹 데이터와 로봇 행동 데이터를 함께 사용해 VLM을 공동 미세 조정했으며, 웹 규모의 vision-language 지식이 로봇 제어로 전이된다는 점을 입증했다. OpenVLA(2024년 6월)는 공개 7B 기준 모델을 제공했다. Physical Intelligence의 π0 시리즈(2024-2025)는 flow-matching 행동 전문가를 추가했다. NVIDIA의 GR00T N1(2025년 3월)은 휴머노이드 로봇을 위한 이중 시스템(System 1 / System 2) 제어를 대규모로 제공했다. VLA 기본형, 즉 vision-language-action은 보고, 읽고, 행동하는 단일 모델이며, 이 단계의 이해 모델과 Phase 15의 자율 시스템을 잇는 다리다.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 minutes

## 학습 목표

- 행동 토큰화(action tokenization)를 설명한다: 이산 bin 인코딩(RT-2), FAST 효율적 행동 토큰, 연속 flow-matching 행동(π0).
- 웹 데이터와 로봇 데이터를 함께 공동 미세 조정하면 왜 새로운 작업으로 일반 지식 전이가 보존되는지 설명한다.
- 같은 로봇 작업에서 OpenVLA(공개 7B Llama+VLM), π0(flow-matching), GR00T N1(이중 시스템)을 비교한다.
- Open X-Embodiment 데이터셋과 RT-X 학습 말뭉치로서의 역할을 말한다.

## 문제

자연어 지시로 집안일을 하는 로봇은 1970년대부터 연구 목표였다. 2020년대의 답은 vision-language-action(VLA) 모델이다. VQA에 쓰는 것과 같은 VLM 아키텍처를 사용하지만, 출력은 텍스트가 아니라 행동(관절 토크, 말단 장치 자세, 이산 명령)이다.

VLA에 특화된 어려움은 다음과 같다.

1. 행동 공간은 연속적이고(관절 각도, 힘), 고차원이다(7-DOF 팔 + 3-DOF 그리퍼 = 30 Hz에서 10차원).
2. 로봇 전용 학습 데이터는 부족하다. Open X-Embodiment에는 약 100만 개 trajectory가 있지만, 웹 텍스트-이미지는 50억 개 이상이다.
3. 제어 주파수가 중요하다. 30 Hz 제어 루프는 행동당 33ms 예산을 뜻한다.
4. 안전. 잘못된 행동은 하드웨어, 사람, 재산을 손상시킨다.

## 개념

### 행동 토큰화(RT-2)

RT-2의 핵심 아이디어는 각 관절 목표값을 양자화된 텍스트 토큰으로 표현하는 것이다. 정규화된 [-1, 1] 범위를 256개 bin으로 이산화하고, 각 bin을 vocabulary ID에 매핑한다. 10-DOF 행동은 각 제어 단계마다 10개 토큰이 된다.

PaLM-X VLM을 다음 혼합 데이터로 공동 미세 조정한다.

- 웹 이미지-텍스트 쌍(캡션 생성, VQA).
- 로봇 시연, 행동은 토큰으로 표현.

모델은 "빨간 큐브를 집어라"(언어) → 이미지(비전) → 10토큰 행동 시퀀스(이산화된 관절 목표값)를 본다. 웹 사전학습은 일반 지식 전이를 보존한다. 그래서 RT-2는 학습 데이터에 "빠르게 움직이는"이라는 표현이 없어도 "빠르게 움직이는 물체 쪽으로 이동하라"를 따를 수 있다.

RT-2 논문에서 추론은 3-5 Hz였고, VLM 자기회귀 디코딩이 병목이었다.

### OpenVLA - 공개 7B 기준 모델

OpenVLA(Kim et al., 2024년 6월)는 공개 가중치 RT-2에 해당한다. 7B Llama 백본, DINOv2 + SigLIP 이중 비전 인코더, 256개 bin 기반 행동 토큰화를 사용한다.

Open X-Embodiment(22개 로봇의 97만 trajectory)로 학습되었다. 새로운 로봇에 적응하기 위한 LoRA 미세 조정 지원을 함께 제공한다.

추론 속도는 양자화를 사용한 A100에서 4-5 Hz다. 느린 조작에는 충분하지만 고주파 제어에는 부족하다.

### FAST 토크나이저 - 더 빠른 행동 디코딩

Pertsch et al.(2024)은 이산 bin 토큰화가 비효율적임을 보였다. 대부분의 행동이 bin 공간의 작은 영역에 몰리기 때문이다. FAST(Frequency-domain Action Sequence Tokenizer)는 DCT로 행동 시퀀스를 압축하고 계수를 양자화한다.

30단계 행동 trajectory가 300개의 이산 bin 토큰 대신 약 10개의 FAST 토큰이 된다. 품질 손실 없이 추론이 3-5배 빨라진다.

### π0와 flow-matching 행동

Physical Intelligence의 π0(Black et al., 2024년 10월)는 이산 행동 토큰을 flow-matching 행동 전문가로 대체한다.

- 작은 행동 transformer가 VLM의 hidden state를 읽고 rectified flow로 연속적인 50단계 행동 시퀀스를 출력한다.
- 행동 head는 flow-matching loss로 학습하고, VLM 사전학습은 그대로 둔다.
- 추론: 전체 행동 시퀀스를 약 5번의 denoising 단계로 내보내며, 실질적으로 50 Hz 제어가 가능하다.

π0의 주장: 넓은 조작 작업 집합에서 OpenVLA와 Octo를 능가한다. 연속 행동 공식화는 이산화가 망가뜨리는 부드러움을 보존한다.

π0.5와 π0-FAST는 점진적 업그레이드다. π0-FAST는 FAST 토큰화와 flow matching을 결합한다.

### GR00T N1 - 휴머노이드를 위한 이중 시스템

NVIDIA의 GR00T N1(2025년 3월)은 휴머노이드 로봇(30 DOF 초과, 전신)을 위해 만들어졌다.

- System 2: 장면과 지시를 읽고 약 1 Hz로 고수준 하위 목표를 만드는 대형 VLM.
- System 1: 하위 목표를 조건으로 50-100 Hz의 저수준 관절 명령을 생성하는 작은 action-head transformer.

이 분리는 카너먼의 빠른 사고와 느린 사고에 대응된다. System 2는 계획하고, System 1은 행동한다. 이점은 느린 VLM 크기의 계획이 빠른 제어를 막지 않고, System 1은 지연 시간을 위해 작게 유지된다는 점이다.

GR00T N1.7(2025년 말)은 데이터 스케일링을 개선한다. GR00T는 Omniverse의 sim-to-real 데이터로 미세 조정된다.

### Open X-Embodiment

학습 데이터다. RT-X(2023년 10월)는 22개 로봇에 걸친 100만 개 trajectory를 포함하는 22개 데이터셋을 모았다. Open X-Embodiment는 모두가 사용하는 말뭉치다.

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- 각 샘플: (로봇 상태, 카메라 뷰, 지시, 행동 시퀀스).
- 학습 위생: 행동 공간 통합, 관절 범위 정규화, 카메라 리사이즈.

OpenVLA와 π0는 Open X-Embodiment로 학습한다. 특정 로봇과의 domain gap은 작업별 시연 100-1000개에 대한 LoRA 미세 조정으로 줄인다.

### 공동 미세 조정 vs 로봇 전용

공동 미세 조정은 웹 VQA 데이터와 로봇 trajectory를 섞는다. 비율이 중요하다. VQA가 너무 많으면 모델이 행동을 잊고, 로봇 데이터가 너무 많으면 모델이 일반 지식을 잃는다.

RT-2의 비율은 약 1:1이다. OpenVLA는 웹 대 로봇이 약 0.5:1이다. π0도 비슷하다. 정확한 비율은 데이터셋 크기별로 조정할 hyperparameter다.

로봇 전용 학습은 작업 특화 모델을 만들지만, 분포 밖 지시에서는 실패한다. 공동 미세 조정은 "빨간 큐브를 집어라(시연에 있음)"와 "왼쪽에서 세 번째로 큰 물체를 집어라(새로운 표현)"의 차이를 만든다.

### 안전과 행동 한계

모든 프로덕션 VLA는 다음을 포함한다.

- 강한 관절 한계(스펙을 넘는 토크 불가).
- 속도 한계(soft clipping).
- 작업 공간 경계(말단 장치가 테이블을 벗어날 수 없음).
- 새로운 작업에 대한 human-in-the-loop 승인.

이것들은 VLA 바깥의 제어 계층 검사로 존재한다. VLA의 출력은 명령이 아니라 제안이다.

## 활용하기

`code/main.py`:

- 256-bin 행동 토큰화와 역토큰화를 구현한다.
- DCT + 양자화 기반 FAST 토크나이저를 스케치한다.
- 이산 bin, FAST, continuous-flow 사이에서 행동 단계당 토큰 수를 비교한다.
- RT-2 → OpenVLA → π0 → GR00T의 계보 요약을 출력한다.

## 산출물

이 lesson은 `outputs/skill-vla-action-format-picker.md`를 만든다. 로봇 작업(조작, 내비게이션, 휴머노이드 전신)이 주어지면 discrete-bin + RT-2, FAST + OpenVLA, flow-matching + π0, dual-system + GR00T 중 하나를 고른다.

## 연습 문제

1. 30 Hz 제어율의 10-DOF 팔이 있다. 256개 bin의 이산 bin 토큰화는 초당 몇 개 토큰을 내보내는가? 7B VLM이 따라갈 수 있는가?

2. FAST 토큰화는 30단계 trajectory를 약 10개 토큰으로 압축한다. trajectory에 고주파 움직임(예: 드럼 연주)이 있으면 사용자는 무엇을 잃는가?

3. π0의 flow-matching head는 약 5단계로 denoise한다. OpenVLA의 4-5 Hz 자기회귀 디코딩과 처리량을 비교하라.

4. GR00T의 System 1 / System 2 분리는 카너먼에 대응된다. 이족 보행에 도움이 될 수 있는 다른 분리(System 3?)를 제안하라.

5. 데이터셋 큐레이션에 관한 Open X-Embodiment Section 4를 읽어라. domain leakage를 막는 세 가지 큐레이션 규칙을 말하라.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | 이미지 + 지시를 받아 행동 명령을 출력하는 모델 |
| Action tokenization | "Discrete bins" | 연속 관절 목표값을 차원마다 256개 bin으로 양자화하고, 각 bin을 vocab ID로 표현 |
| FAST tokenizer | "Frequency action tokens" | DCT + 양자화로 30단계 trajectory를 약 10개 토큰으로 압축 |
| Co-fine-tune | "Mix web + robot" | 일반 지식을 보존하기 위해 웹 VQA 데이터와 로봇 시연을 함께 학습 |
| Flow-matching action head | "π0 continuous output" | rectified flow로 50단계 행동 시퀀스를 출력하는 작은 transformer |
| System 1 / System 2 | "Dual-system control" | 대형 VLM은 느리게 계획하고 작은 action head는 빠르게 행동하는 GR00T 패턴 |
| Open X-Embodiment | "RT-X dataset" | 100만 trajectory 규모의 교차 로봇 데이터셋이자 학습 말뭉치 |

## 더 읽을거리

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
