# 비디오-언어 모델: 시간 토큰과 그라운딩

> Video는 사진 더미가 아니다. 5초 clip에는 causal ordering, action verb, event timing이 있으며, image model은 이를 표현할 수 없다. Video-LLaMA(Zhang et al., 2023년 6월)는 audio-visual grounding을 갖춘 첫 오픈 video-LLM을 내놓았다. VideoChat과 Video-LLaVA가 이 패턴을 확장했다. 2025년에는 Qwen2.5-VL의 TMRoPE가 frontier proprietary model과의 격차를 좁혔다. 각 시스템은 temporal token 문제를 서로 다르게 풀었다. clip마다 Q-former, frame마다 concat-pool, token마다 TMRoPE. 이 lesson은 패턴을 읽고, uniform-vs-dynamic frame sampler를 만들며, temporal grounding task에서 평가한다.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## 학습 목표

- vision encoder와 별개로 temporal positional encoding이 video VLM 성능을 바꾸는 이유를 설명한다.
- uniform, dynamic-FPS, event-driven frame sampling을 tokens-per-second와 grounding accuracy trade-off로 비교한다.
- Q-former-per-clip(Video-LLaMA), pooled-per-frame(Video-LLaVA), M-RoPE-per-token(Qwen2.5-VL) 설계를 설명한다.
- 네 가지 video benchmark를 말한다: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## 문제

30 FPS의 1분 video는 1800 frame이다. frame당 196 visual token(ViT-B at 224)이라면 352k token이다. 이는 2024년대 LLM context보다 크다.

세 가지 축소 전략이 있다.

1. Frame을 subsample한다(content에 따라 1-8 FPS).
2. 각 frame의 patch token을 강하게 pool한다(3x3 또는 4x4 bilinear pool).
3. 16-frame clip을 받아 64 token을 출력하는 Q-former로 압축한다.

각 trade-off는 다르다. Subsampling은 temporal detail을 잃는다. Pooling은 spatial detail을 잃는다. Q-former는 둘을 조금씩 잃지만 token을 절약한다.

Temporal position encoding은 또 다른 축이다. 모델은 frame 5가 frame 6보다 먼저 왔다는 것을 어떻게 아는가? 선택지는 simple 1D temporal RoPE(Video-LLaMA), learned temporal embeddings(Video-LLaVA), TMRoPE(Qwen2.5-VL, full 3D)가 있다.

## 개념

### Video-LLaMA: clip마다 Q-former + audio branch

Video-LLaMA(2023)는 첫 오픈 video-LLM이었다. Architecture:

- 2 FPS의 16-frame clip(즉 8초).
- Per-frame ViT feature -> 모든 16 frame에 cross-attend하는 Video Q-former -> 32 learned query -> LLM.
- 병렬 audio branch: waveform -> ImageBind audio encoder -> Audio Q-former -> 32 query -> LLM.

강점은 audio-visual joint reasoning이다. 약점은 고정 clip length와 임의 time grounding의 부재다.

### VideoChat과 Video-LLaVA

VideoChat은 Video-LLaMA 아이디어를 유지했지만 audio를 제거하고 단순화했다. Video-LLaVA(Lin et al., 2023)는 이미지와 video frame 모두에서 단일 visual encoder를 학습했다("alignment before projection"). 이로써 통합 representation을 얻었다. 둘 다 frozen-CLIP-encoder + MLP + LLM 구조다.

둘 다 long video를 처리하지 못한다. 둘 다 8-16 frame system이다.

### Qwen2.5-VL과 TMRoPE

Qwen2.5-VL은 TMRoPE(Temporal-Modality Rotary Position Embedding)를 도입했다. 각 patch token은 (t, h, w) position을 가진다. 여기서 t는 frame index가 아니라 실제 timestamp다.

simple temporal embedding과의 핵심 차이:

- index가 아니라 absolute time. 모델은 "frame 15"가 아니라 "4.2초 지점"을 본다.
- per-clip이 아니라 per-token rotation. 각 visual token은 자기 timestamp로 독립적으로 rotate된다.
- dynamic FPS와 호환된다. 어떤 구간은 2 FPS, 다른 구간은 4 FPS로 sample해도 TMRoPE는 불균등 spacing을 자연스럽게 처리한다.

TMRoPE는 "고양이가 몇 초에 점프하는가?" 같은 query를 가능하게 한다. 모델은 "4.2초"라고 출력할 수 있다. Video-LLaMA는 "clip 초반"이라고만 말할 수 있었다.

### Frame sampling 전략

Uniform: duration 전체에서 N frame을 고르게 sample한다. 단순하지만 motion peak를 놓친다.

Dynamic FPS: motion intensity에 따라 적응적으로 sample한다. optical flow나 frame differencing이 high-motion segment를 골라 더 조밀하게 sample한다. Qwen2.5-VL은 이 방식으로 학습한다.

Event-driven: 경량 detector를 실행하고 action이 일어나는 곳을 더 sample한다. VideoAgent가 사용한다.

Keyframe + context: shot boundary와 주변 frame 몇 개를 sample한다. cinematic content에 사용한다.

### Frame별 pooling

1 FPS, frame당 576 token이면 5분 clip은 172,800 token이다. Qwen2.5-VL-72B의 128k context로도 가능은 하지만 비싸다.

3x3 bilinear pool은 frame당 64 token으로 줄인다. 5분이면 19,200 token이다. 대부분 task의 sweet spot이다.

spatial detail이 덜 중요한 agent workflow에서는 더 강하게 pool한다(6x6 -> frame당 16 token).

### 네 가지 video benchmark

- VideoMME: short + medium + long을 포함하는 종합 video understanding.
- TempCompass: "before" / "after" 질문 중심의 fine-grained temporal reasoning.
- EgoSchema: long-horizon first-person video.
- Video-MMMU: multimodal multi-discipline video question.

전체 video-VLM 평가는 네 가지를 모두 친다. 서로 다른 축을 압박한다. TempCompass는 ordering이 전부이고, EgoSchema는 3분 이상 reasoning에 집중하며, VideoMME는 다양한 duration을 포괄한다.

### 그라운딩 출력 형식

Temporal grounding을 위한 output format:

- Free text: "고양이는 4초 무렵 점프한다." parsing은 쉽지만 부정확하다.
- Structured JSON: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL은 이를 학습한다.
- Token-based: answer와 interleave된 special `<time>4.1</time>` token. Qwen2.5-VL의 internal format.

Downstream 사용에는 token-based가 가장 정확하다. Qwen2.5-VL의 JSON output format은 바로 parse된다.

### 2026 best practice

2026년 video VLM 기준:

- Encoder: M-RoPE 또는 TMRoPE를 갖춘 SigLIP 2(Qwen2.5-VL).
- Frame sampling: max-frame cap을 둔 dynamic FPS(motion에 따라 1-4).
- Per-frame pooling: 3x3 bilinear.
- 출력: time + event field가 있는 structured JSON.
- Benchmarks: general에는 VideoMME + TempCompass, long-horizon에는 EgoSchema.

## 활용하기

`code/main.py` 포함 사항:

- Uniform 및 dynamic-FPS frame sampler.
- 장난감 temporal-grounding evaluator: time T의 "ground truth" event와 model output이 주어지면 tolerance로 accuracy를 score한다.
- Video-LLaMA(16 frames, Q-former), Video-LLaVA(8 frames, MLP), Qwen2.5-VL(dynamic FPS + TMRoPE) 비교.

## 산출물

이 lesson은 `outputs/skill-video-vlm-frame-planner.md`를 만든다. video task(monitoring, action recognition, temporal grounding, summarization)가 주어지면 frame sampler, pooling factor, output format, expected accuracy tier를 고른다.

## 연습 문제

1. 3분 cooking demo에 대해 uniform vs dynamic FPS를 고르라. token count로 정당화하라.

2. TMRoPE는 simple temporal embedding table이 할 수 없는 무엇을 구체적으로 추가하는가?

3. VLM이 학습해 emit할 수 있는 temporal grounding용 JSON schema를 작성하라. error case도 포함하라.

4. Video-LLaVA의 Section 3 "Alignment Before Projection"을 읽어라. 이것이 별도 image encoder와 video encoder를 학습하는 것보다 왜 나은가?

5. VideoMME leaderboard가 주어졌을 때, 2026년 기준 top open model과 top proprietary model의 격차는 얼마인가? 그 격차 중 temporal encoding과 base LLM scale에서 오는 비중은 각각 어느 정도인가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | event가 언제 일어나는지에 대한 구체적 timestamp range를 VLM이 출력하는 것 |
| TMRoPE | "Time-Multimodal RoPE" | Qwen2.5-VL이 사용하는 absolute timestamp 기반 3D rotary position |
| Dynamic FPS | "Motion-aware sampling" | high-motion segment에서는 더 많이, static segment에서는 더 적게 frame을 sample하는 방식 |
| Frame pooling | "Spatial compress per frame" | LLM 전에 bilinear interpolation으로 frame당 patch 수를 줄이는 것 |
| Video Q-former | "Clip compressor" | N frame을 K learned query로 mapping하는 cross-attention bottleneck |
| VideoMME | "Video bench" | short/medium/long video를 포괄하는 종합 benchmark, 2500+ sample |

## 더 읽을거리

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
