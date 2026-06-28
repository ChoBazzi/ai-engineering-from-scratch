# Qwen-VL 계열과 Dynamic-FPS 비디오

> Qwen-VL family, 즉 Qwen-VL(2023), Qwen2-VL(2024), Qwen2.5-VL(2025), Qwen3-VL(2025)은 2026년 가장 영향력 있는 open vision-language model 계보이다. 각 세대는 open ecosystem의 나머지가 12개월 안에 따라 한 단 하나의 결정적 architectural bet를 했다. M-RoPE를 통한 native dynamic resolution, absolute time alignment가 있는 dynamic-FPS sampling, ViT의 window attention, structured agent output format이 그것이다. Qwen3-VL에 이르러 recipe는 안정화되었다. native-aspect-ratio input을 받는 2D-RoPE-ViT encoder, 큰 Qwen3 language base로 들어가는 MLP projector, 그리고 OCR, grounding, agent behavior를 first-class target으로 강조하는 training stage다. 이 lesson은 각 knob이 왜 그 자리에 있는지 이해하도록 family를 시간순으로 읽는다.

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 minutes

## 학습 목표

- M-RoPE의 세 축 rotation(temporal, height, width)을 계산하고 세 축이 모두 필요한 이유를 설명한다.
- video에 대한 dynamic-FPS sampling strategy를 고르고 tokens-per-second와 event-detection accuracy 사이를 추론한다.
- 네 가지 Qwen-VL generational upgrade를 순서대로 말하고 각각이 가능하게 한 것을 설명한다.
- Qwen2.5-VL-style JSON agent output format을 연결하고 VLM response에서 structured tool call을 parse한다.

## 문제

Qwen-VL은 2023년 8월 LLaVA-1.5와 BLIP-2에 대한 직접적인 응답으로 공개되었다. Qwen team이 겨냥한 gap은 resolution, video, structured output 세 가지였다.

Resolution: LLaVA-1.5는 336x336에서 동작했다. photo에는 괜찮지만 중국어 invoice나 dense spreadsheet screenshot에는 쓸모없다. Qwen-VL의 첫 번째 innovation은 448x448과 grounded bounding-box output으로, model이 대상을 가리킬 수 있게 했다.

Video: Video-LLaMA는 frame별 encoder를 stack하여 LLM에 넣었다. short clip에는 작동했지만 temporal axis 자체가 signal인 multi-minute video에는 맞지 않았다. Qwen team은 시간을 이해하는 단일 encoder를 원했다.

Structured output: LLaVA는 free-form text를 방출했다. agent에는 JSON이 필요하다. Qwen-VL은 bounding-box coordinate를 text로 포함하는 explicit JSON output format으로 학습했다.

모든 Qwen-VL generation은 이 세 축 중 하나를 확장한다.

## 개념

### Qwen-VL(2023년 8월)

첫 세대는 encoder로 OpenCLIP ViT-bigG/14(2.5B params), LLama-compatible Q-Former(256 query를 쓰는 1-step), Qwen-7B base를 사용했다. 기여:

- 448x448 resolution(당시 open VLM 기준 SOTA).
- Grounding: explicit coordinate-token output이 있는 image-text pair로 학습. 예: "The cat is at <box>(112, 204), (280, 344)</box>".
- 처음부터 중국어 + 영어 multilingual training.

당시 benchmark에서는 영어에서 GPT-4V와 경쟁력 있었고 중국어에서는 지배적이었다. grounding supervision이 진짜 headline이었다.

### Qwen2-VL(2024년 9월): M-RoPE와 native resolution

Qwen2-VL은 fixed-resolution + Q-Former stack을 natively dynamic-resolution ViT encoder로 교체했다. 주요 변화:

- Native dynamic resolution. ViT는 28로 나누어지는 임의 HxW를 받는다(patch 14 with 2x spatial merge). 1120x672 이미지(40x24 merged patch)는 960 visual token을 만든다. resize도, tiling도, thumbnail도 없다.
- M-RoPE(Multimodal RoPE). 각 token은 1D 대신 3D position(t, h, w)을 가진다. image는 t=0이고, video는 t = frame_index다. RoPE는 axis별 frequency로 query/key vector를 rotate한다. positional embedding table이 없다.
- MLP projector. Q-Former를 버리고 merged patch token 위에 2-layer MLP를 사용한다.
- dynamic FPS를 가진 video. 기본값은 1-2 FPS sampling이지만 model은 임의 frame count를 받는다.

결과적으로 Qwen2-VL-7B는 여러 multimodal benchmark에서 GPT-4o와 맞먹었고 DocVQA에서는 이겼다(94.5 vs 88.4). architecture change가 결정적 움직임이었다.

### Qwen2.5-VL(2025년 2월): dynamic FPS + absolute time

Qwen2.5-VL의 큰 변화는 video였다. Dynamic FPS는 단순히 "필요하면 더 많은 frame을 sample한다"가 아니다. 논문은 다음을 정식화했다.

- Absolute time token. positional index(frame 0, 1, 2...) 대신 실제 timestamp를 사용한다. "At 0:04, the cat jumps." model은 frame token 사이에 interleaved된 `<time>0.04</time>` token을 본다.
- Dynamic FPS. 느린 footage에는 1 FPS, action에는 4+ FPS로 sample한다. user 또는 trainer가 선택하고 M-RoPE가 적응한다.
- ViT의 window attention. throughput을 위해 spatial attention은 windowed(local within blocks)이고, 몇 layer마다 global attention을 넣는다.
- Explicit JSON output format. tool-call data로 학습한다: "{\"tool\": \"click\", \"coords\": [380, 220]}". 바로 agent-ready다.
- MRoPE-v2 scaling. position을 max input size에 맞춰 scale하여 10분 video에서도 frequency range가 고갈되지 않는다.

Benchmark: Qwen2.5-VL-72B는 대부분의 video benchmark에서 GPT-4o를 이기고, document에서는 Gemini 2.0과 맞먹으며, GUI grounding에서는 open-model SOTA를 세운다(ScreenSpot: GPT-4o 38% 대비 84% accuracy).

### Qwen3-VL(2025년 11월)

Qwen3-VL은 재발명보다 통합에 가까운 incremental upgrade다. 더 큰 LLM backbone(Qwen3-72B), 확장된 training data, 개선된 OCR, Qwen3 "thinking mode"를 통한 더 강한 reasoning을 제공한다. ViT와 M-RoPE는 유지된다. 논문은 architecture보다 data와 training improvement에 초점을 둔다.

계보의 핵심 takeaway: 2025년이 되면 Qwen-VL architecture는 안정화되었다. 추가 세대는 primitive가 아니라 compute와 data를 scale한다.

### M-RoPE 수학

고전 RoPE는 position `m`을 사용해 dimension `d`의 query `q`를 paired coordinate로 rotate한다.

```text
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE는 hidden dim을 세 band로 나눈다. 예를 들어 `d = 96`이면 temporal에 32 dim, height에 32 dim, width에 32 dim을 할당한다. 각 band는 자기 axis position으로 rotate한다. (t=5, h=10, w=20)의 patch는 세 band에 `R_t(5)`, `R_h(10)`, `R_w(20)` rotation을 적용받는다.

Text token은 `t = text_index, h = 0, w = 0`(또는 normalized choice)을 사용해 compatibility를 유지한다. Video frame은 `t = frame_time, h = row, w = col`을 사용한다. Single image는 `t = 0`을 사용한다.

장점은 하나의 position encoding이 branch code나 다른 position table 없이 text, image, video를 처리한다는 점이다.

### Dynamic-FPS 샘플링 로직

길이 `T`초 video와 target-token budget `B`가 주어졌다고 하자.

1. 감당 가능한 최대 FPS를 계산한다. `fps_max = B / (T * tokens_per_frame)`.
2. `{1, 2, 4, 8}`에서 `fps <= fps_max`를 만족하는 target FPS를 고른다.
3. motion이 높으면(optical-flow heuristic 또는 explicit user request) 더 높은 FPS를 고른다. motion이 낮으면 더 낮게 고른다.
4. 선택한 FPS로 균일 sample하고 frame 사이에 `<time>t</time>` token을 넣는다.

Qwen2.5-VL은 이 logic을 암묵적으로 학습한다. inference에서는 user가 `fps` parameter로 제어한다. 4 FPS, frame당 81 token의 60초 action sequence는 19440 token이며 32k context에서 관리 가능하다.

### 구조화된 에이전트 출력

Qwen2.5-VL의 agent training은 structured tool call을 명시적으로 target한다.

```json
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Parsing은 결정적이다. model output에 JSON.parse를 적용하면 된다. free-form "click at (1024, 512)"와 비교하면 regex와 ambiguity handling이 필요하지 않다. 이 전환이 Qwen2.5-VL의 ScreenSpot score가 Qwen2-VL의 55%에서 84%로 뛰어오른 이유다.

## 활용하기

`code/main.py`는 다음을 구현한다.

- text, image patch, video frame이 섞인 packed sequence에 대한 M-RoPE position computation.
- Dynamic-FPS sampler: (duration, budget, motion_level)이 주어지면 FPS를 고르고 frame timestamp를 출력한다.
- coordinate field가 있는 tool-call response를 처리하는 toy Qwen2.5-VL JSON-output parser.

실행한 뒤 5분 video에서 fixed-FPS를 dynamic-FPS로 바꿀 때 차이를 체감해 보라.

## 산출물

이 lesson은 `outputs/skill-qwen-vl-pipeline-designer.md`를 만든다. video task(monitoring, agent, action recognition, accessibility)가 주어지면 Qwen2.5-VL configuration(frame budget, FPS strategy, window-attention flag, agent-output mode)과 latency estimate를 출력한다. video product에 Qwen-VL-family model을 deploy할 때마다 사용하라.

## 연습 문제

1. hidden 48(각 band 16, base theta 10000)에서 (t=3, h=5, w=7) patch의 M-RoPE rotation을 계산하라. 각 band의 첫 세 pair에 대한 rotation angle을 보이라.

2. 10분 security-camera recording을 1 FPS로 sample하면 frame이 몇 개인가? 384 resolution에 3x pool을 쓰면 total token은 몇 개인가? Qwen2.5-VL의 default 32k context가 처리할 수 있는가?

3. 30초 tennis rally, 30초 recipe demo, 30초 UI-agent recording에 대해 FPS를 고르라. dynamic-FPS logic으로 각각을 정당화하라.

4. Qwen2.5-VL은 Q-Former를 완전히 버린다. 2023년에는 아니었지만 2025년에는 simple MLP가 작동하는 이유는 무엇인가? 힌트: data scale과 encoder quality.

5. Qwen2.5-VL JSON tool-call output 세 개를 Python dict로 parse하라. malformed JSON에서는 무엇이 실패하며, Qwen cookbook은 어떤 recovery strategy를 권장하는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | hidden dim 안에 temporal, height, width band를 가진 3D rotary position embedding |
| Dynamic FPS | "Smart sampling" | motion, duration, token budget에 따라 video별로 선택되는 frame sampling rate |
| Absolute time token | "Timestamp token" | model이 frame index가 아니라 실제 초 단위를 보도록 sequence에 interleaved되는 `<time>t</time>` |
| Window attention | "Local attention" | 속도를 위해 작은 window로 제한한 spatial self-attention. global attention은 주기적으로 추가된다 |
| Structured agent output | "JSON mode" | VLM이 coords와 tool name이 있는 parseable JSON을 방출하도록 가르치는 training data supervision |
| min_pixels / max_pixels | "Resolution bounds" | total pixel count와 그에 따른 token count를 제한하는 Qwen2.5-VL per-request control |
| Grounding | "Point-at-it" | bounding-box coordinate를 text token으로 출력하는 것. Qwen-VL v1부터 사용됐다 |

## 더 읽을거리

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
