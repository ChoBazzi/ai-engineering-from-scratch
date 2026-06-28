# ControlNet, LoRA와 Conditioning

> Text만으로는 제어 신호가 둔합니다. ControlNet은 pretrained diffusion model을 복제해 depth map, pose skeleton, scribble, edge image로 조향할 수 있게 합니다. LoRA는 1천만 parameter만 학습해 2B-parameter model을 fine-tune하게 해줍니다. 둘이 결합되면서 Stable Diffusion은 장난감에서 2026년 모든 에이전시가 배포하는 이미지 파이프라인으로 바뀌었습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 minutes

## 문제

"빨간 드레스를 입은 여성이 붐비는 거리에서 개를 산책시키는 장면" 같은 prompt는 개가 *어디에* 있는지, 여성이 *어떤 포즈*인지, 거리의 *perspective*가 어떤지 알려주지 않습니다. Text는 이미지를 지정하는 데 필요한 정보의 약 10%만 고정합니다. 나머지는 시각적이며 말로 효율적으로 설명할 수 없습니다.

모든 신호(pose, depth, canny, segmentation)에 대해 새 conditional model을 처음부터 학습하는 것은 비용이 너무 큽니다. 2.6B-param SDXL backbone은 freeze한 채, conditioning을 읽는 작은 side-network를 붙이고, backbone의 intermediate feature를 살짝 밀어 주고 싶습니다. 이것이 ControlNet입니다.

또한 full model을 다시 학습하지 않고도 모델에 새 concept(내 얼굴, 내 제품, 내 style)을 가르치고 싶습니다. 100배 작은 delta가 필요합니다. 이것이 LoRA입니다. 기존 attention weight에 꽂는 low-rank adapter입니다.

ControlNet + LoRA + text는 2026년 실무자의 toolkit입니다. 대부분의 production image pipeline은 SDXL / SD3 / Flux base 위에 2-5개의 LoRA, 1-3개의 ControlNet, IP-Adapter 하나를 계층적으로 쌓습니다.

## 개념

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Pretrained SD를 가져옵니다. U-Net의 encoder half를 *clone*합니다. 원본은 freeze합니다. Clone이 추가 conditioning input(edge, depth, pose)을 받도록 학습합니다. Clone을 원본의 decoder half에 *zero-convolution* skip connection으로 다시 연결합니다(0으로 초기화된 1×1 conv입니다. 처음에는 no-op이고 delta를 학습합니다).

```text
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv initialization은 ControlNet이 identity로 시작한다는 뜻입니다. 학습 전에도 해를 끼치지 않습니다. 표준 diffusion loss로 1M개의 (prompt, condition, image) triple에서 학습합니다.

Modality별 ControlNet은 작은 side model로 제공됩니다(SDXL은 ~360M, SD 1.5는 ~70M). 추론 시 조합할 수 있습니다.

```text
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Model 안의 임의 linear layer `W ∈ R^{d×d}`에 대해 `W`를 freeze하고 low-rank delta를 추가합니다.

```text
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

여기서 `r << d`입니다. Attention에는 rank 4-16이 표준이고, heavy fine-tune에는 rank 64-128을 씁니다. 새 parameter 수는 `d²` 대신 `2 · d · r`입니다. `d=640`, `r=16`인 SDXL attention에서는 adapter당 410k가 아니라 20k parameter입니다. 20배 감소입니다. 전체 model 기준으로 LoRA는 보통 base 5GB 대비 20-200MB입니다.

추론 시 LoRA를 scale할 수 있습니다. `W' = W + α · B @ A`입니다. `α = 0.5-1.5`가 일반적입니다. 여러 LoRA는 덧셈으로 stack됩니다. 물론 비선형적으로 상호작용한다는 일반적인 주의점은 있습니다.

### IP-Adapter (Ye et al., 2023)

Text와 함께 conditioning으로 *image*를 받는 작은 adapter입니다. CLIP image encoder를 사용해 image token을 만들고, text token과 함께 cross-attention에 주입합니다. Base model당 약 20MB입니다. LoRA 없이도 "이 reference style로 이미지 생성"을 할 수 있게 합니다.

## 조합 가능성 행렬

| 도구 | 제어 대상 | 크기 | 사용할 때 |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | 정확한 layout, composition |
| LoRA | Style, subject, concept | 20-200MB | Personalization, style |
| IP-Adapter | Reference image의 style 또는 subject | 20MB | Text로 look을 설명할 수 없을 때 |
| Textual Inversion | 새 token 하나로 단일 concept 표현 | 10KB | Legacy, 대부분 LoRA로 대체됨 |
| DreamBooth | Subject에 대한 full fine-tune | 2-5GB | 강한 identity, 높은 compute |
| T2I-Adapter | 더 가벼운 ControlNet 대안 | 70MB | Edge device, inference budget |

ControlNet ≈ spatial입니다. LoRA ≈ semantic입니다. 둘 다 쓰세요.

## 직접 만들기

`code/main.py`는 두 메커니즘을 1-D에서 시뮬레이션합니다.

1. **LoRA.** Pretrained linear layer `W`가 있습니다. Freeze합니다. `W + BA`가 target linear layer와 맞도록 low-rank `B @ A`를 학습합니다. `r = 1`만으로 rank-1 correction을 완벽히 학습할 수 있음을 보입니다.

2. **ControlNet-lite.** "Frozen base" predictor와 추가 signal을 읽는 "side network"가 있습니다. Side network의 출력은 0으로 초기화된 학습 가능한 scalar로 gate됩니다(우리 버전의 zero-conv). 학습하면서 gate가 올라가는 것을 확인합니다.

### 1단계: LoRA 수학

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 2단계: 0 초기화 side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

Step 0에서는 출력이 base와 동일합니다. 초기 학습에서는 `gate`가 천천히 update됩니다. Catastrophic drift가 없습니다.

## 함정

- **LoRA over-scaling.** `α = 2`나 `α = 3`은 "더 강하게" 만드는 흔한 hack이지만 over-stylized / broken output을 만듭니다. `α ≤ 1.5`를 유지하세요.
- **ControlNet weight conflict.** Pose ControlNet을 weight 1.0으로, Depth ControlNet도 weight 1.0으로 쓰면 보통 overshoot합니다. Weight 합 ≈ 1.0이 안전한 기본값입니다.
- **잘못된 base의 LoRA.** SDXL LoRA는 attention dimension이 맞지 않으므로 SD 1.5에서 조용히 no-op이 됩니다. Diffusers 0.30+는 경고합니다.
- **Textual Inversion drift.** 한 checkpoint에서 학습한 token은 다른 checkpoint에서 심하게 drift합니다. LoRA가 더 portable합니다.
- **LoRA weight-merging and storage.** 더 빠른 inference를 위해 LoRA를 base model weight에 bake할 수 있지만(runtime addition 없음), runtime에서 `α`를 조절할 수 없게 됩니다. 두 버전을 모두 보관하세요.

## 활용하기

| 목표 | 2026년 pipeline |
|------|---------------|
| 브랜드 art style 재현 | Rank 32에서 엄선한 이미지 ~30장으로 학습한 LoRA |
| 생성 이미지에 내 얼굴 넣기 | DreamBooth or LoRA + IP-Adapter-FaceID |
| 특정 pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| 정확한 layout | ControlNet-Scribble or ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting (Lesson 09) |
| 빠른 1-step style | LCM-LoRA on SDXL-Turbo |

## 출시하기

`outputs/skill-sd-toolkit-composer.md`를 저장하세요. 이 skill은 task(input assets: prompt, optional reference image, optional pose, optional depth, optional scribble)를 받아 tool stack, weight, reproducible seed protocol을 출력합니다.

## 연습 문제

1. **쉬움.** `code/main.py`에서 LoRA rank `r`을 1에서 4까지 바꾸세요. 어느 rank에서 LoRA가 rank-2 target delta와 정확히 일치하나요?
2. **중간.** 두 target transform에 대해 별도 LoRA 두 개를 학습하세요. 함께 load해 additive interaction을 보이세요. 언제 interaction이 linearity를 깨나요?
3. **어려움.** Diffusers로 다음 stack을 사용하세요. SDXL-base + Canny-ControlNet(weight 0.8) + style LoRA(α 0.8) + IP-Adapter(weight 0.6). Stack weight를 바꿔 가며 FID-vs-prompt-adherence trade-off를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-------------------|-----------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skip입니다. Conditioning image를 읽습니다. |
| Zero convolution | "Starts as identity" | 0으로 초기화된 1×1 conv입니다. ControlNet이 no-op으로 시작합니다. |
| LoRA | "Low-rank adapter" | `W + B @ A`, `r << d`입니다. Full fine-tune보다 parameter가 100배 적습니다. |
| rank r | "The knob" | LoRA compression입니다. 보통 4-16, heavy personalization에는 64+입니다. |
| α | "LoRA strength" | LoRA delta의 runtime scaling입니다. |
| IP-Adapter | "Reference image" | CLIP-image token을 통한 작은 image-conditioning adapter입니다. |
| DreamBooth | "Full subject fine-tune" | Subject 이미지 ~30장으로 full model을 학습합니다. |
| Textual Inversion | "New token" | 새 word embedding만 학습합니다. Legacy이며 대부분 대체되었습니다. |

## 프로덕션 노트: LoRA swap, ControlNet lane, multi-tenant serving

실제 text-to-image SaaS는 같은 base checkpoint 위에서 수백 개의 LoRA와 수십 개의 ControlNet을 제공합니다. Serving 문제는 LLM multi-tenancy와 매우 비슷합니다(production literature는 LLM 사례를 continuous batching과 LoRAX / S-LoRA 아래에서 다룹니다).

- **LoRA를 hot-swap하고 merge하지 마세요.** `W' = W + α·B·A`를 base에 merge하면 step당 inference가 약 3-5% 빨라지지만 `α`와 base가 고정됩니다. LoRA를 rank-r delta로 VRAM에 hot 상태로 두세요. Diffusers는 request별 activation을 위해 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`를 제공합니다. Swap cost는 `2 · d · r · num_layers` weight입니다. MB scale이고 sub-second입니다.
- **ControlNet은 두 번째 attention lane입니다.** Cloned encoder는 base와 병렬로 실행됩니다. Weight 1.0짜리 ControlNet 두 개는 step마다 추가 forward pass 두 번이지 merge된 pass 한 번이 아닙니다. Batch-size 여유는 제곱적으로 줄어듭니다. Active ControlNet 하나당 step cost를 약 1.5×로 잡으세요.
- **Quantized LoRA도 가능합니다.** Base를 quantize했다면(Lesson 07, 8GB에서 Flux 참고), LoRA delta도 8-bit 또는 4-bit로 깔끔하게 quantize됩니다. QLoRA-style loading을 쓰면 4-bit Flux base 위에 5-10개 LoRA를 memory 초과 없이 stack할 수 있습니다.

Flux-specific: Niels의 Flux-on-8GB notebook은 base를 4-bit로 quantize합니다. 그 quantized base 위에 style LoRA(`pipe.load_lora_weights("user/style-lora")`)를 `weight_name="pytorch_lora_weights.safetensors"`로 stack해도 여전히 동작합니다. 이것이 2026년에 대부분의 SaaS agency가 배포하는 레시피입니다.

## 더 읽을거리

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA(원래 LLM용이며 diffusion으로 port됨).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — ControlNet보다 가벼운 대안.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — reference pipeline.
