# 인페인팅, 아웃페인팅 및 이미지 편집

> 텍스트-이미지는 새로운 것을 만든다. 인페인팅은 기존 이미지를 고친다. 프로덕션에서 과금 가능한 이미지 작업의 70%는 편집이다. 배경을 바꾸고, 로고를 지우고, 캔버스를 확장하고, 손을 다시 생성한다. 인페인팅은 diffusion이 제값을 하는 지점이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 minutes

## 문제

클라이언트가 배경에 거슬리는 표지판이 들어간 완벽한 제품 사진을 보냈다. 표지판만 지우고 나머지는 픽셀 단위로 동일하게 남기고 싶다. 처음부터 text-to-image를 실행할 수는 없다. 결과의 색, 조명, 제품 각도가 달라질 것이기 때문이다. 원하는 것은 마스크된 영역*만* 다시 생성하고, 그 재생성이 주변 맥락을 따르게 하는 것이다.

이것이 인페인팅이다. 변형은 다음과 같다.

- **인페인팅(Inpainting).** 마스크 안쪽을 다시 생성하고 바깥 픽셀은 유지한다.
- **아웃페인팅(Outpainting).** 마스크 바깥쪽 또는 캔버스 너머를 다시 생성하고 안쪽은 유지한다.
- **이미지 편집(Image editing).** 전체 이미지를 다시 생성하되 원본에 대한 의미적 또는 구조적 충실도를 유지한다(SDEdit, InstructPix2Pix).

2026년의 모든 diffusion 파이프라인은 인페인팅 모드를 제공한다. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit 모두 같은 원리로 동작한다.

## 개념

![인페인팅: 맥락 보존 재주입을 사용하는 마스크 인식 디노이징](../assets/inpainting.svg)

### 순진한 접근법과 왜 틀렸는가

표준 text-to-image를 마스크와 함께 실행한다. 각 샘플링 단계에서 노이즈가 낀 latent의 마스크되지 않은 영역을 forward-diffusion된 깨끗한 이미지로 교체한다. 동작은 하지만 형편없다. 모델이 마스크된 영역 안에 무엇이 있는지 알지 못하므로 경계 아티팩트가 번진다.

### 올바른 인페인팅 모델

4개 대신 9개 입력 채널을 받는 수정된 U-Net을 학습한다.

```text
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

추가 채널은 VAE로 인코딩한 원본 이미지의 복사본과 단일 채널 마스크다. 학습 시에는 이미지의 영역을 무작위로 마스크하고, 마스크되지 않은 영역을 깨끗한 conditioning 신호로 제공한 상태에서 모델이 마스크된 영역만 디노이즈하도록 학습한다. 추론 시 모델은 마스크된 영역을 둘러싼 내용을 "볼" 수 있고, 일관된 보완 이미지를 만든다.

SD-Inpaint, SDXL-Inpaint, Flux-Fill은 모두 이 9채널 또는 유사 입력을 사용한다. Diffusers의 `StableDiffusionInpaintPipeline`, `FluxFillPipeline`도 마찬가지다.

### SDEdit(Meng et al., 2022): 재학습 없는 편집

원본 이미지에 중간 시점 `t`까지 노이즈를 더한 뒤, 새 프롬프트로 `t`에서 0까지 역방향 체인을 실행한다. 재학습은 필요 없다. 시작 `t`의 선택은 충실도와 창의적 자유 사이의 균형을 정한다.

- `t/T = 0.3` -> 원본과 거의 동일하고 작은 스타일 변화만 생김
- `t/T = 0.6` -> 중간 정도의 편집, 대략적 구조 보존
- `t/T = 0.9` -> 거의 노이즈에서 생성되어 원본 보존이 최소화됨

### InstructPix2Pix(Brooks et al., 2023)

diffusion 모델을 `(input_image, instruction, output_image)` 트리플로 fine-tune한다. 추론 시 입력 이미지와 텍스트 지시문("make it sunset", "add a dragon")을 모두 조건으로 사용한다. CFG scale은 두 개이며, image scale과 text scale이다.

### RePaint(Lugmayr et al., 2022)

표준 unconditional diffusion 모델을 유지한다. 각 역방향 단계에서 resample한다. 즉, 가끔 더 노이즈가 많은 상태로 되돌아갔다가 다시 생성한다. 경계 아티팩트를 줄인다. 학습된 인페인팅 모델이 없을 때 사용한다.

## 직접 만들기

`code/main.py`는 5차원 데이터에 대한 장난감 1-D 인페인팅 방식을 구현한다. 각 샘플이 두 클러스터 중 하나에서 나온 5개 float인 5-D mixture 데이터에 DDPM을 학습한다. 추론 시 5개 차원 중 2개를 "마스크"하고, 각 단계에서 마스크되지 않은 3개 차원의 noisy-forward 버전을 주입하며, 마스크된 차원만 다시 생성한다.

### 1단계: 5-D DDPM 데이터

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 2단계: 모든 5개 차원에 대한 denoiser 학습

표준 DDPM이다. 네트워크는 5-D noisy input에 대해 5-D 노이즈 예측을 출력한다.

### 3단계: 추론 시 마스크 인식 역방향

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # 마스크되지 않은 차원을 깨끗한 원본의 새로 노이즈화한 버전으로 교체한다
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...그다음 x_t에 일반 역방향 단계를 실행한다
```

이것은 순진한 접근법이며 장난감 1-D 데이터에서는 동작한다. 실제 이미지 인페인팅은 텍스처 일관성이 훨씬 중요하므로 9채널 입력을 사용한다.

### 4단계: 아웃페인팅

아웃페인팅은 마스크를 뒤집은 인페인팅이다. 새 캔버스, 즉 이전에는 존재하지 않았던 영역을 마스크하고, 나머지는 원본으로 채운다. 학습 목표는 동일하다.

## 함정

- **이음새.** 순진한 접근법은 gradient 정보가 마스크를 가로질러 흐르지 않기 때문에 눈에 보이는 경계를 남긴다. 해결책: 마스크를 8-16픽셀 확장하거나 올바른 인페인팅 모델을 사용한다.
- **마스크 누출.** conditioning 이미지의 마스크되지 않은 영역이 저품질이거나 노이즈가 많으면, 마스크 안쪽 생성도 오염된다. 약간 디노이즈하거나 블러 처리한다.
- **CFG는 마스크 크기와 상호작용한다.** 작은 마스크에 높은 CFG를 쓰면 채도가 과한 패치가 생긴다. 작은 편집에는 CFG를 낮춘다.
- **SDEdit 충실도 절벽.** `t/T = 0.5`에서 `t/T = 0.6`으로만 가도 피사체의 정체성을 잃을 수 있다. sweep하고 checkpoint를 남긴다.
- **프롬프트 불일치.** 프롬프트는 새 내용만이 아니라 이미지 *전체*를 설명해야 한다. "a cat"이 아니라 "A cat sitting on a chair"라고 쓴다.

## 사용하기

| 작업 | 파이프라인 |
|------|----------|
| 객체 제거, 작은 마스크 | SD-Inpaint 또는 Flux-Fill, 표준 프롬프트 |
| 하늘 교체 | SD-Inpaint + "blue sky at sunset" |
| 캔버스 확장 | SDXL outpaint mode(8px feather) 또는 outpaint mask를 사용한 Flux-Fill |
| 손 / 얼굴 재생성 | 피사체를 다시 설명하는 프롬프트 + ControlNet-Openpose를 사용한 SD-Inpaint |
| 한 영역의 스타일 변경 | 마스크된 영역에 `t/T=0.5`로 SDEdit |
| "Make it sunset" | InstructPix2Pix 또는 Flux-Kontext |
| 배경 교체 | SAM mask -> SD-Inpaint |
| 초고충실도 | 가장 어려운 경우 Flux-Fill 또는 GPT-Image(hosted) |

SAM(Meta의 Segment Anything, 2023) + diffusion inpaint는 2026년의 배경 제거 파이프라인이다. SAM 2(2024)는 비디오에서 동작한다.

## 출시하기

`outputs/skill-editing-pipeline.md`를 저장한다. 이 스킬은 원본 이미지 + 편집 설명 + 선택적 마스크 또는 SAM 프롬프트를 입력받아 다음을 출력한다. 마스크 생성 접근법, base model, CFG scale(image + text), SDEdit-t 또는 인페인팅 모드, QA 체크리스트.

## 연습문제

1. **쉬움.** `code/main.py`에서 마스크되는 차원의 비율을 0.2에서 0.8까지 바꿔라. 어떤 비율에서 인페인트 품질(마스크된 차원의 residual)이 unconditional generation과 같아지는가?
2. **중간.** RePaint를 구현하라. 역방향 10단계마다 5단계 뒤로 점프하고(노이즈 추가) 다시 디노이즈하라. 마스크 경계에서 boundary residual이 줄어드는지 측정하라.
3. **어려움.** Hugging Face diffusers로 SD 1.5 Inpaint + ControlNet-Openpose와 Flux.1-Fill을 얼굴 재생성 작업 20개에서 비교하라. 포즈 준수와 정체성 보존을 따로 점수화하라.

## 핵심 용어

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| Inpainting | "구멍 채우기" | 마스크 안쪽을 다시 생성하고 바깥 픽셀은 유지한다. |
| Outpainting | "캔버스 확장" | 캔버스 바깥을 다시 생성하고 안쪽은 유지한다. |
| 9-channel U-Net | "올바른 인페인팅 모델" | 입력으로 `noisy \| encoded-source \| mask`를 받는 U-Net. |
| SDEdit | "노이즈 수준이 있는 img2img" | 시점 `t`까지 노이즈를 넣고 새 프롬프트로 디노이즈한다. |
| InstructPix2Pix | "텍스트만으로 하는 편집" | (image, instruction, output) 트리플로 fine-tune된 diffusion. |
| RePaint | "재학습 없음" | 이음새를 줄이기 위해 역방향 중 주기적으로 다시 노이즈를 넣는다. |
| SAM | "Segment Anything" | 클릭이나 박스로 만드는 마스크 생성기이며 inpaint와 함께 쓴다. |
| Flux-Kontext | "맥락을 보고 편집" | 편집을 위해 reference image + instruction을 받는 Flux 변형. |

## 프로덕션 노트: 편집 파이프라인은 latency에 민감하다

이미지를 편집하는 사용자는 왕복 시간이 5초 미만이기를 기대한다. 1024²에서 30-step SDXL-Inpaint는 L4에서 3-4초가 걸리고, SAM mask generation(~200 ms)과 VAE encode/decode(합쳐서 ~500 ms)가 추가된다. 프로덕션 관점에서는 throughput-bound가 아니라 TTFT-bound다. batch 1, 낮은 concurrency, 모든 단계 최소화가 필요하다.

- **SAM-H가 느린 쪽이다.** 1024²에서 SAM-H는 ~200 ms이고, SAM-ViT-B는 약간의 품질 손실로 ~40 ms다. SAM 2(video)는 temporal overhead를 추가하므로 단일 이미지 편집에는 쓰지 않는다.
- **가능하면 encode를 건너뛰어라.** `pipe.image_processor.preprocess(img)`는 latents로 인코딩한다. 이전 생성에서 나온 latents가 있다면(반복 편집 UI에서는 흔하다) `latents=...`로 직접 전달해 VAE encode 한 번을 건너뛴다.
- **마스크 확장은 throughput에도 중요하다.** 작은 마스크는 U-Net forward pass 대부분을 낭비한다. 어차피 마스크되지 않은 픽셀은 clamp되기 때문이다. `diffusers`의 `StableDiffusionInpaintPipeline`은 상관없이 전체 U-Net을 실행한다. 9채널 proper-inpaint 변형만 masked compute를 활용한다.
- **Flux-Kontext가 2025년의 답이다.** `(source_image, instruction)`에 대한 단일 forward pass다. 별도 마스크도, SDEdit noise sweep도 없다. H100에서는 약 1.5초 만에 편집을 낸다. 아키텍처적 교훈은 단계를 합치라는 것이다.

## 더 읽을거리

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 학습 없는 인페인팅.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 텍스트 지시 편집.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — 마스크 소스인 SAM.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 비디오 SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention 수준 편집.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024년 도구.
