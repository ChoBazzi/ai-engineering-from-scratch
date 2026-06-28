# 3D 생성

> 3D는 2D-to-3D leverage가 가장 강한 modality다. 2023년의 돌파구는 3D Gaussian Splatting이었다. 2024-2026년의 생성형 흐름은 그 위에 multi-view diffusion + 3D reconstruction을 쌓아 단일 prompt 또는 photo에서 object와 scene을 만든다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 4 (Vision), Phase 8 · 07 (Latent Diffusion)
**Time:** ~45 minutes

## 문제

3D content는 다루기 어렵다.

- **Representation.** Meshes, point clouds, voxel grids, signed distance fields(SDFs), neural radiance fields(NeRFs), 3D Gaussians. 각각 trade-off가 있다.
- **데이터 부족.** ImageNet은 14M images를 갖는다. 가장 큰 깨끗한 3D dataset(Objaverse-XL, 2023)은 약 10M objects이고, 대부분 품질이 낮다.
- **Memory.** 512³ voxel grid는 128M voxels다. 유용한 scene NeRF는 ray당 1M samples가 필요하다. 생성은 reconstruction보다 더 어렵다.
- **Supervision.** 2D image에는 pixels가 있다. 3D에서는 보통 몇 개의 2D view만 있고, 이를 3D로 lift해야 한다.

2026년 stack은 두 문제를 분리한다. 먼저 diffusion model로 *2D multi-view images*를 생성한다. 그다음 그 이미지에 *3D representation*(보통 Gaussian splatting)을 fit한다.

## 개념

![3D 생성: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### 표현: 3D Gaussian Splatting(Kerbl et al., 2023)

scene을 약 1M개의 3D Gaussian 구름으로 표현한다. 각 Gaussian은 59개 parameter를 가진다. position(3), covariance(6, 또는 quaternion 4 + scale 3), opacity(1), spherical-harmonics color(degree 3에서 48, degree 0에서 3).

Rendering = projection + alpha-compositing이다. 빠르다(4090에서 1080p 약 100 fps). differentiable하다. ground-truth photos에 대한 gradient descent로 fit한다. consumer GPU에서 한 scene은 5-30분 안에 fit된다.

그 위에 얹힌 2023-2024년의 혁신 두 가지는 다음과 같다.
- **Generative Gaussian splats.** LGM, LRM, InstantMesh 같은 모델은 하나 또는 몇 장의 image에서 Gaussian cloud를 직접 예측한다.
- **4D Gaussian Splatting.** dynamic scene을 위해 per-frame offset을 갖는 Gaussians.

### Multi-view diffusion

pretrained image diffusion model을 fine-tune하여 text prompt 또는 single image에서 같은 object의 여러 consistent views를 생성한다. Zero123(Liu et al., 2023), MVDream(Shi et al., 2023), SV3D(Stability, 2024), CAT3D(Google, 2024). 보통 object 주변의 4-16 views를 출력하고, Gaussian splatting 또는 NeRF로 3D에 lift한다.

### Text-to-3D 파이프라인

| 모델 | 입력 | 출력 | 시간 |
|-------|-------|--------|------|
| DreamFusion (2022) | text | SDS를 통한 NeRF | asset당 ~1 hour |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

2025-2026년의 방향은 game engine에 적합한 PBR material을 갖춘 direct text-to-mesh 모델이다. multi-view diffusion intermediate step은 여전히 일반 object에서 가장 성능이 좋은 recipe다.

### NeRF(맥락용)

Neural Radiance Field(Mildenhall et al., 2020). 작은 MLP가 `(x, y, z, view direction)`을 받아 `(color, density)`를 출력한다. ray를 따라 integrate하여 render한다. mesh 기반 novel-view synthesis보다 품질이 높지만 render가 100-1000배 느리다. 대부분의 real-time 사용에서는 Gaussian splatting에 대체되었지만 연구에서는 여전히 지배적이다.

## 직접 만들기

`code/main.py`는 장난감 2D "Gaussian splatting" fit을 구현한다. synthetic target image(부드러운 gradient)를 2D Gaussian splat의 합으로 표현한다. target과 맞도록 positions, colors, covariances를 gradient descent로 최적화한다. 두 핵심 연산인 forward render(splat + alpha-composite)와 gradient descent fit을 보게 된다.

### 1단계: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### 2단계: splat을 합산해 render

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

실제 3D Gaussian splatting은 Gaussian을 depth로 정렬하고 순서대로 alpha-composite한다. 우리의 2D toy는 단순히 더한다.

### 3단계: gradient descent로 fit

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## 함정

- **View inconsistency.** 4개 view를 독립적으로 생성했는데 object structure가 서로 다르면 3D fit은 흐릿해진다. 해결책: shared attention이 있는 multi-view diffusion.
- **Back-side hallucination.** Single-image -> 3D는 보이지 않는 뒷면을 상상해야 한다. 품질은 크게 흔들린다.
- **Gaussian splat explosion.** 제약 없는 학습은 10M splats까지 늘어나 overfit한다. 3D-GS 원 논문의 densification + pruning heuristic이 필수다.
- **Topology issues.** implicit field(SDF)에서 나온 mesh는 구멍이나 self-intersection이 자주 생긴다. 출시 전에 remesher(예: blender의 voxel remesh)를 실행한다.
- **학습 데이터 license.** Objaverse에는 혼합 license가 있다. commercial use는 모델마다 다르다.

## 사용하기

| 작업 | 2026년 선택지 |
|------|-----------|
| 사진으로부터 scene reconstruction | Gaussian splatting(3DGS, Gsplat, Scaniverse) |
| 게임용 Text-to-3D object | Meshy 4 또는 Rodin Gen-1.5(PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| 적은 이미지로 novel-view synthesis | CAT3D, SV3D |
| Dynamic scene reconstruction | 4D Gaussian Splatting |
| Avatar / clothed human | Gaussian Avatar, HUGS |
| Research / SOTA | 지난주에 나온 것 |

게임 또는 e-commerce pipeline에서 프로덕션 3D를 출시하려면 Meshy 4 또는 Rodin Gen-1.5가 Unity / Unreal에 바로 들어가는 PBR mesh를 출력한다.

## 출시하기

`outputs/skill-3d-pipeline.md`를 저장한다. 이 스킬은 3D brief(input: text / one image / few images; output: mesh / splat / NeRF; usage: render / game / VR)를 입력받아 다음을 출력한다. pipeline(multi-view diffusion + fit 또는 direct mesh model), base model, iteration budget, topology post-processing, 필요한 material channels.

## 연습문제

1. **쉬움.** `code/main.py`를 4, 16, 64 Gaussians로 실행하라. target 대비 final MSE를 보고하라.
2. **중간.** color Gaussians(RGB)로 확장하라. reconstruction이 target color pattern과 맞는지 확인하라.
3. **어려움.** gsplat 또는 Nerfstudio를 사용해 50장 photo capture에서 실제 object를 reconstruct하라. fit time과 held-out view에서의 final SSIM을 보고하라.

## 핵심 용어

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| 3D Gaussian Splatting | "3DGS" | scene을 3D Gaussians의 구름으로 표현하고 differentiable alpha-composite render를 수행한다. |
| NeRF | "Neural radiance field" | 3D point에서 color + density를 출력하는 MLP이며 ray integration으로 render한다. |
| Triplane | "세 개의 2-D plane" | 3D를 세 개의 axis-aligned 2-D feature grid로 factorize하며 volumetric보다 싸다. |
| SDS | "Score distillation sampling" | 2D-diffusion score를 pseudo-gradient로 사용해 3D model을 학습한다. |
| Multi-view diffusion | "여러 view를 한 번에" | consistent camera views의 batch를 출력하는 diffusion model. |
| PBR | "Physically-based rendering" | albedo, roughness, metallic, normal channel을 갖는 material. |
| Densification | "splat 늘리기" | high-gradient region에서 splat을 split / clone하는 3DGS training heuristic. |

## 프로덕션 노트: 3D에는 아직 공유 substrate가 없다

이미지(latent diffusion + DiT)와 비디오(spatiotemporal DiT)와 달리, 2026년의 3D에는 단일 지배 runtime이 없다. 프로덕션 decision tree는 representation에 따라 갈라진다.

- **NeRF / triplane.** Inference는 ray-marching + sample별 MLP forward다. 512² render에는 수백만 번의 MLP forward가 필요하다. ray samples를 공격적으로 batch하라. SDPA/xformers가 적용된다.
- **Multi-view diffusion + LRM reconstruction.** 두 단계 파이프라인이다. Stage 1(multi-view DiT)은 Lesson 07과 같은 diffusion server다. Stage 2(LRM transformer)는 views에 대한 one-shot forward pass다. 전체 latency profile은 "diffusion + one-shot"이다. 단계별 serving primitive를 그에 맞게 고른다.
- **SDS / DreamFusion.** Per-asset optimization이지 inference가 아니다. request handler가 아니라 job을 만든다.

대부분의 2026년 제품에서 정답은 "요청 시 multi-view diffusion model을 실행하고, 비동기로 3DGS로 reconstruct한 뒤, real-time viewing을 위해 3DGS를 serve"하는 것이다. 이렇게 하면 workload가 GPU-inference server(fast)와 offline optimizer(slow)로 깔끔하게 나뉜다.

## 더 읽을거리

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
