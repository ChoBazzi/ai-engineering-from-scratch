# 비디오 생성

> 이미지는 2-D 텐서다. 비디오는 3-D 텐서다. 이론은 같지만 compute는 10-100배 더 어렵다. OpenAI의 Sora(2024년 2월)는 그것이 가능함을 증명했다. 2026년에는 Veo 2, Kling 1.5, Runway Gen-3, Pika 2.0, WAN 2.2가 1080p text-to-video를 프로덕션으로 제공한다. open-weights stack(CogVideoX, HunyuanVideo, Mochi-1, WAN 2.2)은 12개월 정도 뒤처져 있다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 minutes

## 문제

24fps의 10초 1080p 비디오는 1920×1080×3 픽셀 프레임 240장이다. 클립 하나당 raw data가 약 1.5 GB다. pixel-space diffusion은 불가능하다. 필요한 것은 다음과 같다.

1. **시공간 압축.** 프레임이 아니라 비디오를 spatial-temporal patch의 시퀀스로 인코딩하는 VAE.
2. **시간적 일관성.** 프레임들은 몇 초에 걸쳐 content, lighting, object identity를 공유해야 한다. 네트워크는 motion을 모델링해야 한다.
3. **Compute budget.** 비디오 학습은 같은 모델 크기의 이미지보다 10-100배 비싸다.
4. **Conditioning.** 텍스트, 이미지(first-frame), 오디오, 또는 다른 비디오. 대부분의 프로덕션 모델은 네 가지 모두를 받는다.

이를 해결한 아키텍처는 시공간 patch에 적용한 **Diffusion Transformer(DiT)**이며, 거대한 (prompt, caption, video) 데이터셋으로 학습된다. diffusion loss는 Lesson 06과 같다.

## 개념

![비디오 diffusion: patchify, DiT, decode](../assets/video-generation.svg)

### Patchify하기

3D VAE(학습된 시공간 압축)로 비디오를 인코딩한다. latent shape은 `[T_latent, H_latent, W_latent, C_latent]`다. 이를 `[t_p, h_p, w_p]` 크기 patch로 나눈다. Sora 스타일 모델에서는 `t_p = 1`(프레임별 patch) 또는 `t_p = 2`(두 프레임마다 하나)다. 10초 1080p 비디오는 약 20,000-100,000개 patch로 압축된다.

### 시공간 DiT

transformer가 patch의 flat sequence를 처리한다. 각 patch에는 3D positional embedding(time + y + x)이 있다. attention은 보통 factorize된다.

- **Spatial attention**은 각 프레임의 patch 안에서 수행된다.
- **Temporal attention**은 같은 공간 위치의 프레임 간에 수행된다.
- **Full 3D attention**은 16-100배 더 비싸며, 저해상도 또는 연구에서만 사용된다.

### 텍스트 conditioning

대형 text encoder와 cross-attention을 수행한다(Sora는 T5-XXL, CogVideoX-5B도 T5-XXL 사용). 긴 프롬프트가 중요하다. Sora의 학습셋은 GPT가 생성한 dense re-caption을 포함했고, 클립당 평균 200 tokens였다.

### 학습

시공간 latent에 대한 표준 diffusion loss(ε 또는 v prediction)를 사용한다. 데이터는 web video + 약 100M curated clips + synthetic text captions다. compute는 작은 연구 실행도 10,000+ GPU hours가 필요하고, Sora 규모는 100,000+다.

## 2026년 프로덕션 지형

| 모델 | 날짜 | 최대 길이 | 최대 해상도 | Open weights 여부 | 주목할 점 |
|-------|------|--------------|---------|---------------|---------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 아니요 | 규모를 키웠을 때 world simulator 속성을 처음 보여준 모델 |
| Sora Turbo | 2024-12 | 20s | 1080p | 아니요 | 추론이 5배 빠른 프로덕션 Sora |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 아니요 | 2025년 최고 품질 + physics |
| Veo 3 | 2025 Q3 | 15s | 4K | 아니요 | Native audio와 더 강한 camera control |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 아니요 | 2025 Q1 최고의 human motion |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 아니요 | 그 위에 얹힌 professional video tools |
| Pika 2.0 | 2024-10 | 5s | 1080p | 아니요 | 가장 강한 character consistency |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 예 (2B, 5B) | 첫 open 5B-scale video |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 예 (13B) | 2024년 말 open SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 예 (10B) | 가장 permissive한 license |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 예 | 2025년 중반 가장 강한 open model |

Open weights는 이미지 영역보다 더 빠르게 격차를 좁히고 있다. 2026년 중반에는 HunyuanVideo + WAN 2.2 LoRA가 이미 대부분의 open-source workflow를 구동한다.

## 직접 만들기

`code/main.py`는 핵심 시공간 DiT 아이디어를 시뮬레이션한다. 작은 synthetic video를 patchify하고, patch별 position embedding을 더한 뒤, patch에 대한 transformer-style attention으로 전체 sequence를 디노이즈한다. numpy 없이 순수 Python으로 작성했다. 인접 프레임 patch들이 denoiser와 position embedding을 공유할 때, 1-D에서도 temporal coherence가 생김을 보여준다.

### 1단계: synthetic 1-D "video" patchify

```python
def make_video(T_frames=8, rng=None):
    # "video"는 부드러운 trajectory를 따르는 1-D 값의 sequence다
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 2단계: 프레임별 position embedding

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 3단계: denoiser가 전체 sequence를 본다

각 프레임을 독립적으로 디노이즈하는 대신, 우리의 작은 네트워크는 모든 프레임 값과 그 position embedding을 concatenate하고 모든 프레임의 노이즈를 함께 예측한다.

### 4단계: temporal coherence 테스트

학습 후 비디오를 sample한다. frame-to-frame delta를 측정한다. 모델이 temporal structure를 학습했다면, 각 프레임을 독립적으로 sample할 때보다 delta가 더 작게 유지된다.

## 함정

- **프레임별 독립 샘플링 = flicker.** 각 프레임에 image diffusion을 따로 실행하면 각 프레임의 noise가 독립적이어서 출력이 깜빡인다. video diffusion은 attention 또는 shared noise로 프레임을 결합해 이를 해결한다.
- **순진한 3D attention = OOM.** 10초 1080p latent에 대한 full 3D attention은 수천억 연산이다. spatial + temporal로 factorize한다.
- **데이터 captioning은 크기보다 더 중요하다.** 이전 작업 대비 Sora의 주요 업그레이드는 훨씬 더 자세한 caption을 약 10배 많이 사용한 것이다(GPT-4가 clip을 다시 label). OpenAI의 technical report는 이 점을 명시한다.
- **First-frame conditioning.** 대부분의 프로덕션 모델은 이미지를 첫 프레임으로도 받는다. 이것이 "image-to-video" 모드이며, 학습에도 이 변형이 포함된다.
- **Physics drift.** 긴 클립(>10s)은 미묘한 불일치가 누적된다. sliding-window generation + keyframe anchoring이 도움이 된다.

## 사용하기

| 사용 사례 | 2026년 선택지 |
|----------|-----------|
| 최고 품질 hosted text-to-video | Veo 3 또는 Sora |
| 카메라 제어 cinematic | motion brush를 사용한 Runway Gen-3 |
| 클립 간 character consistency | Pika 2.0 또는 Kling 2.1 |
| Open weights, 빠른 fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V, Kling 2.1 I2V 또는 Runway |
| Audio-to-video lip sync | Veo 3(native audio) 또는 dedicated lip-sync model |
| Video editing | Runway Act-Two, Kling Motion Brush, Flux-Kontext(still-frame) |

품질이 같을 때 비디오 1초당 비용은 2024년에서 2026년 사이 20배 하락했다.

## 출시하기

`outputs/skill-video-brief.md`를 저장한다. 이 스킬은 비디오 brief(duration, aspect ratio, style, camera plan, subject consistency, audio)를 입력받아 다음을 출력한다. model + hosting, prompt scaffolding(camera language, subject description, motion descriptors), seed + reproducibility protocol, frame-level QA checklist.

## 연습문제

1. **쉬움.** `code/main.py`에서 (a) 프레임별 독립 샘플링, (b) joint sequence sampling의 frame-to-frame delta를 비교하라. delta의 평균과 분산을 보고하라.
2. **중간.** first-frame condition을 추가하라. frame 0을 주어진 값으로 고정하고 나머지를 sample하라. 고정된 값이 어떻게 전파되는지 측정하라.
3. **어려움.** HuggingFace diffusers로 로컬 GPU에서 CogVideoX-2B를 실행하라. 6초 클립에 대해 720p에서 inference 20단계 시간을 재라. 병목을 찾기 위해 spatiotemporal attention을 profile하라.

## 핵심 용어

| 용어 | 사람들이 말하는 방식 | 실제 의미 |
|------|-----------------|-----------------------|
| Video VAE | "3-D VAE" | `(T, H, W, C)`를 spatiotemporal latent로 압축하는 encoder. |
| Patches | "토큰" | latent의 고정 크기 3-D block이며 DiT 입력이다. |
| Factorized attention | "Spatial + temporal" | space에 대해 attention을 실행한 뒤 time에 대해 실행하고, full 3-D attention은 건너뛴다. |
| Image-to-video (I2V) | "이 사진을 움직이게 해줘" | 모델이 image + text를 받아 그 이미지에서 시작하는 video를 출력한다. |
| Keyframe conditioning | "Anchor frames" | 특정 프레임을 고정해 video의 arc를 제어한다. |
| Motion brush | "방향 힌트" | 사용자가 이미지 위에 motion vector를 칠하는 UI 입력. |
| Re-captioning | "Dense captions" | LLM으로 학습 clip에 detailed prompt를 다시 붙이는 것. |
| Flicker | "Temporal artifact" | 프레임 간 불일치이며 coupled denoising으로 고친다. |

## 프로덕션 노트: 비디오 latent는 memory-bandwidth 문제다

24 fps의 10초 1080p 클립은 240 frames × 1920 × 1080 × 3 ≈ 1.5 GB raw pixels다. 4× video VAE compression(`2 × spatial × 2 × temporal`) 뒤에도 latent는 요청당 약 100 MB다. 이것을 batch 1에서 30단계 spatiotemporal DiT에 통과시키면 HBM을 통해 단계당 약 3 GB를 이동한다. 병목은 FLOPs가 아니라 memory bandwidth다.

프로덕션 knob 세 가지는 모두 production-inference literature의 inference chapter에서 곧장 온다.

- **DiT 전반의 TP.** Text-to-video 모델은 보통 ≥10B params다. 4개 H100에 TP=4가 표준이며, 405B급 모델에는 PP=2 × TP=2를 쓴다. step당 latency는 all-reduce wall에 닿기 전까지 TP에 대체로 선형으로 감소한다.
- **Frame batching = continuous batching.** 생성 시 비디오는 개념적으로 attention으로 연결된 프레임 batch다. Continuous batching(in-flight scheduling)을 적용할 수 있다. 모델 아키텍처가 sliding-window generation을 허용한다면 frame `t-1`이 반환되는 동안 frame `t+1` 렌더링을 시작한다.
- **Clip-level prefill cache.** Image-to-video에서 first-frame conditioning은 LLM의 prompt prefill과 유사하다. 한 번 계산하고 temporal decoder pass에서 재사용한다. 이는 사실상 video용 KV-cache다.

## 더 읽을거리

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora technical report.
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX.
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo.
- [Genmo (2024). Mochi-1 Technical Report](https://www.genmo.ai/blog/mochi) — Mochi-1.
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025년 중반 open SOTA.
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 중요한 초기 video diffusion 논문.
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion의 선행 작업.
