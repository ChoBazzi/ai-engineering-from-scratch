# 평가 - FID, CLIP Score, Human Preference

> 모든 생성 모델 리더보드는 FID, CLIP score, 그리고 human-preference arena의 win rate를 인용합니다. 각 숫자에는 의지가 있는 연구자가 악용할 수 있는 failure mode가 있습니다. 그 failure mode를 모르면 진짜 개선과 지표를 게임한 실행을 구분할 수 없습니다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 minutes

## 문제

생성 모델은 *sample quality*와 *conditioning adherence*로 평가됩니다. 둘 중 어느 것도 닫힌형식의 측정값이 없습니다. 모델은 10,000장의 이미지를 렌더링해야 하고, 무언가가 그 이미지들에 숫자를 매겨야 하며, 우리는 모델 계열, 해상도, 아키텍처가 달라도 그 숫자를 신뢰해야 합니다. 2014-2026년의 검증을 살아남은 metric은 세 가지입니다.

- **FID(Fréchet Inception Distance).** Inception network의 feature space에서 실제 분포와 생성 분포 사이의 거리입니다. 낮을수록 좋습니다.
- **CLIP score.** 생성 이미지의 CLIP-image embedding과 prompt의 CLIP-text embedding 사이 cosine similarity입니다. 높을수록 좋습니다. Prompt adherence를 측정합니다.
- **Human preference.** 같은 prompt에서 두 모델을 head-to-head로 겨루게 하고, 사람(또는 GPT-4급 모델)이 더 나은 결과를 고르게 한 뒤, Elo score로 집계합니다.

그 밖에도 IS(inception score, 대부분 폐기됨), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k를 보게 됩니다. 각각은 이전 metric의 한 가지 실패를 보정합니다.

## 개념

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID - 샘플 품질

Heusel et al. (2017). 절차는 다음과 같습니다.

1. N장의 실제 이미지와 N장의 생성 이미지에서 Inception-v3 feature(2048-D)를 추출합니다.
2. 각 pool에 Gaussian을 맞춥니다. 평균 `μ_r, μ_g`와 covariance `Σ_r, Σ_g`를 계산합니다.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

해석: feature space에서 두 multivariate Gaussian 사이의 Fréchet distance입니다. 낮을수록 분포가 더 비슷합니다.

Failure modes:
- **작은 N에서 편향됨.** FID는 feature distribution에 대한 mean-squared 방식입니다. N이 작으면 covariance가 과소추정되어 거짓으로 낮은 FID가 나옵니다. 항상 N ≥ 10,000을 사용하세요.
- **Inception에 의존함.** Inception-v3는 ImageNet으로 학습되었습니다. ImageNet과 멀리 떨어진 domain(얼굴, 예술, 텍스트 이미지)에서는 FID가 의미 없어집니다. Domain-specific feature extractor를 사용하세요.
- **Gaming.** Inception prior에 과적합하면 시각 품질 개선 없이 낮은 FID를 얻을 수 있습니다. 아래의 CMMD로 보완하세요.

### CLIP score - prompt 준수도

Radford et al. (2021). 생성 이미지 + prompt에 대해:

```text
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

생성 이미지 30k장에 대해 평균을 내면 모델 간 비교 가능한 scalar가 됩니다.

Failure modes:
- **CLIP 자체의 blind spots.** CLIP은 compositional reasoning이 약합니다("a red cube on a blue sphere" 같은 prompt가 자주 실패합니다). 모델은 복잡한 prompt를 실제로 따르지 않아도 CLIP score에서 좋은 순위를 받을 수 있습니다.
- **짧은 prompt 편향.** 짧은 prompt는 실제 웹 데이터에서 CLIP-image match가 더 많습니다. 긴 prompt는 구조적으로 CLIP score가 낮아집니다.
- **Prompt gaming.** Prompt에 "high quality, 4k, masterpiece"를 넣으면 image-text binding을 개선하지 않고도 CLIP score가 부풀려집니다.

CMMD(Jayasumana et al., 2024)는 이 중 일부를 고칩니다. Inception 대신 CLIP feature를 사용하고, Fréchet 대신 maximum-mean discrepancy를 사용합니다. 미묘한 품질 차이를 감지하는 데 더 낫습니다.

### Human preference - 기준 진실

Prompt pool을 고릅니다. Model A와 Model B로 생성합니다. 쌍을 사람(또는 강한 LLM judge)에게 보여주고, win을 Elo 또는 Bradley-Terry score로 집계합니다. Benchmark 예시는 다음과 같습니다.

- **PartiPrompts(Google)**: 12개 category의 다양한 prompt 1,600개.
- **HPSv2**: 사람 annotation 107k개, 자동 proxy로 널리 사용.
- **ImageReward**: prompt-image preference pair 137k개, MIT license.
- **PickScore**: Pick-a-Pic preference 2.6M개로 학습.
- **Chatbot-Arena-style image arenas**: https://imagearena.ai/ 등.

Failure modes:
- **Judge variance.** 비전문가와 전문가의 선호는 다릅니다. 둘 다 사용하세요.
- **Prompt distribution.** Cherry-picked prompt는 한 모델 계열에 유리합니다. 항상 문서화하세요.
- **LLM-judge reward hacking.** GPT-4 judge는 예쁘지만 틀린 출력에 속을 수 있습니다. 사람 평가와 삼각검증하세요.

## 함께 사용하기

프로덕션 eval report에는 다음이 포함되어야 합니다.

1. Held-out real distribution 대비 10-30k samples에서 FID(sample quality).
2. 같은 sample과 prompt에 대한 CLIP score / CMMD(adherence).
3. 이전 모델 대비 blinded arena win rate(overall preference).
4. Failure mode analysis: 무작위 sample 50개를 뽑아 알려진 문제(hand anatomy, text rendering, consistent object count)를 표시합니다.

단일 metric은 거짓말입니다. 서로 corroborate하는 metric 세 개 + qualitative review가 하나의 주장입니다.

## 직접 만들기

`code/main.py`는 synthetic "feature vectors"에서 FID, CLIP-score-like metric, Elo aggregation을 구현합니다. Inception feature의 stand-in으로 4-D vector를 사용합니다. 여기서 다음을 확인합니다.

- 작은 N과 큰 N에서의 FID 계산과 그 편향.
- Feature pool 사이 cosine similarity로 표현한 "CLIP score".
- Synthetic preference stream에서 Elo update rule.

### 1단계: 네 줄 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 2단계: CLIP-style cosine similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 3단계: Elo aggregation

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 함정

- **N=1000에서의 FID.** N=10k 미만에서는 heuristic이 신뢰하기 어렵습니다. Low-N FID를 보고하는 논문은 지표를 게임하고 있는 것입니다.
- **해상도를 넘나드는 FID 비교.** Inception의 299×299 resize는 feature distribution을 바꿉니다. 같은 해상도에서만 비교하세요.
- **Seed 하나만 보고.** 최소 3개 seed를 실행하세요. std를 보고하세요.
- **Negative prompt를 통한 CLIP score inflation.** 일부 pipeline은 prompt에 과적합해 CLIP을 끌어올립니다. 시각적 saturation이 있는지 확인하세요.
- **Prompt overlap에 따른 Elo 편향.** 두 모델 모두 benchmark prompt를 학습 중 봤다면 Elo는 무의미합니다. Held-out prompt set을 사용하세요.
- **Human eval paid-crowd skew.** Prolific, MTurk annotator는 더 젊고 기술 친화적인 쪽으로 치우칩니다. 모집한 예술/디자인 전문가와 섞으세요.

## 활용하기

2026년 프로덕션 eval protocol:

| 축 | 최소 기준 | 권장 기준 |
|--------|---------|-------------|
| Sample quality | Held-out real 대비 10k FID | + 5k CMMD + category별 subset FID |
| Prompt adherence | 30k CLIP score | + HPSv2 + ImageReward + VQA-style question answering |
| Preference | Baseline 대비 blinded pairs 200개 | + paired human 2000개 + LLM-judge + Chatbot Arena |
| Failure analysis | 사람이 표시한 50개 | 사람이 표시한 500개 + automated safety classifier |

네 pillar가 한 report 안에 있으면 주장입니다. 하나만 있으면 marketing입니다.

## 출시하기

`outputs/skill-eval-report.md`를 저장하세요. 이 skill은 새 model checkpoint + baseline을 받아 전체 eval plan을 출력합니다. Sample sizes, metrics, failure-mode probes, sign-off criteria를 포함합니다.

## 연습 문제

1. **쉬움.** `code/main.py`를 실행하세요. 같은 synthetic distributions에서 N=100과 N=1000의 FID를 비교하고 bias magnitude를 보고하세요.
2. **보통.** Synthetic CLIP-style feature에서 CMMD를 구현하세요(공식은 Jayasumana et al., 2024 참고). 품질 차이에 대한 민감도를 FID와 비교하세요.
3. **어려움.** HPSv2 setup을 재현하세요. Pick-a-Pic subset에서 1000개의 image-prompt pair를 가져와, preference로 작은 CLIP-based scorer를 fine-tune하고 held-out set과의 agreement를 측정하세요.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|-----------------|-----------|
| FID | "Fréchet Inception Distance" | Real vs gen Inception feature에 Gaussian fit을 한 뒤의 Fréchet distance입니다. |
| CLIP score | "Text-image similarity" | CLIP image embedding과 text embedding 사이의 cosine similarity입니다. |
| CMMD | "FID's replacement" | CLIP-feature MMD입니다. Bias가 적고 Gaussian assumption이 없습니다. |
| IS | "Inception score" | Exp KL(p(y\|x) \|\| p(y)); 최신 모델에서는 상관이 약해 폐기되었습니다. |
| HPSv2 / ImageReward / PickScore | "Learned preference proxies" | Human preference로 학습한 작은 모델이며 automatic judge로 사용됩니다. |
| Elo | "Chess rating" | Pairwise win의 Bradley-Terry aggregation입니다. |
| PartiPrompts | "The benchmark prompt set" | 12개 category에 걸친 Google-curated prompt 1,600개입니다. |
| FD-DINO | "Self-sup replacement" | DINOv2 feature를 쓰는 FD입니다. Out-of-ImageNet domain에 더 낫습니다. |

## 프로덕션 노트: 평가도 추론 워크로드입니다

10k samples에서 FID를 실행한다는 것은 이미지 10k장을 생성한다는 뜻입니다. 단일 L4에서 1024² 50-step SDXL base를 쓰면 single-request inference로 약 11시간이 걸립니다. 평가 예산은 실제 비용이며, 프레이밍은 정확히 offline-inference scenario입니다(TTFT는 무시하고 throughput을 최대화).

- **Batch를 크게 잡고 latency는 잊으세요.** Offline eval = memory에 들어가는 가장 큰 size로 static batching합니다. 80GB H100에서 `num_images_per_prompt=8`로 `pipe(...).images`를 실행하면 single-request보다 wall-clock 기준 4-6× 빠릅니다.
- **Real features를 cache하세요.** Real reference set에 대한 Inception(FID) 또는 CLIP(CLIP-score, CMMD) feature extraction은 *한 번만* 실행하고 `.npz`로 저장합니다. Eval마다 다시 계산하지 마세요.

CI / regression gate의 경우: PR마다 500-sample subset에서 FID + CLIP score를 실행하세요(약 30분). Full 10k FID + HPSv2 + Elo는 nightly로 실행하세요.

## 더 읽을거리

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) - FID paper.
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) - CMMD.
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) - CLIP.
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) - HPSv2.
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) - ImageReward.
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) - PartiPrompts.
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) - failure-mode survey.
