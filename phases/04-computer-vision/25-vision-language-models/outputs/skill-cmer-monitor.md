---
name: skill-cmer-monitor
description: Cross-Modal Error Rate monitoring, dashboards, alerts로 프로덕션 VLM endpoint를 계측합니다
version: 1.0.0
phase: 4
lesson: 25
tags: [vlm, production, monitoring, hallucination]
---

# CMER 모니터

cross-modal alignment를 일급 프로덕션 KPI로 다루세요.

## 언제 사용할까

- 이미지에 grounded된 텍스트를 생성하는 VLM endpoint를 배포할 때.
- 환각된 응답 보고를 조사할 때.
- 입력 분포 변화가 모델 grounding을 악화시키는지 추적할 때.

## 입력

- `vlm_output`: 생성된 텍스트.
- `text_confidence`: softmax 이후 평균 per-token probability, `[0, 1]` 범위. `exp(mean(log_probs))`로 계산합니다. raw logits를 넘기지 마세요. raw logits는 unbounded이고 `conf_threshold`는 probability를 가정합니다.
- `image_embedding`: 이미지의 CLIP 계열 embedding(DINOv3, SigLIP, CLIP).
- `text_embedding`: 생성된 텍스트의 CLIP 계열 embedding.
- 선택적 `prompt_type`: 그룹화를 위한 label(vqa / ocr / captioning / agent).

## 요청별 계산

```python
import torch

def cmer_flag(image_emb, text_emb, text_conf, sim_thr=0.25, conf_thr=0.8):
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"emb shape mismatch: {image_emb.shape} vs {text_emb.shape}")
    image_emb = image_emb / (image_emb.norm() + 1e-8)
    text_emb = text_emb / (text_emb.norm() + 1e-8)
    sim = float((image_emb * text_emb).sum())
    flagged = (text_conf > conf_thr) and (sim < sim_thr)
    return {"sim": sim, "flagged": flagged}
```

Embeddings는 독립적인 CLIP 계열 encoder에서 나온 1-D PyTorch tensors(`torch.float32`)입니다. NumPy arrays를 사용한다면 `.norm()`을 `np.linalg.norm(...)`으로 바꾸고 output을 그에 맞게 cast합니다.

`sim`, `text_conf`, `flagged`, `prompt_type`, `timestamp`, `model_version`, `request_id`를 monitoring pipeline(Prometheus, DataDog, OpenTelemetry)에 저장합니다.

## 집계 지표

```text
CMER = (flagged requests in window) / (total requests in window)
```

endpoint별, prompt_type별, model version별로 보고합니다.

## 알림 threshold

- Baseline CMER: 정상 트래픽 7일 동안 설정합니다.
- Warning: 1시간 동안 CMER >= baseline의 1.5배.
- Critical: 30분 동안 CMER >= baseline의 2배 또는 어떤 window든 절대값 > 15%.

## 대시보드 패널

1. 시간에 따른 CMER(5분 bucket, 7일 window).
2. prompt_type별 CMER(stacked bar).
3. 시간별 `sim` 분포(histogram).
4. 최상위 hallucinated outputs(사람 검토를 위해 하루에 flagged responses 20개 샘플링).

## CMER가 급등할 때의 조치

1. flagged requests를 샘플링합니다.
2. model version이 의도치 않게 바뀌지 않았는지 확인합니다.
3. input distribution을 확인합니다(new file format? new image source? differently compressed?).
4. spike가 해결될 때까지 영향을 받는 traffic을 human review로 라우팅합니다.
5. spike가 지속되면 모델을 파인튜닝하거나 교체합니다. alert를 억누르지 마세요.

## 규칙

- VLM 자체 embeddings로 CMER를 계산하지 마세요. 독립적인 encoder(DINOv3, SigLIP 또는 CLIP-L/14)를 사용합니다. 그렇지 않으면 alignment가 아니라 모델의 self-consistency를 측정하게 됩니다.
- `flagged` bit만이 아니라 raw `sim` 값을 항상 log합니다. distribution shift는 flag rate가 바뀌기 전에 lower quartile에서 먼저 나타납니다.
- CMER monitoring 없이 VLM endpoint를 출하하지 마세요. 환각은 지배적인 프로덕션 실패 모드이며 이 지표 없이는 조용히 발생합니다.
- 민감한 도메인(medical, legal, financial)에서는 `sim_threshold`를 0.35 이상으로 올립니다. flag 조건은 `sim < sim_threshold`이므로 threshold가 높을수록 잠재적으로 ungrounded인 output을 더 많이 잡습니다. high-stakes 사용 사례에 맞는 기본값입니다.
