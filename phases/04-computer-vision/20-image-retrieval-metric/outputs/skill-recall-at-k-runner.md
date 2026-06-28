---
name: skill-recall-at-k-runner
description: train/val/gallery split과 proper data contract를 갖춘 recall@K clean evaluation harness 작성하기
version: 1.0.0
phase: 4
lesson: 20
tags: [retrieval, evaluation, recall, faiss]
---

# Recall@K Runner

query 및 gallery image folder와 label을 reproducible recall@K number로 바꿉니다.

## 사용 시점

- new backbone의 첫 retrieval benchmark를 만들 때.
- fine-tune epoch 전반의 embedding quality를 tracking할 때.
- 같은 dataset에서 두 retrieval system을 비교할 때.

## Inputs

- `query_images`: path list.
- `gallery_images`: path list(query와 overlap할 수도 있고 아닐 수도 있음).
- `query_labels`, `gallery_labels`: class 또는 instance ID.
- `encoder_fn`: callable `image -> embedding`(precomputed 또는 live).
- `ks`: `[1, 5, 10]` 같은 list.

## Steps

1. 모든 gallery image를 한 번 encode합니다. numpy array로 저장합니다.
2. 모든 query image를 encode합니다.
3. 두 embedding set을 모두 L2-normalise합니다.
4. 각 query에 대해 모든 gallery item과의 similarity를 계산합니다.
5. 내림차순으로 sort하고 top max(ks)를 가져옵니다.
6. 각 K에 대해 top-K gallery item 중 query의 label을 공유하는 것이 있는지 확인합니다.
7. `recall@K = fraction of queries that had at least one correct neighbour in top K`를 report합니다.

## Output template

```python
import numpy as np
from sklearn.preprocessing import normalize

def encode_all(images, encoder_fn, batch=32):
    out = []
    for i in range(0, len(images), batch):
        embs = encoder_fn(images[i:i + batch])
        out.append(embs)
    return np.concatenate(out)


def recall_at_k(query_emb, gallery_emb, q_labels, g_labels,
                ks=(1, 5, 10), query_ids=None, gallery_ids=None):
    if len(query_emb) == 0 or len(gallery_emb) == 0:
        return {f"recall@{k}": 0.0 for k in ks}

    g_label_set = set(g_labels.tolist())
    keep = np.array([lbl in g_label_set for lbl in q_labels])
    if not keep.any():
        return {f"recall@{k}": 0.0 for k in ks}

    q_emb_f = query_emb[keep]
    q_lab_f = q_labels[keep]
    q_id_f = query_ids[keep] if query_ids is not None else None

    q = normalize(q_emb_f)
    g = normalize(gallery_emb)
    sims = q @ g.T

    if q_id_f is not None and gallery_ids is not None:
        self_mask = q_id_f[:, None] == gallery_ids[None, :]
        sims = np.where(self_mask, -np.inf, sims)

    top_k_max = min(max(ks), g.shape[0])
    if top_k_max <= 0:
        return {f"recall@{k}": 0.0 for k in ks}

    top = np.argpartition(-sims, top_k_max - 1, axis=1)[:, :top_k_max]
    sorted_top = np.take_along_axis(
        top, np.argsort(-sims[np.arange(len(q))[:, None], top], axis=1), axis=1
    )
    out = {}
    for k in ks:
        k_eff = min(k, top_k_max)
        hits = np.any(g_labels[sorted_top[:, :k_eff]] == q_lab_f[:, None], axis=1)
        out[f"recall@{k}"] = float(hits.mean())
    return out


def evaluate(query_images, query_labels, gallery_images, gallery_labels, encoder_fn, ks=(1, 5, 10)):
    q_emb = encode_all(query_images, encoder_fn)
    g_emb = encode_all(gallery_images, encoder_fn)
    return recall_at_k(q_emb, g_emb, np.array(query_labels), np.array(gallery_labels), ks)
```

## Report

```text
[evaluation]
  num queries:   <int>
  num gallery:   <int>
  embedding_dim: <int>

[recall]
  recall@1:  <float>
  recall@5:  <float>
  recall@10: <float>
```

## Rules

- similarity를 계산하기 전에 embedding을 normalise합니다. normalised vector에서 FAISS IndexFlatIP는 cosine과 같습니다.
- query의 ground-truth label이 gallery에 없으면 제외합니다. 그렇지 않으면 recall이 trivial하게 1 아래로 cap됩니다.
- query와 gallery가 overlap하면 query 자체를 자신의 top-K에서 제외합니다. 그렇지 않으면 retrieval이 아니라 self-similarity를 측정합니다.
- `num_queries > 10,000`이면 OOM을 피하기 위해 similarity matmul을 batch 처리합니다.
