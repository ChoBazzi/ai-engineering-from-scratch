---
name: skill-image-text-retriever
description: 임의의 CLIP checkpoint로 이미지 임베딩 index를 만들고 query-by-text와 query-by-image 지원
version: 1.0.0
phase: 4
lesson: 18
tags: [clip, retrieval, faiss, zero-shot]
---

# Image-Text Retriever

CLIP embedding을 사용해 이미지 폴더를 검색 가능한 index로 바꿉니다.

## 사용 시점

- 내부 catalog에 zero-shot image search를 구축할 때.
- embedding distance로 거의 동일한 이미지를 deduplicate할 때.
- 라벨 데이터셋 없이 빠른 "find similar" component를 만들 때.

## 입력

- `image_folder`: 이미지 파일 디렉터리.
- `clip_model`: `openai/clip-vit-base-patch32` 또는 `google/siglip-base-patch16-224` 같은 HuggingFace id.
- `index_type`: flat | IVF | HNSW.
- `embedding_dim`: 모델에서 추론합니다.

## 단계

1. CLIP 모델과 preprocessor를 로드합니다.
2. 폴더의 모든 이미지를 batch encode합니다. embedding을 (N, D) float32 + filename list로 저장합니다.
3. embedding 위에 FAISS index를 만듭니다. cosine similarity를 위해 L2-normalised vector에 inner-product를 사용합니다.
4. 두 query interface를 노출합니다:
   - `search_by_text(text, k)` — 텍스트를 임베딩하고 검색합니다.
   - `search_by_image(image_path, k)` — 이미지를 임베딩하고 검색합니다.

## 출력 템플릿

```python
import os
import glob
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import faiss


class ImageTextRetriever:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim
        self.index = None
        self.filenames = []

    @torch.no_grad()
    def _encode_images(self, paths, batch=16):
        embs = []
        for i in range(0, len(paths), batch):
            imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
            inputs = self.processor(images=imgs, return_tensors="pt")
            out = self.model.get_image_features(**inputs)
            out = out / out.norm(dim=-1, keepdim=True)
            embs.append(out.cpu().numpy())
        return np.concatenate(embs).astype(np.float32)

    @torch.no_grad()
    def _encode_text(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        out = self.model.get_text_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu().numpy().astype(np.float32)

    def build_index(self, folder, index_type="flat"):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(folder, ext)))
        self.filenames = sorted(files)
        embs = self._encode_images(self.filenames)
        if index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = min(256, max(4, len(embs) // 32))
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist)
            self.index.train(embs)
        elif index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs)

    def search_by_text(self, text, k=5):
        q = self._encode_text([text])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]

    def search_by_image(self, image_path, k=5):
        q = self._encode_images([image_path])
        dist, idx = self.index.search(q, k)
        return [(self.filenames[i], float(d)) for d, i in zip(dist[0], idx[0])]
```

## 보고

```text
[retriever]
  model:          <name>
  num_images:     <int>
  dim:            <int>
  index_type:     flat | IVF | HNSW
  index_size_mb:  <float>
```

## 규칙

- Indexing 전에 embedding을 항상 L2-normalise하세요. Normalised vector에 대한 FAISS inner product는 cosine similarity와 같습니다.
- 이미지가 < 100k이면 `IndexFlatIP`(exact)가 가장 단순하고 빠릅니다.
- 100k-10M에는 `IndexIVFFlat`이 표준 trade-off입니다.
- > 10M에는 HNSW 또는 product-quantised variant를 사용하세요.
- query마다 index를 다시 만들지 마세요. 한 번 embedding하고 여러 번 검색하세요.
