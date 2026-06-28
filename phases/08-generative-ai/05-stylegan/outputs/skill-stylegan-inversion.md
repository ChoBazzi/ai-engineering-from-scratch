---
name: stylegan-inversion
description: 실제 사진에 대해 pretrained StyleGAN의 inversion 및 editing pipeline을 선택합니다.
version: 1.0.0
phase: 8
lesson: 05
tags: [stylegan, inversion, editing]
---

실제 사진 + pretrained StyleGAN checkpoint(FFHQ-1024, StyleGAN-XL, custom fine-tune)와 목표 편집(age, smile, pose, hair, identity preservation)이 주어지면 다음을 출력합니다.

1. Inversion 방법. e4e(빠름, 낮은 fidelity), ReStyle(iterative encoder), HyperStyle(hypernet), PTI(pivotal tuning), 또는 direct W-optimization. Fidelity와 speed의 trade-off에 연결된 한 문장 이유를 포함합니다.
2. Target space. W, W+, 또는 StyleSpace. Trade-off: W = 가장 disentangled되어 있지만 fidelity가 가장 낮음, W+ = layer별 w, StyleSpace = channel-level.
3. 편집 방향. 이름 있는 direction source: InterFaceGAN(SVM 기반), StyleSpace channel, GANSpace PCA, 또는 learned classifier.
4. Fidelity 예산. Identity drift 전 LPIPS threshold와 rollback heuristic.
5. 평가. ID similarity(ArcFace cosine), 원본 대비 LPIPS, edit strength(target attribute classifier score).

Z에서 직접 편집하는 pipeline은 거부합니다(entangled). Identity check 없이 W에서 큰 편집(&gt;1.5 sigma)은 거부합니다. Open-domain editing이 필요한 요청(예: "make him a cartoon")은 표시하세요. 그런 요청에는 StyleGAN이 아니라 diffusion + IP-Adapter가 필요합니다.
