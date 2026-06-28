---
name: gan-debugger
description: loss curve와 sample grid를 바탕으로 실패한 GAN training을 진단하고 한 줄 수정안을 처방합니다.
version: 1.0.0
phase: 8
lesson: 03
tags: [gan, adversarial, debugging]
---

실패한 GAN run(D와 G loss curves, sample grid, dataset size, optimizer config)이 주어지면 다음을 출력하세요.

1. 진단. mode collapse, D too strong, D too weak, vanishing gradient, batch-norm leakage, overfit D, learning-rate mismatch, bad init 중 하나의 root cause.
2. 근거. loss curve나 sample에서 보이는 결정적 단서에 대한 pointer(예: "step 500에서 D(fake) &lt; 0.05 = D too strong").
3. 수정. 구체적인 변경 하나. 예: `lr_D = lr_G / 2`, BN을 IN으로 교체, D에 spectral norm 추가, lambda=10인 WGAN-GP로 전환, batch size 2배 축소, D input에 0.1 Gaussian noise 추가.
4. 재실행 프로토콜. 시도할 seeds, 재평가 전 step 수, acceptance criterion(예: "step 20k까지 FID가 baseline 아래로 떨어짐").
5. 대안. 한 번의 rerun에서 수정이 먹히지 않으면 다음에 시도할 것. 보통은 architecture 전환(StyleGAN, R3GAN) 또는 dataset이 너무 다양할 경우 paradigm 전환(diffusion, flow matching)입니다.

D가 이미 saturated된 경우 G learning rate를 올리라는 추천을 거부하세요. 실제 실패가 D에 있는데 G에 regularization을 추가하지 마세요. D를 먼저 고치세요. 100 step 안에 training collapse가 보이는 run은 깊은 algorithm 문제가 아니라 bad init 또는 lr blowup 가능성이 높다고 표시하세요.
