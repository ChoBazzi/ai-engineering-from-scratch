---
name: 3d-pipeline
description: 입력 유형, 출력 형식, 사용 사례에 따라 3D generation 또는 reconstruction pipeline을 선택한다.
version: 1.0.0
phase: 8
lesson: 12
tags: [3d, gaussian-splatting, nerf, mesh]
---

입력(text prompt / one image / few images / photo capture / video), 목표 출력(mesh / Gaussian splat / NeRF / point cloud), 사용 사례(real-time render, game engine, AR / VR, cinematic)가 주어지면 다음을 출력한다.

1. 파이프라인. (a) Multi-view diffusion + 3D fit(SV3D, CAT3D + 3DGS), (b) direct single-shot(LRM, TripoSR, InstantMesh), (c) PBR을 갖춘 text-to-mesh(Meshy 4, Rodin Gen-1.5, Hunyuan3D 2.0), (d) photo capture + 3DGS(Gsplat, Postshot, Scaniverse).
2. Base model + hosting. 이름 있는 model + open / hosted. commercial use에 대한 license relevance를 포함한다.
3. 반복 예산. 첫 output까지 예상 시간, iteration cost, refinement strategy.
4. Topology + materials. Remesh pass가 필요한가? PBR channel requirements(albedo, roughness, metallic, normal)? UV layout은 automated인가 manual인가?
5. 평가. Held-out views의 SSIM, CLIP score, mesh watertightness, poly count, texture resolution.
6. 플랫폼 target. Unity / Unreal / Blender / web(three.js / Babylon) / AR(USDZ / glb).

mesh conversion pass 없이 3DGS를 game engine에 바로 출시하는 것을 거부한다(대부분의 engine은 splat을 native로 render하지 않는다). 복잡한 articulated characters에는 text-to-3D를 거부하고, 대신 rigging-aware pipeline을 사용한다. downstream tool이 NeRF를 render할 수 없을 때는 NeRF-only output을 flag한다(대부분의 DCC tools).
