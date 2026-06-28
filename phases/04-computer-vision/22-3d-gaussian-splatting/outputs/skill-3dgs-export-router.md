---
name: skill-3dgs-export-router
description: downstream viewer 또는 engine에 따라 올바른 3DGS export format(.ply / .splat / glTF KHR_gaussian_splatting / USD) 선택
version: 1.0.0
phase: 4
lesson: 22
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# 3DGS Export Router

downstream target을 올바른 3DGS file format에 map한다. "it does not load" debugging에 쓰는 시간을 줄인다.

## 사용할 때

- 3DGS scene을 training한 뒤 content pipeline과 공유하기 전에.
- research-grade(.ply)와 production-grade(glTF / USD) format 사이를 고를 때.
- Pipeline handoff: capture team -> 3DGS engineer -> game designer / VFX artist / web developer.

## 입력

- `target_engine`: unreal | unity | omniverse | blender | vision_pro | three_js | babylon_js | cesium | playcanvas | supersplat
- `priority`: portability | file_size | quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## Format decision

| Target | Recommended format | Why |
|--------|--------------------|-----|
| Unreal Engine (virtual production) | Volinga plugin or glTF KHR_gaussian_splatting | Native Unreal SDK path |
| Unity (XR / game) | .ply via Aras-P Unity-GaussianSplatting plugin | Community-standard Unity pipeline |
| NVIDIA Omniverse, Pixar tools | OpenUSD 26.03 (UsdVolParticleField3DGaussianSplat) | Native USD prim type |
| Apple Vision Pro | OpenUSD 26.03 | visionOS 2.x native |
| Blender | .ply + KIRI Engine add-on | Community add-on이 raw splat을 읽음 |
| Three.js web viewer | glTF KHR_gaussian_splatting or .splat | Browser-standard, `GaussianSplats3D`와 동작 |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9에서 native support 추가 |
| Cesium (CesiumJS 1.139+, Cesium for Unreal 2.23+) | glTF KHR_gaussian_splatting | Explicit support shipped |
| PlayCanvas | .splat | PlayCanvas native quantised format |
| SuperSplat (editor) | .ply or .splat | Import + export |

## Quantisation trade-offs

- `.ply` full-precision: 가장 큰 file, lossless, 모든 viewer.
- `.splat`: 4x-8x 더 작고, SH3 coefficient에 약간의 quality loss가 있으며, PlayCanvas ecosystem standard.
- glTF KHR: EXT_meshopt_compression으로 configurable하다. 가장 작고 compatibility가 높다.
- USD: USDZ packaging으로 compressed된다. Apple pipeline에서 가장 작다.

## 출력 report

```text
[export plan]
  target:         <engine>
  format:         <name>
  sh degree:      <0|1|2|3>
  compression:    <none|meshopt|quantisation|usdz>
  expected size:  <MB>
  compatible with: <viewer list>

[pipeline]
  1. source: <.ply from training>
  2. optional: SuperSplat cleanup pass
  3. convert: <tool + CLI or API call>
  4. package: <.gltf / .glb / .usd / .usdz / .splat / .ply>
  5. validate: <viewer sanity check>
```

## 규칙

- SH3 coefficient를 silently strip하지 않는다. specular reflection이 눈에 띄게 바뀐다.
- `priority == file_size`이면 `.splat` 또는 meshopt를 사용한 glTF를 추천하고 quality loss를 경고한다.
- Apple platform에서는 2026년 기준 glTF보다 USD / USDZ를 선호한다. USDZ는 visionOS support가 first-class다.
- target viewer의 3DGS support가 pre-standard(pre-Feb 2026)이면 `.ply`와 viewer의 custom loader를 추천한다. Khronos-standard glTF는 아직 인식되지 않을 수 있다.
- handoff 전에 exported file을 적어도 하나의 viewer에서 항상 validate한다. quantisation 중 silent corruption이 발생할 수 있다.
