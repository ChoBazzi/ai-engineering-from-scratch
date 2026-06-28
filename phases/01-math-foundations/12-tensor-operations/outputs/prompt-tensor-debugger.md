---
name: prompt-tensor-debugger
description: 딥러닝 코드의 텐서 shape 오류를 단계별로 디버깅하는 프롬프트
phase: 1
lesson: 12
---

내 딥러닝 코드에 텐서 shape 오류가 있습니다. 고칠 수 있게 도와주세요.

**오류 메시지:** [여기에 오류를 붙여넣기]

**내 tensor shapes:**
- [이름]: [shape]
- [이름]: [shape]

**하려는 연산:** [설명하기]

---

디버깅할 때는 이 정확한 절차를 따르세요.

**Step 1: 연산 type 식별하기.**
어떤 연산이 오류를 만들었나요? 다음 중 하나로 매핑하세요.
- Matrix multiply / Linear layer (inner dimensions must match)
- Broadcasting (align from right, each dim must be equal or 1)
- Concatenation (all dims match except the cat dimension)
- Convolution (expects specific rank and channel position)
- Reshape (total elements must be preserved)

**Step 2: shape contract 작성하기.**
식별한 연산에 대해 기대되는 shape를 명시적으로 쓰세요.
```text
matmul(A, B): A is (..., m, k), B is (..., k, n) -> (..., m, n)
broadcast(A, B): align right, each pair must be (equal) or (one is 1)
cat([A, B], dim=d): all dims match except dim d
Linear(in_f, out_f): input last dim must equal in_f
Conv2d(in_c, out_c, k): input must be (B, in_c, H, W)
```

**Step 3: mismatch 찾기.**
실제 shape를 contract와 비교하세요. 규칙을 위반하는 정확한 차원을 찾아내세요.

**Step 4: 최소 수정 선택하기.**
이 표에서 고르세요.

| 증상 | 수정 |
|---|---|
| batch 차원 누락 | `.unsqueeze(0)` |
| channel 차원 누락 | `.unsqueeze(1)` |
| 불필요한 size-1 차원 | `.squeeze(dim)` |
| matmul의 inner dims가 틀림 | `.transpose(-1, -2)` 또는 weight shape 확인 |
| NHWC에서 NCHW가 필요함 | `.permute(0, 3, 1, 2)` |
| NCHW에서 NHWC가 필요함 | `.permute(0, 2, 3, 1)` |
| linear를 위해 spatial dims를 flatten | `.flatten(1)` 또는 `.reshape(B, -1)` |
| heads 분리: (B,T,D)에서 (B,H,T,D/H) | `.reshape(B, T, H, D//H).transpose(1, 2)` |
| heads 병합: (B,H,T,D/H)에서 (B,T,D) | `.transpose(1, 2).reshape(B, T, H*(D//H))` |
| .view()와 non-contiguous tensor | `.contiguous().view(...)` 또는 `.reshape(...)` 사용 |

**Step 5: 수정 검증하기.**
각 단계의 결과 shape를 보여주세요. reshape 전후 전체 원소 수가 보존되는지 확인하세요. 연산의 shape contract가 이제 만족되는지 확인하세요.

**Step 6: silent bugs 확인하기.**
shape가 맞더라도 다음을 확인하세요.
- Broadcasting이 의도한 축을 따라 일어나는지(우연히가 아니라)
- Reduction이 올바른 차원에서 합산하는지
- batch 차원(dim 0)이 전체 forward pass 동안 유지되는지
- 차원 순서가 중요할 때 단순 reshape가 아니라 transpose + reshape를 쓰는지

응답 형식은 다음과 같이 하세요.
```text
OPERATION: [실패한 연산]
EXPECTED: [shape contract]
ACTUAL: [제공된 shape]
MISMATCH: [어느 차원인지, 왜인지]
FIX: [정확한 코드]
RESULT: [수정 후 shape]
```
