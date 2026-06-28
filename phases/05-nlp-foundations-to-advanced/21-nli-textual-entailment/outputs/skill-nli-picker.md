---
name: nli-picker
description: classification / faithfulness / zero-shot task를 위한 NLI model, label template, evaluation setup을 고릅니다.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

use case(faithfulness check, zero-shot classification, document-level inference)가 주어지면 다음을 출력하세요.

1. 모델. 이름이 있는 NLI checkpoint. domain, length, language와 연결된 이유.
2. 템플릿(zero-shot인 경우). verbalization pattern. 예시.
3. 임계값. decision rule을 위한 entailment cutoff. calibration에 근거한 이유.
4. 평가. held-out labeled set의 accuracy, hypothesis-only baseline, adversarial subset.

100-example labeled sanity check 없이 zero-shot classification을 출시하지 마세요. document-length premise에 sentence-level NLI model을 사용하지 마세요. NLI가 hallucination을 해결한다는 claim은 flag하세요. NLI는 hallucination을 줄일 뿐, 제거하지는 않습니다.
