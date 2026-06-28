---
name: mt-evaluator
description: shipping 가능한 machine translation output인지 평가합니다.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

source text와 candidate translation이 주어지면 다음을 출력하라.

1. 자동 점수 추정. 예상되는 BLEU와 chrF 범위. Reference 사용 가능 여부를 명시한다.
2. 인간이 검증할 수 있는 checklist 5개: content preservation(no hallucinations), correct target language, register / formality match, glossary가 제공된 경우 terminology consistency, truncation 또는 length explosion 없음.
3. 조사할 domain-specific issue 하나. Legal: named entities, statute citations. Medical: drug names, dosages. UI: `{name}` 같은 placeholder variables.
4. 신뢰도 flag. "Ship" / "Ship with review" / "Do not ship". 발견한 issue의 severity에 연결한다.

Output에 language-ID check가 없으면 배포를 거부하라. 사용자가 reference-free scoring(COMET-QE, BLEURT-QE)을 명시적으로 선택하지 않는 한 reference 없이 평가하지 말라. 1000 token을 넘는 content는 chunked translation이 필요할 가능성이 높다고 표시하라.
