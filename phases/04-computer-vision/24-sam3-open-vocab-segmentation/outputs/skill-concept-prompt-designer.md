---
name: skill-concept-prompt-designer
description: 사용자 발화를 splitting, disambiguation, fallbacks를 갖춘 잘 구성된 SAM 3 concept prompt로 변환
version: 1.0.0
phase: 4
lesson: 24
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# Concept Prompt 설계기

SAM 3의 정확도는 concept prompt가 어떻게 표현되는지에 크게 좌우됩니다. 이 skill은 자유 형식의 사용자 발화를 SAM 3가 잘 처리하는 prompt로 정규화합니다.

## 사용할 때

- 자연어 object query를 받는 UI를 만들 때.
- upstream caller가 문장을 보내는 API를 통해 SAM 3를 노출할 때.
- SAM 3 match 품질이 나쁜 이유를 디버깅할 때. 문제는 모델이 아니라 malformed prompt인 경우가 많습니다.

## 입력

- `utterance`: 원본 사용자 문자열.
- `context`: 선택적 domain hint(예: "surveillance", "medical", "retail").
- `max_concepts`: utterance당 추출할 최대 concepts 수. 기본값 5.

## SAM 3가 선호하는 규칙

- **문장이 아니라 짧은 noun phrases.** `"there is a cat"`보다 `"cat"`이 낫습니다.
- **구체적인 명사.** `"thing to ride on"`보다 `"skateboard"`가 낫습니다.
- **modifier는 noun 바로 앞에.** `"car that is red"`보다 `"red car"`가 낫습니다.
- **소문자.** SAM 3는 robust하지만 경험적으로 lowercase input에서 약간 더 좋습니다.
- **단수 또는 복수.** 둘 다 동작합니다. 여러 instance가 예상될 때는 plural이 도움이 됩니다.

## 단계

1. **일반 구분자로 tokenise** — comma, semicolon, "and", "or", "&".
2. **filler prefix 제거** — "find", "show me", "segment", "detect", "locate", "a", "an", "the".
3. **prepositional modifier는 시각적인 경우에만 유지** — `"striped red umbrella"`는 yes, `"umbrella from yesterday"`는 no입니다(`"from yesterday"`는 image 안에 있지 않습니다).
4. **선택적 `context`로 collision을 disambiguate**:
   - surveillance context의 `"window"` -> `"building window"`.
   - medical context의 `"window"` -> 보통 오류입니다. 사용자에게 명확히 해 달라고 제안하세요.
5. splitting 결과 concept이 0개이고 utterance에 구체 명사가 하나 이상 있으면 exact string으로 **fallback**합니다. 추출 가능한 구체 명사가 없으면 concept을 emit하지 마세요. warnings만 반환하고 사용자에게 명확히 해 달라고 요청하세요(규칙 참고).
6. **`max_concepts`에서 자르세요.** 추출된 concept이 caller가 요청한 것보다 많으면 utterance 순서의 처음 `max_concepts`만 유지하고 나머지는 reason `"exceeded max_concepts"`와 함께 `dropped`에 넣으세요. 사용자가 긴 열거를 붙여 넣어도 latency가 bounded됩니다.

## 출력 형식

```text
[designed prompts]
  utterance:    <original>
  concepts:     ["concept_1", "concept_2", ...]
  dropped:      ["filler_1", ...]
  warnings:     ["concept too abstract", "may match many classes", ...]

[sam3 calls]
  For each concept run: sam3.detect(image, concept)
  Merge outputs with distinct concept tags per detection.
```

## 예시

```text
in:  "can you find me a cat or two dogs?"
out: ["cat", "dogs"]
dropped: ["can you find me", "a", "or two", "?"]
note: "dogs" kept plural because the utterance says "two dogs" — plural hint preserved.

in:  "segment the big red truck and the blue sedan"
out: ["big red truck", "blue sedan"]
dropped: ["segment", "the", "and"]

in:  "thing near the door"
out: ["door"]
warnings: ["'thing' is too abstract for SAM 3; fell back to 'door'"]

in:  "striped red umbrella, green hat, pink balloon"
out: ["striped red umbrella", "green hat", "pink balloon"]
```

## 규칙

- 8단어보다 긴 문장을 SAM 3에 전달하지 마세요. 그 이상에서는 정확도가 떨어집니다.
- utterance에 추출 가능한 구체 명사가 없으면 SAM 3를 실행하지 마세요. warnings를 반환하고 clarification을 요청하세요.
- quoted string 안의 punctuation에서는 split하지 마세요. quote된 경우 `"black and white cat"`을 하나의 concept으로 보존하세요.
- 프로덕션 디버깅을 위해 original utterance와 derived concepts를 항상 log하세요.
