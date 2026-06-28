---
name: prompt-data-quality-checker
description: LLM pre-training pipeline의 data quality 검증과 debug
version: 1.0.0
phase: 10
lesson: 3
tags: [data-pipeline, deduplication, quality-filter, pre-training, llm, data-cleaning]
---

# LLM Pre-Training용 Data Quality Checker

LLM pre-training용 data pipeline을 구축하거나 audit할 때, 문제가 모델에 도달하기 전에 이 프레임워크로 잡아내세요.

## Pipeline 출력의 위험 신호

**Deduplication이 web data의 20% 미만만 제거함.** Common Crawl에는 보통 30-40%의 duplicate가 있습니다. dedup 단계가 20% 미만을 제거했다면 MinHash parameter가 너무 보수적이거나 threshold가 너무 높습니다. 확인 항목: shingle size k, hash function 수, LSH band 수, Jaccard threshold.

**Compression ratio가 2.0 chars/token 미만.** tokenizer가 너무 공격적으로 분리한다는 뜻입니다. merge 수를 늘려 재학습하거나 vocabulary 크기를 키우거나 pre-tokenization이 텍스트를 불필요하게 조각내지 않는지 확인하세요.

**Compression ratio가 6.0 chars/token 초과.** tokenizer가 일반화되지 않을 수 있는 매우 domain-specific한 merge를 학습했다는 뜻입니다. domain-specific model에는 괜찮지만 general-purpose model에는 경고 신호입니다.

**Sequence utilization이 90% 미만.** padding이 너무 많습니다. 문서가 매우 짧거나(필터링하거나 minimum document length를 늘리세요) sequence packing이 비효율적입니다(naive padding에서 multi-document packing으로 전환하세요).

**Vocab utilization이 50% 미만.** 이 corpus에서 vocabulary의 절반 이상이 사용되지 않습니다. domain에 비해 vocabulary가 너무 크거나 tokenizer가 매우 다른 데이터로 학습된 것입니다.

## Quality Filter 보정

각 pipeline stage에서 1,000개 문서의 random sample로 다음 검사를 실행하세요.

1. **cleaning 후 random document 20개를 읽습니다.** 잔여 HTML, JavaScript, navigation text, boilerplate가 있나요? 그렇다면 HTML stripping이 불완전합니다.

2. **quality filter를 통과한 random document 20개를 읽습니다.** spam, keyword list, machine-generated 문서가 있나요? 그렇다면 filter threshold를 더 엄격하게 하세요.

3. **quality filter에 실패한 random document 20개를 읽습니다.** 실제로 좋은 콘텐츠가 있나요? 그렇다면 filter가 너무 공격적입니다. threshold를 완화하거나 특정 pattern에 대한 예외를 추가하세요.

4. **dedup에서 나온 random near-duplicate pair 20개를 읽습니다.** 실제로 유사한가요? 그렇지 않다면 Jaccard threshold를 낮추거나 hash function 수를 늘리세요.

## Data Mixing 비율

보편 공식은 없습니다. 다음 baseline에서 시작해 evaluation 결과에 따라 조정하세요.

| Category | Llama 3 비율 | 시작점 |
|----------|--------------|----------------|
| Web text | 50% | 50% |
| Code | 25% | 15-25% |
| Books/academic | 13% | 10-15% |
| Math | 8% | 5-10% |
| Multilingual web | 4% | 5-10% |

모델이 programming에 강해야 한다면 code 비율을 높이세요. reasoning이 중요하면 math 비율을 높이세요. noise를 줄여야 한다면 web 비율을 낮추세요. 비율을 바꾼 뒤에는 항상 benchmark로 평가하세요.

## Scaling 추정

주어진 목표 token 수에 대해:

- web에서 1T tokens: raw text 약 3-5TB, cleaning과 dedup 후 약 1.5-2TB 예상
- Tokenization speed (Rust): core당 약 100M tokens/second
- Tokenization speed (Python): core당 약 1-10M tokens/second
- 128 hashes, 16 bands의 MinHash dedup: core당 약 10K documents/second
- Sequence packing: I/O bound이므로 10GB 초과 corpus에는 memory-mapped file 사용

15T tokens(Llama 3 scale)에는 raw input data 약 30-50TB, 64-core machine에서 1-2주 preprocessing, intermediate file용 100TB 이상의 disk를 계획하세요.

## Training 전 체크리스트

1. 총 token 수가 compute budget과 맞습니다(Chinchilla scaling 또는 Llama 3 overtrain ratio를 기준으로 사용)
2. dedup이 web data의 30-40%를 제거했습니다
3. quality filter가 남은 데이터의 10-20%를 제거했습니다
4. 영어 기준 compression ratio가 3-5 chars/token입니다
5. sequence utilization이 95% 이상입니다
6. random spot-check에서 모든 pipeline stage의 텍스트가 깨끗하고 일관됩니다
7. data mix ratio가 small-scale training run에서 검증되었습니다
8. PII removal이 sample에서 검증되었습니다
9. 모든 binary format(packed sequences, token ID arrays)이 round-trip encoding/decoding test를 통과합니다
10. pipeline이 재현 가능합니다: fixed random seed에서 같은 입력이 동일한 출력을 만듭니다
