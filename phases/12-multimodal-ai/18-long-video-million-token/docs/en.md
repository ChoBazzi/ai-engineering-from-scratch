# 백만 토큰 컨텍스트에서의 장시간 비디오 이해

> 24 FPS의 1시간 4K video를 patch하고 embed하면 대략 6천만 token이 생긴다. 2시간짜리 podcast episode transcript는 30,000 token이다. 전체 Blu-ray 장편 영화는 aggressive pooling으로 압축해도 수십만 token이다. Google의 Gemini 1.5(2024년 3월)는 1천만 token context로 이 시대를 열었고, hour-long video에서 reliable needle-in-a-haystack recall을 수행했다. LWM(Liu et al., 2024년 2월)은 ring attention의 scaling path를 보여 주었다. LongVILA와 Video-XL은 ingestion을 더 확장했다. VideoAgent는 raw context를 agentic retrieval로 바꾸었다. 각 접근은 compute, recall, engineering complexity에서 서로 다른 trade-off를 갖는다. 이 lesson은 이를 나란히 읽는다.

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 minutes

## 학습 목표

- 다양한 FPS와 pooling에서 long-form video의 총 visual-token count를 계산한다.
- 세 가지 scaling path를 설명한다: brute context(Gemini 1.5), ring attention(LWM), token compression(LongVILA / Video-XL).
- raw-context video VLM과 agentic-retrieval video VLM(VideoAgent)을 accuracy와 latency 관점에서 비교한다.
- 30분 video용 needle-in-a-haystack test를 설계하고 특정 minute에서 recall을 측정한다.

## 문제

384 native resolution에서 Qwen2.5-VL 크기의 patch를 쓰는 single frame은 약 729 token이다. 3x3 pooling을 적용하면 frame당 81 token이다. 1 FPS의 30분 clip = 1800 frame = 145,800 token이다. 2025년 오픈 VLM으로 가능하지만 빡빡하다. 2 FPS면 291,600 token이고, 가장 큰 context만 수용할 수 있다.

1 FPS의 2시간 영화는 583k token이다. 대부분의 2026년 오픈 모델 범위를 넘는다. Gemini 2.5 Pro가 필요하거나 pooling을 더 강하게 해야 한다.

세 가지 scaling path가 등장했다.

## 개념

### 경로 1: brute context(Gemini 1.5, Claude Opus)

문제에 hardware를 던진다. context를 수백만 token으로 키우고 모든 것을 한 번의 forward pass로 처리한다.

Gemini 1.5 Pro는 1M token으로 출시되었다. Gemini 1.5 Ultra는 10M까지 갔고, 2026년의 Gemini 2.5 Pro는 수 시간 video를 안정적으로 처리한다. 논문(arXiv:2403.05530)은 약 9.5M token까지 99.7%의 needle-in-a-haystack recall을 기록한다.

Engineering: memory hierarchy(local + global + sparse)를 갖춘 custom attention implementation과 long-context efficiency를 위한 MoE expert routing. 전체 상세는 공개되지 않았다. open-source도 아니다.

### 경로 2: ring attention(LWM, LongVILA)

Ring attention은 긴 sequence를 device 여러 대에 분산한다. 각 device는 chunk를 하나 들고, 전체 sequence에 대한 attention은 각 device가 자기 chunk를 ring pattern으로 다음 device에 보내며 partial attention을 계산하고 aggregation하는 방식으로 수행된다.

LWM(Liu et al., 2024)은 이 방식으로 1M-token context model을 학습했다. 학습 compute는 context에 대해 quadratic이 아니라 linear로 scale된다. attention의 quadratic hit가 ring의 device들에 걸쳐 amortize되기 때문이다.

LongVILA(arXiv:2408.10188)는 이 패턴을 VLM에 맞게 적용했다. frame당 192 token인 1400-frame video = 268k context를 8-way parallelism의 ring attention으로 학습했다.

### 경로 3: 토큰 압축(Video-XL, LongVA)

brute context보다 싸다. LLM이 sequence를 보기 전에 강하게 압축한다.

Video-XL(arXiv:2409.14485)은 visual summary token을 사용한다. N frame의 각 clip은 N에 attend하는 하나의 "summary" token을 만든다. 추론 때 LLM은 clip마다 summary token 하나만 보므로 context가 크게 줄어든다.

LongVA는 "long context transfer" 기법으로 LLM context를 200k에서 2M으로 확장한다. long-context text로 학습한 뒤, shared representation을 통해 long-context video로 전이한다.

Token compression은 특정 timestamp의 recall을 scalability와 맞바꾼다. 모델은 전반적으로 무슨 일이 있었는지는 알지만 exact frame을 놓치기도 한다.

### 경로 4: 에이전트형 검색(VideoAgent)

전체 video를 LLM에 넣지 않는다. 대신 video를 database처럼 다루고 LLM이 query하게 한다.

VideoAgent(arXiv:2403.10517):

1. LLM이 question을 읽는다.
2. LLM이 retrieval tool에 관련 clip을 요청한다("고양이가 있는 segment를 보여 줘").
3. Tool이 matching clip timestamp를 반환한다.
4. LLM이 VLM을 통해 해당 clip을 읽는다.
5. LLM이 answer를 구성하거나 follow-up query를 요청한다.

이것은 long video에 적용한 LLM-as-agent 패턴이다. 추론은 더 싸다(relevant clip만 encode). engineering은 더 어렵다(retrieval quality가 bottleneck이 된다).

### Needle-in-a-haystack 벤치마크

표준 long-context test는 video의 random point에 unique visual 또는 textual marker를 삽입한 뒤, 그것을 기억해야 답할 수 있는 query를 묻는 것이다.

Metric: video length와 marker position 전반의 Recall@k.

Gemini 2.5 Pro는 최대 90분 video에서 >99% recall을 기록한다. 오픈 72B 모델(Qwen2.5-VL-72B, InternVL3-78B)은 30분에서 약 85-90%이고, 60분을 넘으면 degrade된다.

VideoAgent는 tool이 needle을 잘 찾는다면 2시간 이상에서도 raw-context model과 맞먹거나 이길 수 있다.

### 어떤 path를 고를 것인가

frontier accuracy가 필요한 15분 clip: open 72B + native context가 보통 동작한다. Qwen2.5-VL-72B를 고른다.

30분-1시간 content: open에서는 LongVILA 또는 Video-XL, closed에서는 Gemini 2.5 Pro. quality bar가 중요하다면 frontier는 closed로 간다.

2시간 이상 content: VideoAgent 또는 유사 retrieval pattern. 대안으로 더 작은 chunk로 summarize하고 hierarchical summary를 넣는다.

### 2026 production pattern

실무 long-video pipeline은 보통 hybrid다.

1. 전체 video에 dynamic-FPS sampling + aggressive pooling을 실행한다(100k-token global representation 확보).
2. 72B VLM에 넣어 global summary를 만든다.
3. 사용자가 세부 질문을 하면 summary를 index로 사용해 agentic retrieval을 실행한다.

이는 global understanding에는 brute-context를, local detail에는 retrieval을 결합한다.

## 활용하기

`code/main.py`:

- 1분부터 3시간까지 video에 대해 다양한 FPS + pooling에서 token budget을 계산한다.
- needle-in-a-haystack run을 시뮬레이션한다. random timestamp에 marker를 주입하고, question을 묻고, recall을 score한다.
- 특정 clip을 downstream VLM에 넣도록 고르는 agentic-retrieval router simulator를 포함한다.

budget table을 실행하고 scale gap을 체감하라.

## 산출물

이 lesson은 `outputs/skill-long-video-strategy-planner.md`를 만든다. video duration과 query complexity가 주어지면 brute-context, compression, agentic retrieval 중에서 고르고 latency + quality expectation을 계산한다.

## 연습 문제

1. 45분 lecture, 1 FPS, frame당 81 token. 총 token은 얼마인가? 어떤 model context에 들어가는가?

2. needle-in-a-haystack test를 설계하라. marker를 몇 분 지점에 삽입하고, exact query format은 무엇인가?

3. 1시간 video에서 brute-context Qwen2.5-VL-72B(80k context)와 VideoAgent(Claude 3.5 + retrieval)를 비교하라. recall은 어느 쪽이 이기는가? latency는 어느 쪽이 이기는가?

4. Ring attention의 memory cost는 sequence length에 선형이고 device count에도 선형이다. 그 이유와 ring-rotation phase를 제거하면 무엇이 실패하는지 설명하라.

5. Gemini 1.5 Section 5의 needle-in-a-haystack을 읽어라. 논문은 1M vs 10M token boundary에서 recall에 대해 무엇을 발견했는가?

## 핵심 용어

| 용어 | 사람들이 부르는 말 | 실제 의미 |
|------|-----------------|------------------------|
| Brute context | "Just more tokens" | LLM context를 수백만 token까지 확장하고 모든 것을 한 번에 처리하는 방식 |
| Ring attention | "LWM-style parallel" | 각 device가 chunk를 들고 rotate하는 distributed attention pattern |
| Token compression | "Summary tokens" | LLM 전에 learned compressor로 clip당 token을 줄이는 방식 |
| Needle-in-haystack | "NIH test" | random point에 unique marker를 삽입하고 test time에 모델이 기억하는지 묻는 평가 |
| Agentic retrieval | "LLM as query planner" | LLM이 retrieval tool에 relevant clip을 요청하고 VLM으로 읽은 뒤 answer를 구성하는 방식 |
| VideoAgent | "Retrieval pattern for video" | question -> tool -> clip -> answer로 이어지는 canonical agentic-retrieval design |

## 더 읽을거리

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
