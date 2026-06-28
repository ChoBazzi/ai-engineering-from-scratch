# 오디오-언어 모델: Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> 2026년 audio-language model은 speech + environmental sound + music을 함께 추론한다. Qwen2.5-Omni-7B는 MMAU-Pro에서 GPT-4o Audio와 맞먹는다. Audio Flamingo Next는 LongAudioBench에서 Gemini 2.5 Pro를 앞선다. 오픈과 폐쇄 모델의 격차는 사실상 닫혔다. 단, multi-audio task에서는 모두 거의 random에 가깝다.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 minutes

## 문제

5초짜리 오디오가 있다. 개가 짖고, 누군가 "stop!"이라고 외친 뒤 침묵한다. 유용한 질문은 여러 축에 걸쳐 있다.

- **전사.** "무슨 말을 했나?": ASR 영역이다.
- **의미 추론.** "그 사람이 위험한가?": 짖음 + 외침 + 침묵을 함께 이해해야 한다.
- **음악 추론.** "어떤 악기가 melody를 연주하나?"
- **긴 오디오 검색.** "이 90분 lecture에서 강사가 gradient descent를 설명한 부분은 어디인가?"

하나의 prompt로 이 모든 것에 답하는 단일 모델이 **audio-language model**(LALM / ALM)이다. 순수 ASR과는 다르다. LALM은 transcript만이 아니라 자유 형식의 자연어 답변을 생성한다.

## 개념

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 세 구성요소 템플릿

2026년의 모든 LALM은 같은 뼈대를 갖는다.

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · 또는 모델별 custom encoder.
2. **Projector.** audio-encoder feature를 LLM token embedding space로 연결하는 linear 또는 MLP.
3. **LLM.** Llama / Qwen / Gemma 기반 decoder. interleaved text + audio token을 받아 text를 생성한다.

학습:

- **학습 1단계.** encoder + LLM을 freeze하고 ASR / captioning data에서 projector만 학습한다.
- **학습 2단계.** instruction-following audio task(QA, reasoning, music understanding)로 full / LoRA fine-tune한다.
- **학습 3단계(선택).** voice-in / voice-out은 speech decoder를 추가한다. Qwen2.5-Omni와 AF3-Chat이 이렇게 한다.

### 2026 모델 지도

| 모델 | Backbone | 오디오 encoder | 출력 방식 | 접근 방식 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### 벤치마크 현실 점검(2026)

**MMAU-Pro.** speech / sound / music / mixed를 다루는 1800개 QA pair. Multi-audio subset 포함.

| 모델 | 전체 | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | n/a | n/a | n/a | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | n/a | n/a | n/a | n/a |
| Audio Flamingo Next | LongAudioBench SOTA | n/a | n/a | n/a | n/a |

**multi-audio 열은 모두에게 혹독하다.** 4지선다 random chance는 25%이고, 대부분 모델은 그 근처다. LALM은 아직 두 clip을 비교하는 데 어려움을 겪는다.

### 2026년에 LALM이 유용한 곳

- **콜센터 녹음 compliance audit.** "상담원이 필수 고지를 말했나?"
- **접근성.** 청각 장애 사용자에게 sound event를 설명한다(전사만이 아니다).
- **Content moderation.** 폭력적 언어 + 위협적인 tone + background context를 감지한다.
- **Podcast / meeting chaptering.** 단순 speaker turn이 아니라 의미 기반 요약.
- **음악 catalog 분석.** "B-section key change가 있는 모든 track을 찾아라."

### 아직 유용하지 않은 곳

- 세밀한 음악 이론(chord-level 아래).
- 긴 대화에서 speaker attribution이 필요한 추론(10분을 넘기면 저하).
- Multi-audio comparison(22-26%는 random보다 아주 조금 높은 수준).
- Real-time streaming reasoning(대부분은 offline batch inference).

## 직접 만들기

### 1단계: Qwen2.5-Omni 질의하기

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 2단계: projector 패턴

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

전부 이것이다. projector는 보통 1-3개의 linear layer다. ASR pair(audio → transcript)로 학습시키는 것이 Stage-1 pretext task다.

### 3단계: MMAU / LongAudioBench 벤치마크

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

category별(speech / sound / music / multi-audio)로 따로 보고하라. 집계 수치는 모델이 어디서 실패하는지 숨긴다.

## 활용하기

| 작업 | 2026년 선택 |
|------|-----------|
| 자유 형식 audio QA(오픈) | Qwen2.5-Omni-7B |
| 긴 오디오에서 최고의 오픈 모델 | Audio Flamingo Next |
| 최고의 폐쇄 모델 | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni 또는 GPT-4o Audio |
| 음악 추론 | Audio Flamingo 3 또는 2(music-specialized AF-CLAP) |
| 콜센터 audit | Gemini 2.5 Pro via API, policy doc 위 RAG 포함 |

## 함정

- **Multi-audio 과신.** 작업이 "어느 clip에 X가 있는가"를 요구한다면 random-chance 수준 성능이 현실이다.
- **긴 오디오 성능 저하.** 10분을 넘기면 대부분 모델의 speaker attribution이 무너진다. 먼저 diarize하고(Lesson 6), 그다음 요약하라.
- **침묵에서의 hallucination.** Whisper encoder를 쓰는 LALM이 물려받는 동일한 Whisper식 문제다. VAD-gate하라.
- **Benchmark cherry-picking.** Vendor blog post는 best-case category를 강조한다. MMAU-Pro multi-audio subset을 직접 실행하라.

## 출시하기

`outputs/skill-alm-picker.md`로 저장하라. 주어진 audio-understanding task에 대해 LALM + benchmark subset + output-modality(text vs speech)를 선택한다.

## 연습문제

1. **쉬움.** `code/main.py`를 실행해 toy projector pattern과 가짜 LALM routing((audio-embedding, text-tokens) → output tokens)을 확인하라.
2. **중간.** Qwen2.5-Omni-7B를 MMAU-Pro speech item 100개에서 채점하라. 논문에 보고된 수치와 비교하라.
3. **어려움.** 최소 audio-captioning baseline을 만들어라: BEATs encoder + 2-layer projector + frozen Llama-3.2-1B. AudioCaps에서 projector만 fine-tune하라. Clotho-AQA에서 SALMONN과 비교하라.

## 핵심 용어

| 용어 | 사람들이 말하는 것 | 실제 의미 |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | audio feature를 LLM embedding space로 mapping하는 작은 MLP. |
| MMAU | Benchmark | speech, sound, music 전반의 10k audio-QA pair. |
| MMAU-Pro | 더 어려운 MMAU | multi-audio / reasoning-heavy question 1800개. |
| LongAudioBench | Long-form eval | 의미 query가 있는 multi-minute clip. |
| Voice-in / voice-out | Speech-native | 모델이 text 우회 없이 speech를 ingest하고 speech를 emit한다. |

## 더 읽을거리

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759): reference architecture.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B): speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128): 오픈 long-audio leader.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905): LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289): dual-encoder pioneer.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/): 2026년 live ranking.
