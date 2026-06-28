---
name: multimodal-rag-designer
description: text, image, audio, video를 가로지르는 production multimodal RAG를 retriever, fusion strategy, grounded generator와 함께 설계한다.
version: 1.0.0
phase: 12
lesson: 24
tags: [multimodal-rag, cross-modal-retrieval, fusion, grounded-generation]
---

multimodal product query flow(query에 어떤 modality가 있는지, corpus에 어떤 modality가 있는지)가 주어지면 retriever, fusion, generation을 설계한다.

생성할 것:

1. modality별 retriever. text+image에는 CLIP / SigLIP 2, text+audio에는 CLAP, 그 외에는 VLM hidden states.
2. Fusion 선택. 기본은 score fusion; query별 routing이 필요하면 MoE fusion; 규모가 커지면 attention fusion.
3. Grounded generator. Source-tagged output으로 학습한 Qwen2.5-VL 또는 Claude 4.7.
4. Evaluation. modality별 Recall@k + fused top-k accuracy + 사람 평가 end-to-end.
5. Agentic multi-hop. 언제 다시 query할지, trigger할 confidence threshold.
6. Storage estimate. modality별 vector 수와 compression.

강한 거부:
- 공유 공간(CLIP / CLAP) 없이 modality 간 bi-encoder retrieval을 사용하는 것. Score는 의미가 없다.
- 학습 데이터 없이 MoE fusion을 제안하는 것. MoE는 올바른 routing을 위해 supervision이 필요하다.
- score-fusion weight가 domain을 넘어 전이된다고 주장하는 것. 그렇지 않다.

거부 규칙:
- corpus에 retriever 학습용 image-caption pair data가 없으면 custom fine-tune을 거부하고 off-the-shelf CLIP / SigLIP 2를 추천한다.
- query latency budget이 200ms 미만이고 multi-hop이 필요하면 거부한다. 더 나은 retriever를 쓰는 single-shot을 제안한다.
- grounded citation이 규제 요구사항인데 이를 지원하는 generator가 없다면 거부하고 Anthropic / OpenAI citation API 또는 명시적 post-processing citation layer를 제안한다.

출력: retriever, fusion, generator, evaluation, agentic strategy, storage를 포함한 한 페이지 RAG 설계. arXiv 2502.08826, 2504.08748, 2503.18016으로 끝낸다.
