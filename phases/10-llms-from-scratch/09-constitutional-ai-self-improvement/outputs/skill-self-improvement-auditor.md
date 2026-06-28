---
name: self-improvement-auditor
description: Constitutional AI 또는 self-improvement pipeline을 scale에서 실행하기 전에 감사합니다.
version: 1.0.0
phase: 10
lesson: 9
tags: [alignment, cai, grpo, rlhf, self-improvement, reward-hacking]
---

Constitutional AI, RLAIF, GRPO, 또는 어떤 형태의 self-generated preference data를 사용한다고 주장하는 training pipeline이 주어지면 다음을 포함한 감사를 작성하세요.

1. Reward rule. 정확한 verifier(regex, sympy, test suite, LLM judge)를 명시하세요. deterministic, stochastic-LLM, hybrid 중 하나로 분류하세요. 외부 grounding이 없는 "self-improvement" loop는 거부하세요. 모델은 아무 데서도 signal을 끌어낼 수 없습니다.
2. Group statistics. GRPO pipeline에서는 group size, advantage 계산 방식(z-score vs relative rank), group reward std가 0으로 붕괴될 때의 처리를 확인하세요. pipeline은 zero-variance group을 skip하거나 downweight해야 하며, epsilon으로 나누고 signal이 진짜인 척해서는 안 됩니다.
3. KL budget. 실행 전체의 누적 KL(policy || reference)에 대한 numeric cap을 두세요. cap에 도달하면 pipeline은 중단하거나 reset하거나 더 따뜻한 reference로 전환해야 합니다. 무제한 KL은 무제한 drift입니다.
4. Diversity floor. task가 허용하는 것 중 per-group reward std, response length variance, n-gram entropy의 측정된 lower bound를 두세요. floor가 N round 연속 깨지면 pipeline은 새 human data 또는 더 넓은 prompt distribution을 섞어야 합니다.
5. Human data quota. training mix에 남아 있어야 하는 human-authored data의 최소 비율을 정하세요. 보통 5-10%입니다. self-distillation-only pipeline은 3-5 round 뒤 붕괴합니다. 이를 명시적으로 지적하세요.
6. Mode-collapse watchdog. round별 reward std, held-out prompt의 unique n-gram count, length distribution, refusal rate 같은 automatic check를 표시하세요. 이 중 하나라도 threshold를 넘으면 training을 중단합니다.
7. Constitution drift. CAI pipeline에는 versioned constitution file, changelog, 그리고 edit 사이에서 expected behavior가 바뀌면 안 되는 prompt로 구성된 "constitutional regression test set"을 요구하세요.

다음 pipeline은 승인하지 마세요.

- 외부 verifier(rule, tool, environment) 없이 "zero human data"를 주장하는 경우.
- process-reward hacking probe 없이 PRM을 사용하는 경우. 모델이 proof를 진전시키지 않으면서 그럴듯해 보이는 step을 쓰는지 확인해야 합니다.
- held-out diversity benchmark 없이 rejection-sampling fine-tuning을 5 round 넘게 실행하는 경우.
- reference model을 policy와 공유하는 경우. reference가 없으면 KL이 없고, KL이 없으면 anchor가 없습니다.
- policy와 같은 모델인 LLM judge로 채점하는 경우(judge contamination).

출력: gate별 pass/fail, 측정 또는 명시된 값, 각 signal을 만들어 내는 pipeline의 정확한 step을 담은 한 페이지 audit을 작성하세요. gate가 하나라도 실패하면 pass로 바꾸기 위한 minimum viable change를 나열하세요.
