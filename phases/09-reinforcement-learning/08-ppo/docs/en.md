# Proximal Policy Optimization (PPO)

> A2C는 각 rollout을 한 번 update한 뒤 버린다. PPO는 policy gradient를 clipped importance ratio로 감싸서 policy가 폭주하지 않게 하면서 같은 데이터로 10+ epoch를 돌릴 수 있게 한다. Schulman et al. (2017). 2026년에도 여전히 기본 policy-gradient 알고리즘이다.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## 문제

A2C(Lesson 07)는 on-policy다. Gradient `E_{π_θ}[A · ∇ log π_θ]`는 *current* `π_θ`에서 샘플링한 데이터를 요구한다. Update를 한 번 하면 `π_θ`가 바뀐다. 방금 사용한 데이터는 이제 off-policy다. 재사용하면 gradient가 biased된다.

Rollout은 비싸다. Atari에서 8개 env × 128 step의 rollout 하나는 1024개 transition과 수십 초의 environment time을 뜻한다. Gradient step 하나 후에 이를 버리는 것은 낭비다.

Trust Region Policy Optimization(TRPO, Schulman 2015)이 첫 해결책이었다. old policy와 new policy 사이의 KL divergence가 `δ` 아래에 머물도록 각 update를 constrain한다. 이론적으로는 깔끔하지만 update마다 conjugate-gradient solve가 필요하다. 2026년에 TRPO를 돌리는 사람은 거의 없다.

PPO(Schulman et al. 2017)는 hard trust-region constraint를 단순한 clipped objective로 바꾼다. 코드 한 줄이 더 들어간다. Rollout당 10 epoch. Conjugate gradient 없음. 충분히 좋은 이론적 보장. 9년 뒤에도 MuJoCo부터 RLHF까지 모든 곳에서 기본 policy-gradient 알고리즘이다.

## 개념

![PPO clipped surrogate objective: 1 ± ε에서 ratio clipping](../assets/ppo.svg)

**Importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

이는 new policy와 데이터를 수집한 policy의 likelihood ratio다. `r_t = 1`이면 변화가 없다는 뜻이다. `r_t = 2`이면 new policy가 old policy보다 `a_t`를 선택할 가능성이 두 배라는 뜻이다.

**Clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

두 항이 있다.

- Advantage `A_t > 0`이고 ratio가 `1 + ε`를 넘어서 커지려 하면, clip이 gradient를 평평하게 만든다. 좋은 action의 확률을 old probability보다 `+ε` 이상 더 밀지 말라는 뜻이다.
- Advantage `A_t < 0`이고 ratio가 `1 - ε`를 넘어서 커지려 하면(나쁜 action을 clipped reduction에 비해 더 likely하게 만들겠다는 뜻), clip이 gradient를 cap한다. 나쁜 action을 `-ε`보다 아래로 더 밀지 말라는 뜻이다.

`min`은 반대 방향을 처리한다. Ratio가 *유리한* 방향으로 움직였다면 여전히 gradient를 받는다(손해가 되는 쪽만 clipping된다).

보통 `ε = 0.2`를 쓴다. `r_t`의 함수로 objective를 그리면 "좋은 쪽"에는 평평한 지붕이 있고 "나쁜 쪽"에는 평평한 바닥이 있는 piecewise-linear function이다.

**전체 PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

A2C와 같은 actor-critic 구조다. 세 coefficient는 보통 `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`를 쓴다.

**학습 loop.**

1. `N`개의 parallel env에서 각각 `T` step을 수집해 `N × T` transition을 만든다.
2. Advantage(GAE)를 계산하고 constant로 고정한다.
3. Current `π_θ`의 snapshot으로 `π_{θ_old}`를 고정한다.
4. `K` epoch 동안 `(s, a, A, V_target, log π_old(a|s))` minibatch마다:
   - `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`를 계산한다.
   - `L^{CLIP}` + value loss + entropy를 적용한다.
   - Gradient step을 수행한다.
5. Rollout을 버린다. 1단계로 돌아간다.

`K = 10`과 minibatch size 64는 표준 hyperparameter set이다. PPO는 robust하다. 정확한 숫자는 ±50% 안에서는 큰 문제가 되지 않는 경우가 많다.

**KL-penalty variant.** 원 논문은 adaptive KL penalty를 쓰는 대안도 제안했다. `L = L^{PG} - β · KL(π_θ || π_old)`이고, 관측된 KL에 따라 `β`를 조정한다. Clipping version이 지배적이 되었고, KL variant는 RLHF에서 살아남았다(reference policy에 대한 KL은 어차피 항상 원하는 별도 constraint이기 때문이다).

## 직접 만들기

### 1단계: rollout 시점에 `log π_old(a | s)` 캡처

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Snapshot은 rollout 시점에 한 번 찍는다. Update epoch 동안 변하지 않는다.

### 2단계: GAE advantage 계산(Lesson 07)

A2C와 같다. Batch 전체에서 normalize한다.

### 3단계: clipped surrogate update

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"clipped → zero gradient" 패턴이 PPO의 핵심이다. New policy가 이미 유리한 방향으로 너무 멀리 drift했다면 update가 멈춘다.

### 4단계: value와 entropy

A2C와 마찬가지로 critic target에 standard MSE를 더하고 actor에 entropy bonus를 더한다.

### 5단계: 진단 지표

Update마다 세 가지를 확인하라.

- **Mean KL** `E[log π_old - log π_θ]`. `[0, 0.02]` 안에 있어야 한다. `0.1`을 훌쩍 넘으면 `K_EPOCHS`나 `LR`을 줄여라.
- **Clip fraction** — ratio가 `[1-ε, 1+ε]` 밖에 있는 sample의 비율. `~0.1-0.3`이어야 한다. `~0`이면 clip이 전혀 작동하지 않는 것이므로 `LR` 또는 `K_EPOCHS`를 올려라. `~0.5+`이면 rollout에 overfit하고 있으므로 낮춰라.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Critic quality metric이다. Critic이 학습되면 1에 가까워져야 한다.

## 함정

- **Clip coefficient mistuned.** `ε = 0.2`가 사실상의 표준이다. `0.1`로 가면 update가 너무 소심하고, `0.3+`는 불안정을 부른다.
- **Too many epochs.** `K > 20`은 policy가 `π_old`에서 멀리 drift하므로 자주 불안정해진다. 특히 큰 네트워크에서는 epoch를 제한하라.
- **No reward normalization.** 큰 reward scale은 clip range를 잠식한다. Advantage를 계산하기 전에 reward를 normalize하라(running std).
- **Forgetting advantage normalization.** Batch별 zero-mean/unit-std normalization은 표준이다. 빼먹으면 대부분 benchmark에서 PPO가 망가진다.
- **Learning rate not decayed.** PPO는 linear LR decay to zero에서 이득을 본다. Constant LR은 종종 더 나쁘다.
- **Importance ratio math errors.** Numerical stability를 위해 항상 `exp(log_new - log_old)`를 써라. `new / old`가 아니다.
- **Wrong gradient sign.** Surrogate를 maximize한다는 것은 `-L^{CLIP}`를 *minimize*한다는 뜻이다. 부호를 뒤집는 것이 가장 흔한 PPO bug다.

## 활용하기

PPO는 2026년에 놀라울 만큼 많은 domain에서 기본 RL 알고리즘이다.

| 사용 사례 | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | Gaussian policy, GAE(0.95)를 쓰는 PPO |
| Atari / discrete games | categorical policy와 rolling 128-step rollout을 쓰는 PPO |
| LLM용 RLHF | reference model에 대한 KL penalty와 response 끝 RM reward를 쓰는 PPO |
| Large-scale game agents | IMPALA + PPO(AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO(Lesson 12) — critic 없는 PPO variant |
| preference-only data | DPO — PPO+KL의 closed-form collapse, online sampling 없음 |

PPO의 *loss shape*, 즉 clipped surrogate + value + entropy는 DPO, GRPO, 거의 모든 RLHF pipeline의 scaffolding이다.

## 산출물

`outputs/skill-ppo-trainer.md`로 저장하라.

```markdown
---
name: ppo-trainer
description: 주어진 환경을 위한 PPO training config와 diagnostic plan을 만든다.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

환경과 training budget이 주어지면 다음을 출력하라.

1. Rollout 크기. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate parameter. `ε`(clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. 명시적 `γ`와 `λ`를 포함한 GAE(`λ`).
5. 진단 계획. KL, clip fraction, explained variance threshold와 alert.

`K > 30` 또는 `ε > 0.3`은 거부하라(unsafe trust region). Advantage normalization 또는 KL/clip monitoring이 없는 PPO run은 거부하라. Clip fraction이 지속적으로 0.4를 넘으면 drift로 표시하라.
```

## 연습문제

1. **쉬움.** `ε=0.2, K=4`로 4×4 GridWorld에서 PPO를 실행하라. Env step 수를 맞춰 A2C(rollout당 one epoch)와 sample efficiency를 비교하라.
2. **보통.** `K ∈ {1, 4, 10, 30}`를 sweep하라. Return vs env step을 그리고 update마다 mean KL을 추적하라. 이 task에서 어떤 `K`에서 KL이 폭주하는가?
3. **어려움.** Clipped surrogate를 adaptive KL penalty(`KL > 2·target`이면 `β`를 두 배, `KL < target/2`이면 절반)로 바꿔라. Final return, stability, clip-free-ness를 비교하라.

## 핵심 용어

| 용어 | 사람들이 하는 말 | 실제 의미 |
|------|------------------|-----------|
| Importance ratio | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`. 데이터를 수집한 policy에서 얼마나 벗어났는지. |
| Clipped surrogate | "PPO의 핵심 요령" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`. 유리한 쪽에서 clip을 지나면 gradient가 평평해진다. |
| Trust region | "TRPO / PPO의 의도" | 각 update의 KL을 제한해 monotone improvement를 보장한다. |
| KL penalty | "soft trust region" | 대안 PPO: `L - β · KL(π_θ \|\| π_old)`. Adaptive `β`. |
| Clip fraction | "clipping이 얼마나 자주 일어나는가" | Diagnostic. 0.1-0.3이어야 하며, 벗어나면 tuning이 잘못된 것이다. |
| Multi-epoch training | "data 재사용" | 각 rollout에서 K epoch를 돈다. Variance 비용을 sample efficiency와 맞바꾼다. |
| On-policy-ish | "대체로 on-policy" | PPO는 명목상 on-policy지만 K>1 epoch에서 약간 off-policy인 데이터를 안전하게 사용한다. |
| PPO-KL | "다른 PPO" | KL-penalty variant. KL-to-reference가 이미 constraint인 RLHF에서 쓰인다. |

## 더 읽을거리

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 이 논문.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, PPO의 predecessor.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 모든 PPO hyperparameter ablation.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT. PPO-in-RLHF recipe.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — PyTorch가 포함된 깔끔한 현대적 설명.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 여러 논문에서 쓰는 reference single-file PPO.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — language model용 PPO production recipe. Lesson 09(RLHF)와 함께 읽어라.
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — "37 code-level optimizations" 논문. 어떤 PPO 요령이 load-bearing이고 어떤 것이 folklore인지 다룬다.
