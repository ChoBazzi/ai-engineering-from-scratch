# Figure Index

`site/assets/figures/` 아래에 포함된 모든 figure를 아래에 나열합니다. FIG 번호는 전역 번호이며, 단조 증가하고, 재사용하지 않습니다.

미학은 이 repo와 별도로 배포되는 `blueprint-diagram` Claude Code skill에 문서화되어 있습니다(프로젝트의 "repo에 vendor/tooling artifact를 넣지 않는다" 규칙에 따름). 설치 후 skill source는 `~/.claude/skills/blueprint-diagram/` 아래에 있습니다. 설치 경로는 maintainer에게 문의하거나, skill 없이 수동으로 진행하려면 아래 [추가 방법](#추가-방법) 섹션을 따르세요.

| FIG | slug | phase | lesson | added | notes |
|---|---|---|---|---|---|
| 000 | (curriculum stack - README banner에 embedded) | - | - | 2026-05-09 | hero, 이 dir이 아니라 `assets/banner.svg`에 있음 |
| 001 | exploded-view-floppy | - | - | 2026-05-09 | skill reference example, `~/.claude/skills/blueprint-diagram/references/examples/` 아래에 있음 |
| 001.A | prompts | - | - | 2026-05-13 | README "every lesson ships something" card - prompt artifact icon |
| 001.B | skills | - | - | 2026-05-13 | README card - SKILL.md drop-in icon |
| 001.C | agents | - | - | 2026-05-13 | README card - ReAct-style agent loop icon |
| 001.D | mcp-servers | - | - | 2026-05-13 | README card - tool/resource/prompt가 있는 MCP server rack icon |
| 002 | kernel-surface-gaussian | - | - | 2026-05-09 | skill reference example |
| 003 | pixel-vector-bezier | - | - | 2026-05-09 | skill reference example |
| 004 | gaussian-kernel-blur | 1 | 8 | 2026-05-09 | "Optimization: Gradient Descent Family" lesson용 gaussian blur visualization |
| 005 | transformer-attention-heads | 7 | 1 | 2026-05-09 | multi-head attention block의 exploded view |

## 번호 체계

- `001`-`099`: 초기 curriculum figure용 예약 범위(Phases 0-7).
- `100` 이상: 작성 순서대로 배정.
- Sub-figure는 letter suffix를 사용합니다: `004.A`, `004.B`. parent row를 공유합니다.

## 추가 방법

`blueprint-diagram` skill이 설치되어 있다면:

1. concept description으로 skill을 실행합니다.
2. skill은 SVG를 `site/assets/figures/NNN-slug.svg`에 쓰고, 다음 사용 가능한 번호로 여기에 row를 추가하며, 요청받은 경우 관련 lesson markdown에 `![FIG_NNN](path)`로 figure를 연결합니다.

skill이 없다면 수동으로 진행하세요.

1. cream + blueprint aesthetic으로 SVG를 작성합니다(cream `#fafaf5` paper, `#3553ff` blueprint blue stroke, leader line이 있는 JetBrains Mono uppercase label, 다른 chromatic accent 없음).
2. 위 table의 다음 사용 가능한 FIG 번호를 사용해 `site/assets/figures/<NNN>-<slug>.svg`로 저장합니다.
3. FIG number, slug, target phase + lesson, today's date, 한 줄 note를 담은 row를 여기에 추가합니다.
4. lesson markdown에서 `![FIG_NNN](../../site/assets/figures/<NNN>-<slug>.svg)`로 figure를 참조합니다.
5. 480 / 720 / 1200 px viewport width에서 검증합니다. label은 geometry와 겹치면 안 되고, leader line은 target에 닿아야 합니다.

## 라이선스

Figure는 repo의 MIT license로 배포됩니다. MIT license는 source SVG 배포 시 copyright notice 보존을 요구합니다. 렌더링된 이미지의 시각적 재사용(예: blog post나 slide deck에 embed)은 추가 attribution 없이 가능합니다.
