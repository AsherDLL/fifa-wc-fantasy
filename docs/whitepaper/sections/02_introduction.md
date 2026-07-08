# 02 - Introduction

Status: **DRAFT**

## 2.1 The fantasy points prediction problem

Fantasy football games translate match events into a numeric score per
player per match. The FIFA Fantasy World Cup 2026 game follows the
familiar Premier League Fantasy template: 15-player squad under a budget
cap, 11-player starting XI in a valid formation, a captain whose points
double, automatic substitutions for non-starters, and per-stage rules on
transfers and country counts. The full rule set is documented in the
game's official guidelines and reproduced for reference in `docs/Fantasy.md`.

The user's objective is to maximise total fantasy points across the
tournament, often within a private league of friends or colleagues that
adds a ranking objective on top of raw scoring. Two of those objectives
diverge in non-trivial ways: an entry far behind the field benefits from
**variance** in differential picks even at the cost of expected value,
while an entry near the lead benefits from **template** picks that lock
in the field's expected output.

## 2.2 Cross-domain transfer: club football to international tournaments

Predicting fantasy points for international football has structurally
less prior data than predicting Premier League fantasy points:

- A World Cup is 64 matches over five weeks; an EPL season is 380
  matches over ten months.
- Players assemble into national teams for ~10 days a year. Their
  international form is observed sparsely.
- The opponent distribution is unlike any club league: Argentina plays
  Cape Verde, not Cape Verde's domestic-league third-tier equivalent.
- The matchday cadence in the group stage exposes rotation behaviour
  that does not exist in single-leg knockout play: a coach whose team
  has already qualified rests starters in matchday three.

A model trained purely on club football transfers imperfectly. The
features that drive club-football fantasy points (opponent strength,
home advantage, position-specific scoring rates) generalise; the
features that drive international-football fantasy points
(manager-specific rotation, top-scorer-race incentives, knockout-stage
attacking emphasis) do not exist in the club training set.

## 2.3 Contributions

This work makes the following technical contributions:

1. **A three-backend ensemble** combining a hand-tuned heuristic, a
   structural Poisson goals model, and a LightGBM regressor trained on
   three seasons of EPL fantasy data. Each backend's per-position RMSE
   is measured on a held-out EPL 2024-25 GW 30-38 split, allowing an
   honest per-position best-backend choice.

2. **A live international Elo signal** computed from the full
   martj42/international_results corpus (every international match since
   1872) with FIFA-style tournament weights and goal-margin
   adjustments. Country-name normalisation between martj42, FIFA
   Fantasy, and football-data.co.uk uses a small canonical lookup so the
   signal joins cleanly into the per-(player, round) feature table.

3. **A MILP optimiser** (PuLP with the CBC backend) that handles squad
   selection, stage-aware transfer planning with the official -3
   per-extra-transfer hit, and starting XI choice including formation
   and captain. Constraints encode the stage-by-stage rule set: budget
   ($100M group stage, $105M R32+), country cap (3 to 8 depending on
   stage), and free-transfer budget with roll-over.

4. **An honest empirical study** of the system's live decisions over six
   matchdays of the WC 2026, including a transfer log with rationale
   and post-hoc outcome. We document specifically where the model's
   recommendation was overridden by user domain prior (and whether that
   was correct), and where the model's recommendation was followed
   reluctantly (and what happened).

## 2.4 Roadmap

Section 3 covers background. Section 4 documents the four data sources.
Section 5 develops each predictor backend formally. Section 6 describes
the implementation. Section 7 reports held-out RMSE on the EPL split.
Section 8 logs the live tournament results match-by-match. Sections 9
through 11 contain the analysis, lessons, and future-work discussion.
