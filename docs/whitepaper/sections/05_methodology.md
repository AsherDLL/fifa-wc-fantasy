# 05 — Methodology

Status: **DRAFT**

## 5.1 Per-(player, round) feature contract

The feature table is built in `src/fifa_fantasy/features/build.py`. One
row per (player, upcoming round in which their squad plays). The
columns fall into five groups:

**Player metadata:** `player_id`, `full_name`, `position`, `country`,
`country_abbr`, `squad_id`, `price_millions`, `ownership_fraction`,
`status`, `is_eliminated`, `one_to_watch`, `total_points`,
`last_round_points`, `form`, `round_points`.

**Fixture context:** `fixture_id`, `round_id`, `stage`, `is_home`,
`kickoff`, `opponent_squad_id`, `opponent_name`, `opponent_abbr`,
`venue_name`, `venue_city`.

**Own-squad strength:** `squad_total_price`, `squad_avg_price`,
`squad_top_n_avg_price`, `squad_top_n_rank`, `squad_rank_points`,
`squad_rank_position`. The `top_n_avg_price` is the mean price of the
top 11 (configurable) players per squad, used as a proxy for first-team
strength.

**Opponent strength:** identical columns prefixed `opp_`.

**Derived:** `strength_diff` (own top-n minus opp top-n), `rank_diff`
(FIFA ranking gap), `country_elo`, `opp_country_elo`,
`country_elo_diff`, `country_last10_form`, `opp_country_last10_form`,
`days_since_prev_match`, `days_to_next_match`.

The feature table is intentionally over-wide; each backend consumes the
subset it needs.

## 5.2 Backend 1: hand-tuned heuristic

A hand-written rule of thumb. The formula:

> predicted_points = points_per_price_unit[position] * price_millions
>                  * (1 + alpha * tanh(combined_diff))
>                  * (1 + beta * is_home)
>                  + premium_boost * max(0, price_millions - threshold)

`points_per_price_unit` is per-position (GK 0.50, DEF 0.55, MID 0.60,
FWD 0.65). `combined_diff` blends three z-scored signals with priority:

> z_elo = country_elo_diff / 400
> z_rank = rank_diff / 250
> z_price = strength_diff / 2

The selected `z_strength` is the highest-priority available signal:
prefer `z_elo` when present (live, derived from realised match history),
fall back to `z_rank` otherwise (static FIFA snapshot), fall back to
price-only when neither is present (EPL training).

Strengths: every number is explainable, no training data required, no
distribution-shift risk between EPL and WC.

Weaknesses: cannot learn from data. If a real pattern exists that the
formula does not capture, the heuristic will never find it. The slope
in price is linear within each position; in reality marginal return
on the last million is not linear.

## 5.3 Backend 2: structural Poisson goals model

Three steps:

1. Estimate per-team expected goals (xG_for, xG_against) for the
   fixture. Base rate 1.30 goals per team; multiplied by a matchup
   factor `exp(elo_diff/400 + price_diff/6)` with a small home boost.
2. Distribute team xG across player positions by share. Forwards take
   ~0.50 of team goals; midfielders 0.30; defenders 0.10; goalkeepers 0.
3. Compute clean-sheet probability as `exp(-opp_xg)` under Poisson. Sum
   the contributions (goals at position-specific rates, assists,
   clean-sheet bonus, appearance, GK save bonus, goals-conceded
   penalty) into a fantasy point total.

Strengths: every number traces to a single football statement. Clean
sheet term is exact under the Poisson assumption. No training data
needed.

Weaknesses: per-position share is a population average. The model
cannot tell a star striker from his backup if they have the same price
and matchup.

## 5.4 Backend 3: LightGBM v2

Per-position regressors. Each position trains four heads:

- `mean`: regression on `total_points` as the target
- `q10`, `q50`, `q90`: quantile regression at the 10th, 50th, 90th
  percentiles

Features (five):

> price_millions, is_home (0/1), strength_diff,
> squad_top_n_avg_price, opp_squad_top_n_avg_price

Hyperparameters (picked by `tune.py`'s sweep):

> num_leaves=15, learning_rate=0.05, n_estimators=200,
> min_child_samples=30, feature_fraction=0.9, bagging_fraction=0.9,
> bagging_freq=5

A deterministic seed (`seed=42` plus `bagging_seed=42` and
`feature_fraction_seed=42`) plus `deterministic=True` makes training
reproducible; without it, run-to-run RMSE wobbles around ±0.05 from
LightGBM's bagging RNG, which masks small feature-engineering A/Bs.

Strengths: captures non-linear interactions a hand-written formula
cannot express compactly. Best in class for tabular data at this scale.

Weaknesses: extrapolation. If the inference data has nothing like
itself in training, the GBM has nothing to draw on. Most consequentially
for our use case, the GBM was trained on EPL FPL data; non-EPL stars
(Messi, Mbappé, Neymar, Modrić, etc.) are out of distribution and the
GBM under-predicts them at premium price tiers.

## 5.5 Why three backends instead of one?

The held-out validation (Section 7) shows per-position differences:
Poisson dominates GK, heuristic dominates DEF, GBM dominates MID and
FWD. The differences are small (RMSE within 0.3 across the three for
DEF, MID, FWD) but they are stable across runs.

In production we score a single backend per run and pass it to the
optimiser, rather than averaging across backends. Two reasons:

- The MILP objective consumes one prediction column; ensembling shifts
  the optimum in non-obvious ways. The per-position best-backend choice
  is honest about which model we trust where.
- The three backends are useful for cross-checking. When they agree, we
  are confident; when they diverge sharply (Messi: heuristic 8.5, GBM
  2.2), the disagreement itself is data.

## 5.6 Scouting bonus encoding

The FIFA Fantasy rules add a +2 scouting bonus when a player scores
more than 4 points AND is owned by less than 5% of teams. We encode
this as a post-prediction step (`apply_scouting_bonus` in
`optimizer/pipeline.py`) using the model's deterministic prediction as a
proxy for the realised outcome. This over-credits players who the model
predicts at 4.5 (will likely score under 5 and not trigger) and
under-credits players who the model predicts at 3.5 (might still trigger
with realised variance), but the deterministic shape is good enough for
ordinal squad selection.

## 5.7 MILP optimiser

PuLP with the bundled CBC backend. Three solver entry points:

- `solve_squad(players, stage_config)`: pick the 15-player squad
  maximising the sum of `total_effective_points` over the stage horizon
  under position counts, country cap, and budget constraints.
- `solve_transfer(players, current_squad_ids, stage_config,
  rolled_over_transfers)`: same as squad selection plus a slack variable
  for transfers above the free quota, penalised at 3 points per extra
  per the official rule.
- `solve_lineup(squad_round)`: pick the starting XI from the 15-player
  squad for a single round, including formation choice (one of seven
  valid formations) and captain/vice-captain. Bench is auto-ordered by
  predicted points within position.

Stage configurations encode budget ($100M group stage; $105M R32+),
country cap (3 group; 4 R16; 5 QF; 6 SF; 8 Final), and free transfers
per stage. Pre-tournament and pre-R32 are "unlimited" transfers, which
we model by short-circuiting to `solve_squad`.

## 5.8 Live decision tools

After matches kick off, a `live` module provides:

- A captain switch tool: given the running scores of all 16 fixtures in
  a matchday and a deadline for the manual sub option, compute the
  expected-value gain of switching captain from the current pick to any
  other XI member who has not played yet.
- A sub advisor: given a finished match in which a starter played 0
  minutes (so the FIFA auto-sub already ran), advise whether to
  manually swap an unplayed bench player into a starter slot. The
  decision balances the expected gain from the manual swap against the
  expected loss from forfeiting auto-subs for any remaining unplayed
  starters in the round.
