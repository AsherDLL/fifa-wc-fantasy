# Predictor approaches

This project ships three different predictors. Each maps the per-(player,
round) feature table into a `predicted_points` column. The optimizer
then ranks and picks under Fantasy rules.

Pick a predictor with `--backend`:

```bash
python -m fifa_fantasy.model                    # default: heuristic
python -m fifa_fantasy.model --backend gbm      # LightGBM trained on EPL
python -m fifa_fantasy.model --backend poisson  # structural Poisson goals
```

The chosen backend is stamped into the predictions Parquet (column
`model_backend`) and propagated into the optimizer's output filename
and JSON metadata, so a recommendation always declares which approach
generated it.

## 1. heuristic

Code: `src/fifa_fantasy/model/baseline.py`. Default.

```
base    = points_per_price_unit[position] * price_millions
matchup = 1 + alpha * tanh(combined_strength_z)
home    = 1 + beta * is_home
premium = premium_boost * max(0, price_millions - 9.0)

predicted_points = base * matchup * home + premium
```

The combined strength signal blends a price-based proxy
(`strength_diff = own_top_11_avg - opp_top_11_avg`) with the FIFA
World Ranking gap (`rank_diff`). The squad-price proxy dominates if
the ranking is missing.

Position-specific coefficients (GK 0.50, DEF 0.55, MID 0.60, FWD 0.65)
reflect rough scoring ceilings; alpha 0.40 caps the matchup adjustment
at 40 percent at saturation; beta 0.05 is a small home boost. The
optional `--premium-boost` knob adds a linear-above-threshold term
that tilts the optimizer toward $9M+ players.

Strengths: interpretable, no training data needed, fast.

Weaknesses: linear in price within position; under-weights the
non-linear ceiling of premium attackers; ignores per-match variance.

## 2. gbm (LightGBM)

Code: `src/fifa_fantasy/model/gbm.py`, `src/fifa_fantasy/model/train.py`,
`src/fifa_fantasy/training/`.

Four LightGBM ensembles, one per position. Each has four heads (mean,
q10, q50, q90). Trained on one full Premier League FPL season
(`fantasy.premierleague.com/api`). Same scoring components, same
auction-priced market.

Strengths: data-driven; learns patterns the heuristic cannot encode by
hand; provides quantile bands (P10/P50/P90) that a future captain
heuristic can use to favour high-ceiling picks.

Weaknesses: the EPL-to-WC transfer is fragile (see `docs/gbm.md`). On
the day-1 WC pool it picks cheap defenders for budget national sides
in winning matchups, which reads aggressive next to the heuristic.
The `--include-wc` flag on `model.train` appends realised WC labels
once games are played; one or two refit cycles into the group stage
should improve fit substantially.

## 3. poisson (structural goals model)

Code: `src/fifa_fantasy/model/poisson.py`.

A third backend, neither price-anchored nor data-driven. It encodes
the football itself:

1. **Team xG**. Each team's expected goals for a fixture is a base rate
   (1.30) scaled by a matchup factor `exp(rank_diff/800 +
   strength_diff/6)` and a small home boost. The opponent's xG flips
   the sign and gets the away penalty.
2. **Player goal/assist share**. Per position, take a fixed share of
   the team's xG as the player's expected goals (DEF 0.10, MID 0.30,
   FWD 0.50), and a similar share for expected assists. Goalkeepers
   get essentially zero.
3. **Clean sheet probability**. Under Poisson, `P(opp scores zero) =
   exp(-opp_xg)`. GK/DEF get 5 points times this probability, MID 1
   point.
4. **Goals conceded penalty**. Under Poisson, `E[max(0, k-1)] =
   opp_xg - 1 + exp(-opp_xg)`. GK takes a full -1 per goal beyond the
   first; DEF takes a discounted version of that hit.
5. **Scouting bonus and appearance**. Same constants as the canonical
   `scoring.py`: appearance 2, scouting bonus 2 (when predicted > 4
   and ownership < 5 percent).

Strengths: every number is traceable to a single line of football
intuition. No training data; no distributional risk. Picks scale with
team-level matchups (Spain v Cabo Verde, Germany v Curacao), which is
exactly the signal the user flagged as central to fantasy point
generation.

Weaknesses: the goal/assist shares are population averages, not
player-specific. It will not distinguish a star striker from a backup
on the same team unless they have a different price (which the model
ignores). The matchup multiplier is bounded by an `exp` clip; the
extreme high end may still over-state premium-vs-weak matchups.

## Picking a backend

For pre-tournament submission and as a sanity check: **use the
heuristic**. It is the most conservative and produces a squad that
matches widely held expectations.

For experimentation and analysis: run all three and compare. The
static HTML report (`python -m fifa_fantasy.web`) shows all
recommendations side by side under `results/`.

After MD1 finishes: refit the GBM with WC labels
(`python -m fifa_fantasy.model.train --include-wc`) and re-evaluate.
The Poisson model does not need refitting; the heuristic constants
could be re-tuned by hand against early WC results.

## Optimizer is independent of backend

The squad-selection MILP, the lineup MILP and the transfer MILP all
consume a single `predicted_points` column. Swapping backends does not
require any optimizer change. Quantile columns from the GBM
(`predicted_q10`, `predicted_q50`, `predicted_q90`) are written
through to the predictions Parquet but not yet used; a future captain
heuristic can read them.
