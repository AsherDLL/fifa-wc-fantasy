# Phase 3a — Heuristic Baseline Predictor

A non-trained, deterministic predictor of `predicted_points` per (player,
round) row. Exists so Phase 4 (the optimizer) has something concrete to
consume before the LightGBM models in Phase 3b are trained on Euro 2024
data.

## Formula

```
predicted_points = points_per_price_unit[position] * price_millions
                 * (1 + alpha * tanh(strength_diff / scale))
                 * (1 + beta * is_home)
```

with the result **zeroed** for any player whose `status` is not `"playing"`
or whose squad is eliminated.

### Constants (in `src/fifa_fantasy/model/baseline.py`)

| Constant | Value | Reasoning |
|---|---|---|
| `points_per_price_unit[GK]` | 0.50 | Lower ceiling — clean sheet is binary, save bonuses bounded. |
| `points_per_price_unit[DEF]` | 0.55 | Clean sheet + occasional goal, capped upside. |
| `points_per_price_unit[MID]` | 0.60 | Balanced — goals, assists, chances, tackles. |
| `points_per_price_unit[FWD]` | 0.65 | Highest goal frequency, highest ceiling. |
| `STRENGTH_DIFF_SCALE` | 2.0 | `strength_diff` (top-11 avg price difference) is roughly ±3 in practice; `tanh(x/2)` saturates by ±5. |
| `STRENGTH_DIFF_ALPHA` | 0.25 | At saturation, fixture is worth ±25% of base prediction. |
| `HOME_ADVANTAGE_BETA` | 0.05 | Home side gets a +5% nudge. WC neutral venues blunt the usual home boost. |

All constants are **calibrated by intuition, not by data.** The Phase 3b
LightGBM models will replace this with quantile-regressed predictions
trained on Euro 2024 (same scoring rubric).

## What this predictor captures

- The market's pricing signal — strongest single feature available
  pre-tournament.
- Fixture difficulty via opponent strength.
- A small home-advantage bias.
- Availability gating (transferred / eliminated → 0).

## What it does NOT capture

- Per-player form, expected goals (xG), defensive contributions, set-piece
  duties — none of these are knowable from price alone.
- Rotation risk — a star priced for 90 min might play 30.
- Booster effects (Phase 4 problem).
- Captaincy doubling (Phase 4 problem).
- The scouting bonus from `scoring.py` (Phase 4 will inject this at the
  optimizer level using `ownership_fraction`).

## Live snapshot (day-1 pool)

```
predicted_points: min=0.00 mean=2.52 max=8.74
```

Top 5 by predicted_points: Mbappé (8.74), Kane (8.64), Ronaldo (8.01),
Messi (7.92), Haaland (7.89). These rankings agree with general
expert consensus, which is the most we can ask of a price-driven
heuristic.

## Tests

`tests/test_baseline.py` pins:

- The per-position coefficient.
- Strength-diff direction (positive boosts, negative shrinks).
- Strength-diff saturation at ±alpha.
- Multiplicative home advantage.
- Zeroing for non-`playing` status and eliminated squads.
- Input DataFrame is not mutated.
- All input columns survive the call.
