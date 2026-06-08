# Phase 3b: LightGBM predictor

A second predictor backend that wraps four position-specific LightGBM
models trained on Premier League fantasy data. The default backend
remains the heuristic; the GBM is opt-in via `--backend gbm` on the
model CLI.

## Training data

Source: Premier League Fantasy (`fantasy.premierleague.com/api`) over
one completed season. Same scoring components as the WC fantasy game
(appearance, goals, assists, clean sheets, saves, cards, bonus), same
auction-priced market, similar fixture-difficulty mechanic.

One season is around 29,000 player-gameweek rows. After dropping
non-appearances, around 11,500 rows remain. Per-position counts vary
(GK 770, DEF 3,900, MID 5,300, FWD 1,400).

Rescraping:

```bash
python -m fifa_fantasy.training --season 2024-25
```

Writes `data/training/fpl_player_gameweek_<season>.parquet`.

## Model

Each position has four heads:

- mean: standard regression for `E[total_points]`.
- q10, q50, q90: quantile regression at the 10th, 50th, 90th percentiles.

LightGBM, 400 boosted rounds, default learning rate 0.05. Model
artefacts persist to `data/models/gbm_<position>_<head>.txt`.

Training:

```bash
python -m fifa_fantasy.model.train
```

## Inference

```bash
python -m fifa_fantasy.model --backend gbm
```

Reads the latest features Parquet under `data/processed/`, loads the
four position models, writes `data/processed/predictions_<date>.parquet`
with `predicted_points` plus `predicted_q10`, `predicted_q50`,
`predicted_q90`. The optimizer consumes `predicted_points` exactly as
it did with the heuristic.

Features fed to the model are a subset of the WC feature table:

```
price_millions, is_home (0/1), strength_diff,
squad_top_n_avg_price, opp_squad_top_n_avg_price
```

`rank_diff` (FIFA World Ranking gap) is computed at inference but not
present in EPL training, so the column is dropped from the model input.

## Why the GBM picks look aggressive (read first)

The GBM has learned from EPL that mid-priced players for budget teams
in a winning matchup score well per dollar. The canonical EPL pattern
is a 6.5M Brighton forward facing a relegation side: the player is
cheap by FPL standards, the team is mid-table, the opponent is weak,
and the resulting points-per-pound is excellent. Transferring that
pattern to the WC pool produces cheap-defender-heavy picks (Norway and
Sweden back lines, Chris Wood at vice) that read aggressive next to
the heuristic's consensus picks.

Three concrete reasons the EPL-to-WC transfer is fragile:

1. **Volume bias toward cheap players in EPL training data.** The EPL
   pool has many more cheap players (4.0M-5.5M) than expensive ones,
   and those cheap players are mostly defenders for mid-table teams.
   The model sees a lot of rows where "cheap defender + favourable
   strength_diff" actually scored well, and learns to lean on that.
2. **Different fixture-difficulty distribution.** Top-versus-bottom
   matchups in EPL (Man City vs Sheffield United, etc.) are far more
   lopsided than the worst WC group-stage matchups, where every team
   has at least qualified. So the WC strength_diff range is narrower
   than the EPL range the model trained on, and the model extrapolates
   modestly outside its training distribution.
3. **No rotation signal.** EPL fantasy has heavy rotation (cup
   competitions, midweek games), and the model implicitly uses price
   as a proxy for "definitely starts". At the WC every fit starter
   starts; that proxy is much weaker, so cheap-defender-with-good-
   matchup picks that rely on the rotation signal lose grounding.

Whether the transfer is right or wrong depends on whether WC group-
stage scoring patterns look like EPL. We will know within a week.

## If it stays bad, possible pivots

These are options I would consider if MD1 results show the GBM
underperforming the heuristic by a meaningful margin (say >5 pts):

- **Append WC labels and refit.** First move once MD1 finishes.
  `python -m fifa_fantasy.model.train --include-wc` will pull realised
  per-(player, round) points and concatenate them with the EPL rows.
  Two MDs of WC data is small but every WC sample is right-domain.
- **Structural Poisson goals model.** No training data needed. Use
  rankings + price to estimate team xG; split by player role (penalty
  taker, set piece taker, etc.). Beats LightGBM when training data is
  distributionally far from inference. Has no opt-in CLI yet; would
  ship as `--backend poisson`.
- **Tighter EPL-to-WC feature alignment.** Currently the only signal
  difference is `rank_diff` (present at inference, absent in training).
  We could compute a synthetic "club rank" feature in EPL (using FPL
  team strength) and a "team rank" feature in WC (using FIFA ranking)
  and route them through the same column. Modest expected lift.
- **Restrict the GBM to a narrower decision.** Instead of replacing
  the heuristic, use the GBM only for the captain pick (where the
  quantile q90 head matters most) and keep heuristic for the
  15-player selection. Lowest risk if the GBM ranks players sensibly
  even when absolute values are off.

## Caveats

The EPL-to-WC transfer is imperfect on at least two axes:

1. EPL fixture variance is higher than WC. The top 6 versus bottom 6
   gap in EPL is wide enough that "cheap defender for a top side versus
   a relegation candidate" is a recurring scoring pattern the model
   leans on. At the WC, all 48 teams have qualified; the equivalent
   pattern (cheap defender for a top-ranked nation versus a weak one)
   is rarer.
2. EPL FPL prices live on a slightly different scale and use different
   pricing rules from FIFA Fantasy. The model still generalises
   correctly in shape, but absolute predicted values are not directly
   comparable to the heuristic's.

The practical consequence on the day-1 WC pool: the GBM picks a
defender-heavy squad with cheap selections from Norway and Sweden plus
a Chris Wood (NZL) vice-captain. The heuristic picks the more obvious
Germans, Spaniards, French and English. The honest expectation is that
the GBM looks aggressive pre-tournament; whether it pays off depends on
whether MD1 produces EPL-shaped or different-shaped scoring.

The right next step (once MD1 and MD2 are played):

- Append the realised WC results to the training set.
- Retrain on `train = EPL + WC_so_far`, keeping the four-position split.
- Compare against the heuristic each round on real points-per-pound.

For now, both backends ship side by side and the user picks.
