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
