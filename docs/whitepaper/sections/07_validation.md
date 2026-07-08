# 07 - Held-out validation on EPL 2024-25 GW 30-38

Status: **DRAFT**

## 7.1 Strategy

We use the EPL 2024-25 season as a proxy for the WC. The scoring rules
are nearly identical (the FPL game and the FIFA WC Fantasy game share a
heritage), so a model that ranks players well on FPL data should rank
players well on WC data, subject to the distribution-shift caveats
discussed in Section 9.

The hold-out split is GW 30-38 of the 2024-25 season (the last nine
gameweeks). All earlier rows from all three seasons (2022-23, 2023-24,
and 2024-25 GW 1-29) form the training set:

- Raw: ~78,500 player-gameweek rows across the three seasons
- After dropping did-not-play rows: 34,221 rows (the modelling set)
- Holdout: the n values in the RMSE table below are the after-filter
  per-position counts for GW 30-38

The held-out RMSE is computed per position against the realised
`total_points`. The same rows are scored by all three backends, so the
comparison is fair.

## 7.2 The canonical RMSE table

Lower is better. Numbers are produced by
`src/fifa_fantasy/training/validate_main.py` with the deterministic seed.

| Position | n | Heuristic | Poisson | GBM v2 |
|---|---:|---:|---:|---:|
| GK | 180 | 2.698 | **2.503** | 2.650 |
| DEF | 910 | **3.141** | 3.400 | 3.187 |
| MID | 1228 | 2.874 | 4.371 | **2.767** |
| FWD | 368 | 3.515 | 4.560 | **3.153** |

Per-position winners:

- **GK: Poisson.** The clean-sheet term is exact under the Poisson
  assumption; goalkeeper scoring is dominated by appearance and clean
  sheets, both of which the structural model gets right.
- **DEF: heuristic.** The price-coefficient times matchup factor tracks
  defender variance better than the GBM at this data scale.
- **MID and FWD: GBM v2.** Individual variance is largest at attacking
  positions; the GBM has room to learn non-linear interactions a
  hand-written formula cannot capture compactly.

## 7.3 GBM v1 vs v2

Two iterations are documented in this repo:

| Aspect | v1 (earlier) | v2 (current) |
|---|---|---|
| Training data | one EPL season (2024-25), ~11,500 rows | three seasons (2022-23, 2023-24, 2024-25), ~34,200 rows |
| Hyperparameters | LightGBM defaults: 31 leaves, 400 trees | sweep-tuned: 15 leaves, 200 trees |
| GK RMSE | 2.710 | 2.650 |
| DEF RMSE | 3.297 | 3.187 |
| MID RMSE | 2.898 | 2.767 |
| FWD RMSE | 3.281 | 3.153 |

The v1 squad picks looked aggressive (Norwegian and Swedish DEF
clusters, Chris Wood as vice captain) because the model was overfitting:
too many leaves on too little data. The lighter v2 config produced more
conventional squads (Dembélé as captain, Olise as vice on the
pre-tournament pool).

## 7.4 GBM v3 candidate (rejected)

A v3 candidate was tested with a sixth feature, `team_elo_diff`. The
training-side value comes from `compute_club_elo_history` on the
football-data.co.uk match table; the inference-side value comes from
`country_elo_diff` aliased to the same column name. Same hyperparameters
as v2, deterministic seed:

| Position | v2 | v3 | Δ |
|---|---:|---:|---:|
| GK | 2.650 | 2.632 | -0.018 (better) |
| DEF | 3.187 | 3.221 | +0.034 (worse) |
| MID | 2.767 | 2.748 | -0.019 (better) |
| FWD | 3.153 | 3.168 | +0.015 (worse) |

Net: a ±1% wash on EPL. Worse, the new feature introduces
distribution-shift risk at WC inference. Country-Elo gaps span ~480
points at the WC extreme (Argentina 2072 vs Curaçao 1592); EPL
club-Elo gaps span ~80 points. The GBM has no training rows in the
larger range and would extrapolate poorly.

We did not ship v3. The Elo column lives on for the heuristic and
Poisson backends, where the signal is consumed directly without
distribution-shift risk.

## 7.5 Determinism and noise floor

Pre-determinism, the same code over the same data wobbled by ±0.05 per
position from LightGBM's bagging RNG. That noise floor masked the v3
feature A/B above: the first run showed mixed results that were
indistinguishable from noise. Setting `seed`, `bagging_seed`,
`feature_fraction_seed`, and `deterministic=True` collapsed the variance
to zero and made the v3 decision honest.

The lesson generalises: any tabular-ML feature engineering A/B needs
reproducible training. The wobble cost us an unnecessary half-day of
debugging in the v3 attempt; we now treat the deterministic seed as part
of the GBM contract.

## 7.6 What the validation does NOT measure

EPL RMSE is a useful proxy for fantasy-points prediction quality, but
several phenomena specific to international tournament play are
invisible to it:

- **Star international players are absent from training.** Messi,
  Mbappé, Modrić, Neymar, and others never appear in the EPL feature
  set. The GBM scores them on price alone, undervaluing them
  systematically.
- **Manager rotation behaviour differs.** EPL coaches rotate based on
  fixture congestion; international coaches rotate based on
  group-stage clinching. The GBM has no signal for either.
- **Knockout-stage variance is higher.** Single-leg elimination games
  have different team incentives than round-robin group games. The
  training rows are all from a 38-game round-robin schedule.

These show up in the live-tournament results (Section 8) and the
analysis (Section 9), not in the held-out RMSE.
