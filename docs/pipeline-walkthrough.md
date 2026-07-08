# Pipeline Walkthrough - Lautaro Martínez through every stage

A worked example. Traces one player from the raw FIFA Fantasy API record
through to his selection as captain in the MD1 recommendation. Every
number below was computed by the actual code in this repo - same one you
get by running:

```bash
python -m fifa_fantasy.collector
python -m fifa_fantasy.features
python -m fifa_fantasy.model
python -m fifa_fantasy.optimizer
```

Reference run: 2026-06-07 group-stage snapshot. Player: Lautaro Martínez.

---

## Step 1 - Collector (from FIFA API)

`src/fifa_fantasy/collector/` hits
`https://play.fifa.com/json/fantasy/players.json`, validates with
Pydantic, and writes `data/raw/players_<UTC-date>.parquet`. Lautaro's
raw record:

```
player_id   = 1338
position    = FWD
country     = Argentina
price       = $8.8M
ownership   = 4.0%
status      = playing
eliminated  = False
```

---

## Step 2 - Features (one row per round)

`src/fifa_fantasy/features/` joins his player record with Argentina's
three group-stage fixtures and the per-squad strength proxy
(`squad_top_n_avg_price`, mean of the top-11 most-expensive players in
each squad's pool). Argentina's top-11 average price = **$7.19M**.

| Round | Opponent | Home | Opp top-11 | strength_diff |
|---|---|---|---|---|
| MD1 | ALG | yes | 5.79 | **+1.40** |
| MD2 | AUT | yes | 6.00 | **+1.19** |
| MD3 | JOR | × | 4.51 | **+2.68** |

Stored in `data/processed/features_<UTC-date>.parquet`.

---

## Step 3 - Heuristic predictor

`src/fifa_fantasy/model/baseline.py` applies the Phase 3a formula:

```
predicted_points
    = points_per_price_unit[position] * price_millions          # base
    * (1 + alpha * tanh(strength_diff / scale))                 # matchup
    * (1 + beta * is_home)                                      # home
```

Constants for Lautaro: FWD coefficient `0.65`, scale `2.0`,
α=`0.25`, β=`0.05`. Base = `0.65 × 8.8 = 5.72`.

| Round | Calculation | predicted_points |
|---|---|---|
| MD1 | 5.72 × 1.1511 × 1.05 | **6.913** |
| MD2 | 5.72 × 1.1335 × 1.05 | **6.808** |
| MD3 | 5.72 × 1.2180 × 1.00 | **6.967** |

Stored in `data/processed/predictions_<UTC-date>.parquet`.

---

## Step 4 - Scouting bonus + horizon aggregation

`src/fifa_fantasy/optimizer/pipeline.py` injects the official scouting
bonus (rule reused from `fifa_fantasy.scoring`): **+2 if
predicted_points > 4 AND ownership < 5%**. Lautaro at 4.0% owned with
all-three-rounds predictions above 4 → he triggers in all three rounds.

```
effective_points[MD1] = 6.913 + 2 = 8.913
effective_points[MD2] = 6.808 + 2 = 8.808
effective_points[MD3] = 6.967 + 2 = 8.967

total_effective_points = 26.69
```

The aggregator sums this per player across the horizon
(`(1, 2, 3)` for the pre-tournament selection).

---

## Step 5 - Squad MILP (15-player optimization)

`src/fifa_fantasy/optimizer/solvers.py::solve_squad` builds a PuLP/CBC
MILP that **maximizes total horizon effective_points** subject to:

- exactly 15 players, with 2 GK / 5 DEF / 5 MID / 3 FWD,
- total price ≤ $100M,
- ≤ 3 players from any one country (group-stage cap),
- exclude eliminated squads.

The solver picks Lautaro because of points-per-dollar:

| Player | Price | Horizon eff. pts | per $M |
|---|---|---|---|
| **Lautaro Martínez** | $8.8M | 26.69 | **3.03** |
| Harry Kane | $10.5M | 25.43 | 2.42 |
| Ollie Watkins | $7.9M | 25.13 | 3.18 |
| Kylian Mbappé | $10.5M | 24.96 | 2.38 |
| Ferran Torres | $7.8M | 24.65 | 3.16 |

Lautaro returns the highest projected points-per-dollar of any forward
in the pool. The £10.5M premiums (Kane, Mbappé, Haaland) don't return
their price advantage under the heuristic - the MILP correctly
reallocates that budget to a second / third premium forward (Watkins,
Torres) instead. **Total horizon = 263.87 pts** beats any squad that
includes Mbappé.

This is the heuristic's known blind spot: it can't see that premium
forwards have a non-linear ceiling (fatter right tail in their
distribution of match outcomes). Phase 3b's quantile regression on Euro
2024 data should fix this - see `docs/baseline.md`.

---

## Step 6 - Captain pick

`src/fifa_fantasy/optimizer/solvers.py::solve_lineup` runs a second MILP
to pick the starting XI under one of the seven valid formations,
maximizing **target-round** predicted_points. Captain = highest-predicted
starter for that round; vice = second. For MD1:

```
1. Lautaro Martínez (FWD, ARG)  E=6.91   ← CAPTAIN
2. Ferran Torres   (FWD, ESP)   E=6.49   ← VICE
3. Ollie Watkins   (FWD, ENG)   E=6.39
4. Nico Williams   (MID, ESP)   E=5.99
… (formation 3-4-3)
```

So the captain's points double:

```
MD1 expected total
    = 58.20 (sum of 11 starters' predicted_points)
    + 6.91 (captain bonus, doubling Lautaro)
    = 65.11 pts
```

The recommendation lands in `results/<host>_recommendation_GROUP_MD1_<UTC-date>.{json,md}`.

---

## Where each file lives in the repo

| Step | Code | Output |
|---|---|---|
| 1 | `src/fifa_fantasy/collector/` | `data/raw/players_<date>.parquet`, etc. |
| 2 | `src/fifa_fantasy/features/` | `data/processed/features_<date>.parquet` |
| 3 | `src/fifa_fantasy/model/baseline.py` | `data/processed/predictions_<date>.parquet` |
| 4 | `src/fifa_fantasy/optimizer/pipeline.py` | (in-memory; combined with Step 5) |
| 5 | `src/fifa_fantasy/optimizer/solvers.py::solve_squad` | (in-memory) |
| 6 | `src/fifa_fantasy/optimizer/solvers.py::solve_lineup` + `report.py` | `results/<host>_recommendation_<STAGE>_<date>.{json,md}` |

Every step is a pure function on pandas DataFrames; the CLI just chains
them together.
