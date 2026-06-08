# FIFA Fantasy World Cup 2026

A prediction, optimization, and live decision-support system for the official
FIFA World Cup 2026 Fantasy game. Predicts per-player fantasy points,
selects the optimal 15-player squad under stage-aware constraints, and advises
on in-round captain switches and substitutions.

See [`fifa-fantasy-project-sketch.md`](./fifa-fantasy-project-sketch.md) for the
full design and [`docs/`](./docs) for operational details, scoring reference,
and design decisions.

## Status

- **Phase 0 — Scoring rules.** Complete. Canonical scoring functions plus tests.
- **Phase 1 — Data collection.** Complete. Fetches players, squads, and
  fixtures from the FIFA Fantasy public JSON endpoints into Parquet.
- **Phase 2 — Feature engineering.** Complete. Builds a per-(player, round)
  feature table joining players × fixtures × squad-strength proxies.
- **Phase 3a — Baseline predictor.** Complete. Heuristic
  position-coef × price × matchup × home, gives the optimizer something
  concrete to consume before training data exists.
- Phase 3b — LightGBM models. Not started (needs Euro 2024 training data).
- **Phase 4 - Optimizer.** Complete. PuLP/CBC MILPs for the 15-player
  squad, transfer planning (-3 per extra), and starting XI + formation,
  all stage-aware. CLI emits a JSON + markdown recommendation.
- **Phase 4.7 - Strength signals.** Complete. Blends FIFA World Ranking
  with the price-based squad proxy so opponent strength is first-class
  in the matchup multiplier. Static snapshot under `data/static/`.
- Phase 5 — Live decision support. Not started.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the tests

```bash
pytest
```

## Collect data

Fetches the player pool, national squads, and fixtures from
`play.fifa.com` and writes Parquet files under `data/raw/`:

```bash
python -m fifa_fantasy.collector              # all three
python -m fifa_fantasy.collector --only players
python -m fifa_fantasy.collector --data-dir /tmp/out
```

See [`docs/api-endpoints.md`](./docs/api-endpoints.md) for the endpoint specs.

## Build features

Reads the latest `data/raw/{players,squads,fixtures}_*.parquet` snapshots
and writes a single per-(player, round) feature table to
`data/processed/features_<UTC-date>.parquet`:

```bash
python -m fifa_fantasy.features
python -m fifa_fantasy.features --raw-dir data/raw --out-dir data/processed
```

See [`docs/features.md`](./docs/features.md) for the column dictionary and
the squad-strength rationale.

## Predict points

Runs the Phase 3a baseline predictor over the latest feature table and
writes `data/processed/predictions_<UTC-date>.parquet`:

```bash
python -m fifa_fantasy.model
```

See [`docs/baseline.md`](./docs/baseline.md) for the formula and tuning notes.

## Recommend a squad

Runs the Phase 4 squad + lineup MILPs and writes
`data/processed/recommendation_<STAGE>_<UTC-date>.json`:

```bash
python -m fifa_fantasy.optimizer                   # default GROUP_MD1 (fresh)
python -m fifa_fantasy.optimizer --stage R16

# Plan transfers from a previous-round recommendation
python -m fifa_fantasy.optimizer --stage GROUP_MD2 \
    --from results/<host>_recommendation_GROUP_MD1_<date>.json

# Daily pre-lockout refresh: alternatives section + diff vs yesterday
python -m fifa_fantasy.optimizer --report-alternatives \
    --compare-to results/<host>_recommendation_GROUP_MD1_<yesterday>.json
```

Premium-tier knob on the predictor (default 0.0):

```bash
python -m fifa_fantasy.model --premium-boost 0.4   # tilt toward £9M+ players
```

One-shot daily wrapper:

```bash
./scripts/daily-snapshot.sh                        # collector → features → model → optimizer
PREMIUM_BOOST=0.4 ./scripts/daily-snapshot.sh
```

See [`docs/optimizer.md`](./docs/optimizer.md) for the MILP formulation
and stage-config table.
