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
- Phase 2 — Feature engineering. Not started.
- Phase 3 — Prediction models. Not started.
- Phase 4 — Optimizer. Not started.
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
