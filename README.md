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
- Phase 1 — Data collection. Not started.
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
