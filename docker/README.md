# Docker setup for FIFA Fantasy WC 2026

A reproducible container that bundles the data pipeline, models,
optimizer, and the daily snapshot loop. Lets anyone clone the repo,
build once, and run the system end-to-end without touching their
host Python environment.

## Quick start

```bash
# Build the image (one-time, ~3 minutes the first run)
cd docker/
docker compose build

# Interactive: run any pipeline stage on demand
docker compose run --rm fantasy python -m fifa_fantasy.collector
docker compose run --rm fantasy python -m fifa_fantasy.features
docker compose run --rm fantasy python -m fifa_fantasy.optimizer --stage R32

# Background: run the snapshot loop until the WC ends (2026-07-18)
docker compose --profile snapshot up -d

# Check the snapshot loop logs
docker compose --profile snapshot logs -f snapshot

# Stop the snapshot loop (e.g. after the final)
docker compose --profile snapshot down
```

## What the snapshot loop does

The loop runs four independent schedules in one process:

- **FIFA tick** (every `FIFA_INTERVAL_HOURS`, default 12): pulls the
  FIFA Fantasy API (players, squads, fixtures) into `data/raw/`,
  rebuilds the features Parquet under `data/processed/`, retrains the
  GBM with `--include-wc` (EPL + completed WC rounds), runs four
  backends (heuristic, poisson, gbm, ensemble), then runs the optimizer
  for the current stage (env var `STAGE`) on the ensemble predictions
  (override with `OPTIMIZER_BACKEND`) and writes the recommendation
  JSON+MD under `results/`, plus a static `results/index.html`.
- **Markets tick** (every `MARKETS_INTERVAL_HOURS`, default 3):
  Polymarket / Kalshi prediction-market snapshots under `data/external/`.
- **News tick** (every `NEWS_INTERVAL_HOURS`, default 6): RSS article
  collection under a disk budget, then team-news lineup extraction.
- **Elo tick** (every `ELO_INTERVAL_HOURS`, default 24): refreshes the
  martj42 international Elo CSV.

A dashboard portal is served on port `WEB_PORT` (default 8770). It
renders the results page fresh on each request; there is no
auto-refresh, a manual browser refresh shows the latest.

Errors in any single stage are logged and do not block the rest of the
tick. The loop exits cleanly after `WC_END_DATE` (default 2026-07-18).

## Environment variables

| Variable | Default | What |
|---|---|---|
| `FIFA_INTERVAL_HOURS` | 12 | Hours between FIFA data + model + optimizer ticks |
| `MARKETS_INTERVAL_HOURS` | 3 | Hours between prediction-market snapshots |
| `NEWS_INTERVAL_HOURS` | 6 | Hours between RSS news + team-news ticks |
| `ELO_INTERVAL_HOURS` | 24 | Hours between Elo refreshes |
| `NEWS_BUDGET_MB` | 2048 | Disk cap for the news article cache |
| `NEWS_MAX_PER_FEED` | 20 | Max articles fetched per feed per tick |
| `NEWS_FIXTURES_AHEAD` | 3 | Fixtures ahead for lineup extraction |
| `GBM_INCLUDE_WC` | 1 | Retrain the GBM with `--include-wc` each FIFA tick |
| `OPTIMIZER_BACKEND` | ensemble | Predictions the optimizer consumes |
| `WEB_PORT` | 8770 | Dashboard HTTP port |
| `WC_END_DATE` | 2026-07-18 | Loop exits after this UTC date |
| `STAGE` | QF (compose; code default R32) | Which stage config the optimizer runs for |

Override at compose run time:

```bash
FIFA_INTERVAL_HOURS=6 STAGE=SF docker compose --profile snapshot up -d
```

## Data persistence

All written outputs go under host-mounted `../data` and `../results`
directories. The container itself is stateless; you can rebuild and
restart without losing snapshots.

To completely wipe everything after the WC:

```bash
docker compose --profile snapshot down
docker rmi fifa-fantasy:latest
# Optional: rm -rf ../data/raw/* ../data/processed/* ../data/external/prediction_markets/
```

## CBC solver note

PuLP bundles the CBC MILP solver as a Linux x86_64 wheel; the image
inherits that and the optimizer works out of the box. On ARM hosts
(Apple Silicon, ARM servers), uncomment the `coinor-cbc` apt install
line in `Dockerfile` to use the system solver.

## Disk usage budget

A single FIFA tick writes about 1.5 MB of Parquet (raw + features +
predictions) plus a few KB of JSON results; at the default 12h cadence
that is a few MB per day. The prediction-market JSONL snapshots add
~200 KB each, and the news article cache is capped by `NEWS_BUDGET_MB`
(default 2048 MB). Easily manageable.
