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

Every `SNAPSHOT_INTERVAL_HOURS` (default 6) the container:

1. Pulls the FIFA Fantasy API (players, squads, fixtures) and saves a
   dated Parquet under `data/raw/`.
2. Refreshes the martj42 international Elo CSV and the Polymarket /
   Kalshi WC 2026 prediction-market snapshot under `data/external/`.
3. Rebuilds the per-(player, round) features Parquet under
   `data/processed/`.
4. Runs all three predictor backends and writes their `predictions_*.parquet`.
5. Runs the optimizer for the current stage (env var `STAGE`) and
   writes the recommendation JSON+MD under `results/`.
6. Regenerates `results/index.html`.

Errors in any single stage are logged and do not block the rest of the
tick. The loop exits cleanly after `WC_END_DATE` (default 2026-07-18).

## Environment variables

| Variable | Default | What |
|---|---|---|
| `SNAPSHOT_INTERVAL_HOURS` | 6 | Hours between full ticks |
| `WC_END_DATE` | 2026-07-18 | Loop exits after this UTC date |
| `STAGE` | R32 | Which stage config the optimizer runs for |

Override at compose run time:

```bash
SNAPSHOT_INTERVAL_HOURS=3 STAGE=R16 docker compose --profile snapshot up -d
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

A single tick writes about 1.5 MB of Parquet (raw + features +
predictions) plus a few KB of JSON results. Over the remaining
tournament (~3 weeks at 6h ticks = ~84 ticks): roughly 130 MB. Easily
manageable. The prediction-market JSONL snapshots add ~200 KB each.
