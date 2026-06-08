# FIFA Fantasy World Cup 2026

A prediction, optimization, and live decision-support system for the
official FIFA World Cup 2026 Fantasy game. Predicts per-player fantasy
points, selects the optimal 15-player squad under stage-aware
constraints, and advises on in-round captain switches and substitutions.

Design background is in [`fifa-fantasy-project-sketch.md`](./fifa-fantasy-project-sketch.md);
[`docs/`](./docs) holds the rules reference, scoring contract, design
decisions and operational notes.

## Status

| Phase | What it does | State |
|---|---|---|
| 0 Scoring rules | Canonical FIFA Fantasy scoring as pure functions | done |
| 1 Collector | Players, squads, fixtures from `play.fifa.com` | done |
| 1b Live stats | total/last-round/form/round_points in the same Parquet | done |
| 2 Features | Per-(player, round) table with rest days and matchup signal | done |
| 3a Predictor | Heuristic, price-coef x matchup x home, with optional premium tilt | done |
| 3b Predictor | LightGBM quantile regressors (needs Euro 2024 data) | not started |
| 4 Optimizer | PuLP MILPs: squad, transfer with -3 hit, lineup + captain | done |
| 4.5 Polish | `--compare-to`, `--report-alternatives`, `oneToWatch` flag | done |
| 4.7 Strength | FIFA World Ranking blended into the matchup multiplier | done |
| 5 Live tools | Captain playbook, captain switcher, sub advisor | done |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## What each tool does

| Tool | Reads | Writes |
|---|---|---|
| `python -m fifa_fantasy.collector` | `play.fifa.com/json/fantasy/*` | `data/raw/{players,squads,fixtures}_<UTC-date>.parquet` plus verbatim JSON under `data/raw/raw/` |
| `python -m fifa_fantasy.features` | latest Parquet under `data/raw/` plus `data/static/fifa_rankings.csv` | `data/processed/features_<UTC-date>.parquet` |
| `python -m fifa_fantasy.model` | latest features Parquet | `data/processed/predictions_<UTC-date>.parquet` |
| `python -m fifa_fantasy.optimizer` | latest predictions Parquet (and optional previous recommendation JSON) | `results/<host>_recommendation_<STAGE>_<UTC-date>.{json,md}` |
| `python -m fifa_fantasy.live` | a recommendation JSON, latest collector and predictions Parquet | `results/<host>_live_<STAGE>_R<n>_<UTC-time>.md` |

All five compose. The wrapper `./scripts/daily-snapshot.sh` runs the
first four in order with sensible defaults and a yesterday-vs-today diff.

# Runbook

The cadence below is driven by Fantasy.md (transfer limits, lockout
mechanics) and the WC 2026 round schedule.

## What changes you can make at each lockout

Read this once before running anything. The system writes
recommendations but Fantasy enforces these constraints; running the tool
more often than allowed does not help.

| Stage | Free transfers | Roll-over | Wildcard allowed | Notes |
|---|---|---|---|---|
| Before MD1 (group round 1) | unlimited | n/a | no | initial squad selection, all 15 picked from scratch |
| Before MD2 | 2 | 1 from MD1 | yes | each extra transfer costs -3 |
| Before MD3 | 2 | 1 from MD2; cannot roll into R32 | yes | -3 per extra |
| Before R32 | unlimited | n/a | no | knockout reset, full rebuild allowed |
| Before R16 | 4 | none | yes | -3 per extra |
| Before QF | 4 | none | yes | -3 per extra |
| Before SF | 5 | none | yes | -3 per extra |
| Before Final | 6 | none | yes | -3 per extra |

Other constraints (in addition to the transfer limits):

- Budget: $100M group stage; $105M from R32 onward.
- Squad shape: 2 GK, 5 DEF, 5 MID, 3 FWD; starting XI in one of 7 valid formations.
- Nationality cap: 3 per country (group + R32); 4 (R16); 5 (QF); 6 (SF); 8 (Final).
- During a live round: captain can be switched unlimited times, but only when current captain's match is not in progress, and the new captain's match has not started. Switching forfeits the captain bonus on the old player.
- Manual substitutions during a live round CANCEL all automatic substitutions for that round.

## Pre-lockout cadence (now until June 11, 2026)

The MD1 lockout is at the kickoff of the first WC match. The pool grows
slightly as FIFA adds players, ownership shifts daily, and FIFA may flag
players as `oneToWatch` closer to lockout.

Run **once per day** (cron friendly):

```bash
./scripts/daily-snapshot.sh
```

That executes collector, features, predictor, optimizer in order; auto-
finds yesterday's recommendation under `results/` and emits a diff plus
an alternatives / sensitivity section. Tilt toward premium attackers
with `PREMIUM_BOOST=0.4 ./scripts/daily-snapshot.sh`.

Run **once before lockout** (within an hour of the first MD1 kickoff):

```bash
./scripts/daily-snapshot.sh
python -m fifa_fantasy.live --recommendation results/<host>_recommendation_GROUP_MD1_<date>.json
```

The live module produces the captain playbook: initial captain plus the
threshold chain that tells you when to switch in each subsequent
kickoff window.

## During each live round

A "live round" is the seven-day window from the first kickoff of the
round to the last whistle. During this window:

- Transfers for the next round are still possible but they apply to the
  next round, not the current XI.
- Captain switching is available in real time.
- Manual subs are available in real time (and cancel auto-subs).

Between kickoff windows (typically a 3 to 6 hour gap), run:

```bash
python -m fifa_fantasy.collector              # refresh live points
python -m fifa_fantasy.live --recommendation results/<host>_recommendation_<STAGE>_<date>.json
```

The live module will detect that at least one fixture has finished and
switch from "playbook" to "live" mode. It evaluates:

- Captain. If current captain finished and scored below the best
  unplayed alternate's expected points, recommend a switch.
- Subs. If any finished starter scored below an unplayed bench player's
  expected points, recommend the swap with the auto-sub cancellation
  warning attached.

You can also run the live tool before any match starts (playbook mode)
to keep the policy fresh as ownership shifts.

## Between rounds: planning transfers

After the current round ends and before the next round's lockout, plan
transfers from your previous squad:

```bash
./scripts/daily-snapshot.sh                   # refresh data

python -m fifa_fantasy.optimizer \
    --stage GROUP_MD2 \
    --from results/<host>_recommendation_GROUP_MD1_<date>.json \
    --rolled-over 1                           # if you carried a free transfer
```

The optimizer's MILP maximizes total horizon expected points minus the
hit penalty for any transfer above the quota. The markdown report adds
an OUT and IN section so the changes are obvious.

For the knockout transition (after MD3 ends and before R32 lockout):

```bash
# Unlimited transfers, so solve fresh; do not pass --from.
python -m fifa_fantasy.optimizer --stage R32
```

From R16 onward, transfer mode is the right choice again:

```bash
python -m fifa_fantasy.optimizer --stage R16 \
    --from results/<host>_recommendation_R32_<date>.json
```

## Refreshing the FIFA World Ranking

The strength signal blends the FIFA Men's World Ranking (snapshot under
`data/static/fifa_rankings.csv`) with the squad-price proxy. Refresh the
CSV whenever FIFA publishes a new ranking (roughly monthly). Country
names must match the FIFA Fantasy API spelling exactly; see
[`docs/strength-signals.md`](./docs/strength-signals.md).

## How often to run, summary

| Window | Cadence | Command |
|---|---|---|
| Pre-tournament (until June 11) | daily | `./scripts/daily-snapshot.sh` |
| Final pre-lockout check | once, within 1 hour of first kickoff | `./scripts/daily-snapshot.sh` then `python -m fifa_fantasy.live --recommendation <latest>.json` |
| During a live round | once per kickoff window gap | `python -m fifa_fantasy.collector` then `python -m fifa_fantasy.live --recommendation <latest>.json` |
| Between rounds | once daily; final run within 1 hour of next lockout | `./scripts/daily-snapshot.sh` then `python -m fifa_fantasy.optimizer --stage <NEXT> --from <previous>.json` |
| Knockout reset (before R32) | once | `python -m fifa_fantasy.optimizer --stage R32` |
| Anytime FIFA publishes new rankings | once per refresh | edit `data/static/fifa_rankings.csv`, then re-run features and onward |

## Where output lives

```
data/raw/                  collector parquet + verbatim JSON
data/processed/            features and predictions parquet
data/static/               FIFA ranking snapshot (hand-maintained)
results/                   one .json + .md per recommendation,
                           plus one .md per live decision
```

Files are prefixed with the hostname so different machines pushing to
the same git repo do not collide.

## Documentation index

- [`docs/Fantasy.md`](./docs/Fantasy.md) the official guidelines, verbatim
- [`docs/scoring-rules.md`](./docs/scoring-rules.md) scoring contract
- [`docs/operations.md`](./docs/operations.md) substitution and booster mechanics
- [`docs/features.md`](./docs/features.md) feature column dictionary
- [`docs/baseline.md`](./docs/baseline.md) heuristic predictor formula
- [`docs/strength-signals.md`](./docs/strength-signals.md) FIFA ranking blend
- [`docs/optimizer.md`](./docs/optimizer.md) MILP formulation, stage table
- [`docs/api-endpoints.md`](./docs/api-endpoints.md) FIFA Fantasy endpoints
- [`docs/decisions.md`](./docs/decisions.md) running design-decision log
- [`docs/pipeline-walkthrough.md`](./docs/pipeline-walkthrough.md) Lautaro Martinez traced through every step
