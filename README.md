# FIFA Fantasy World Cup 2026

A prediction, optimization, and decision-support system for the official
FIFA World Cup 2026 Fantasy game. Predicts per-player fantasy points
with several competing models, selects the optimal 15-player squad under
stage-aware constraints, advises on in-round captain switches and
substitutions, and serves an analyst dashboard that keeps the whole
operation inspectable.

Design background is in [`fifa-fantasy-project-sketch.md`](./fifa-fantasy-project-sketch.md);
[`docs/`](./docs) holds the rules reference, scoring contract, design
decisions and operational notes; [`docs/whitepaper/`](./docs/whitepaper)
is the full scholarly write-up.

## Quickstart

Requires Python 3.12. The repo ships a pre-tournament data snapshot, the
EPL training parquet and 16 trained LightGBM models, so a fresh clone
works offline:

```bash
git clone <repo> && cd fifa_wc_fantasy
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest                                # hermetic test suite, no network
python -m fifa_fantasy.optimizer      # squad from the bundled predictions
python -m fifa_fantasy.web            # write the dashboard pages
xdg-open results/index.html           # macOS: open results/index.html
```

Continuous integration runs the same install and `pytest -m "not
network"` on every push (`.github/workflows/ci.yml`).

## The dashboard

Four pages, rendered server-side with inline styles and figures, no
external assets. They open equally well over HTTP and as plain files.

| Page | What it answers |
|---|---|
| `index.html` (Overview) | the official recommended squad for the current stage, decision tiles, news alerts touching the squad, run history |
| `algorithms.html` | one tab per backend: what it computes, the actual formula, where the ensemble routes it, its current squad and realized record |
| `intelligence.html` | crawled news signals per player with filters, market odds over time, corpus coverage |
| `research.html` | the model-evolution story: timeline, validation and walk-forward figures, calibration, negative results, reproduction commands |

The Docker snapshot daemon serves all four on port 8770, rendering fresh
on each request; `python -m fifa_fantasy.web` writes the same pages into
`results/` as static files. The figures and the aggregate dataset behind
them are rebuilt by `python -m fifa_fantasy.report`.

## Validation: how the approaches compare on real labels

The last 9 gameweeks of the 2024-25 Premier League fantasy season were
held out from training and every backend was scored on the same rows.
Root mean squared error per position, lower is better:

| Position | n (rows held out) | heuristic | poisson | gbm |
|---|---|---|---|---|
| GK | 180 | 2.698 | **2.491** | 2.650 |
| DEF | 910 | **3.141** | 3.400 | 3.187 |
| MID | 1228 | 2.874 | 4.371 | **2.767** |
| FWD | 368 | 3.515 | 4.560 | **3.153** |

No backend wins everywhere. Poisson takes goalkeepers, the heuristic
takes defenders, the GBM takes midfielders and forwards, and the
production `ensemble` backend is exactly that routing applied per
position. The leak-free walk-forward validation on realized World Cup
rounds, the calibration record and the negative-results ledger live on
the dashboard's Research page and in the whitepaper.

Reproduce locally with:

```bash
python -m fifa_fantasy.training.validate_main
```

See [`docs/algorithms-explained.md`](./docs/algorithms-explained.md) for
a beginner-friendly explanation of what every term in that table means
(EPL, GW, RMSE, held-out), what each predictor actually does, and how
they differ from each other and from neural networks. The executed
notebook [`notebooks/model_evolution.ipynb`](./notebooks/model_evolution.ipynb)
walks the full evolution story with the same code the dashboard uses.

## Approaches available

Six predictor backends, selectable via `--backend`: `heuristic`,
`heuristic_v2`, `gbm`, `poisson`, `monte_carlo`, and `ensemble`. The
per-position `ensemble` (Poisson GK, heuristic DEF, GBM MID/FWD) is the
production default the optimizer consumes. Each backend maps the
per-(player, round) feature table into a `predicted_points` column for
the full squad (not partial, not captain-only). The three core
approaches are summarised below; full write-up in
[`docs/approaches.md`](./docs/approaches.md).

| Backend | Idea | Trained on | When to use |
|---|---|---|---|
| `heuristic` (default) | position_coef x price x matchup x home, with a small premium-tier knob | nothing | conservative, transparent, the right call for MD1 submission |
| `gbm` | LightGBM mean + q10/q50/q90 per position | three seasons of Premier League FPL (2022-23, 2023-24, 2024-25 via the vaastav community mirror plus the live API) plus completed WC rounds when retrained with `--include-wc` | wins midfielder and forward on held-out EPL RMSE; competitive at GK and DEF. Lighter hyperparameters than v1 (15 leaves, 200 trees), tuned by sweep |
| `poisson` | structural Poisson goals: team xG, per-position goal/assist share, clean-sheet probability | nothing | independent of price and of EPL; a useful third opinion when the heuristic and the GBM disagree |

Pick a backend on the model CLI:

```bash
python -m fifa_fantasy.model                     # default: heuristic
python -m fifa_fantasy.model --backend gbm       # LightGBM
python -m fifa_fantasy.model --backend poisson   # structural goals
python -m fifa_fantasy.model --backend ensemble  # per-position routing (production default)
python -m fifa_fantasy.optimizer                 # consumes the latest predictions, whichever backend wrote them
```

The backend is stamped into the predictions Parquet, the recommendation
filename, the JSON `model_backend` field, and the first lines of the
markdown report. There is no ambiguity about which approach produced a
given recommendation.

## Round-by-round command reference

Exactly which optimizer command to run for each round of the tournament,
keyed to Fantasy.md's transfer rules. Replace `<prev>` with the path of
the most recent recommendation JSON from the previous round.

| Round | Command |
|---|---|
| Before MD1 (initial squad) | `python -m fifa_fantasy.optimizer` |
| Before MD2 | `python -m fifa_fantasy.optimizer --stage GROUP_MD2 --from <prev>` |
| Before MD2 with one rolled-over transfer | `python -m fifa_fantasy.optimizer --stage GROUP_MD2 --from <prev> --rolled-over 1` |
| Before MD3 | `python -m fifa_fantasy.optimizer --stage GROUP_MD3 --from <prev>` |
| Before R32 (knockout reset, unlimited transfers) | `python -m fifa_fantasy.optimizer --stage R32` |
| Before R16 | `python -m fifa_fantasy.optimizer --stage R16 --from <prev>` |
| Before QF | `python -m fifa_fantasy.optimizer --stage QF --from <prev>` |
| Before SF | `python -m fifa_fantasy.optimizer --stage SF --from <prev>` |
| Before FINAL | `python -m fifa_fantasy.optimizer --stage FINAL --from <prev>` |

In every case, refresh the underlying data first by running
`./scripts/daily-snapshot.sh`, which chains collector + features +
model + optimizer. The `--from` flag is what activates transfer-mode
planning (with the -3 hit penalty per transfer above the free quota);
omitting it solves a fresh selection, which is correct only for MD1
and R32 because those are the two stages with unlimited transfers.

The `model_backend` you used to predict propagates automatically into
the output filename and JSON. To compare backends side by side:

```bash
python -m fifa_fantasy.model --backend heuristic && python -m fifa_fantasy.optimizer
python -m fifa_fantasy.model --backend poisson   && python -m fifa_fantasy.optimizer
python -m fifa_fantasy.model --backend gbm       && python -m fifa_fantasy.optimizer
python -m fifa_fantasy.web
xdg-open results/index.html
```

## Browsing results

After running the optimizer one or more times, rebuild the report
dataset and the pages, then open any of them locally:

```bash
python -m fifa_fantasy.report   # results/report/report_data.json + results/figures/*.svg
python -m fifa_fantasy.web      # results/{index,algorithms,intelligence,research}.html
xdg-open results/index.html     # macOS: open results/index.html
```

The Overview page features the official squad (the latest run of the
production backend, ensemble by default) and links every archived run's
`.json` and `.md`. The Docker snapshot daemon serves the same four pages
on port 8770, rendering fresh on each request; the static files remain a
fallback you can open directly.

## Status

| Phase | What it does | State |
|---|---|---|
| 0 Scoring rules | Canonical FIFA Fantasy scoring as pure functions | done |
| 1 Collector | Players, squads, fixtures from `play.fifa.com` | done |
| 1b Live stats | total/last-round/form/round_points in the same Parquet | done |
| 2 Features | Per-(player, round) table with rest days and matchup signal | done |
| 3a Predictor | Heuristic, price-coef x matchup x home, with optional premium tilt | done |
| 3b Predictor | LightGBM mean + q10/q50/q90 per position, trained on three EPL FPL seasons plus completed WC rounds | done (opt-in via `--backend gbm`) |
| 3c Predictor | Structural Poisson goals: team xG, per-position share, clean sheets | done (opt-in via `--backend poisson`) |
| 4 Optimizer | PuLP MILPs: squad, transfer with -3 hit, lineup + captain | done |
| 4.5 Polish | `--compare-to`, `--report-alternatives`, `oneToWatch` flag | done |
| 4.7 Strength | FIFA World Ranking blended into the matchup multiplier | done |
| 5 Live tools | Captain playbook, captain switcher, sub advisor | done |
| 6 Ensemble | Per-position routing to the held-out winner per position | done (production default) |
| 7 Reporting | Model registry, reconciliation dataset, figures, four-page dashboard | done |

## Setup notes beyond the quickstart

`pip install -e ".[dev]"` resolves current versions from
`pyproject.toml`; `requirements.txt` is the generated lockfile the
Docker image installs, regenerate it with `pip freeze` after dependency
changes (its header has the exact recipe). Optional extras:
`.[sources]` adds the soccerdata team-news parser, `.[analysis]` adds
jupyter for the companion notebook.

```bash
python -m fifa_fantasy.optimizer                  # uses the bundled predictions; writes results/
python -m fifa_fantasy.optimizer --stage GROUP_MD1   # explicit stage
python -m fifa_fantasy.model --backend gbm        # rerun the GBM if you want
```

To refresh against the live FIFA Fantasy API:

```bash
./scripts/daily-snapshot.sh
```

To re-scrape EPL and retrain the GBM:

```bash
python -m fifa_fantasy.training --season 2024-25  # ~1 minute
python -m fifa_fantasy.model.train                # ~10 seconds
```

To retrain on EPL plus realised WC labels (works only after at least
one round of WC matches has finished):

```bash
python -m fifa_fantasy.model.train --include-wc
```

To rebuild the dashboard (a set of files you can open in any browser;
the snapshot daemon additionally serves live-rendered copies on port
8770):

```bash
python -m fifa_fantasy.report
python -m fifa_fantasy.web
open results/index.html       # or just double-click it
```

## What each tool does

| Tool | Reads | Writes |
|---|---|---|
| `python -m fifa_fantasy.collector` | `play.fifa.com/json/fantasy/*` | `data/raw/{players,squads,fixtures}_<UTC-date>.parquet` plus verbatim JSON under `data/raw/raw/` |
| `python -m fifa_fantasy.features` | latest Parquet under `data/raw/` plus `data/static/fifa_rankings.csv` | `data/processed/features_<UTC-date>.parquet` |
| `python -m fifa_fantasy.model [--backend heuristic\|gbm]` | latest features Parquet (and `data/models/` for `--backend gbm`) | `data/processed/predictions_<UTC-date>.parquet` (with a `model_backend` column) |
| `python -m fifa_fantasy.training` | `fantasy.premierleague.com/api` | `data/training/fpl_player_gameweek_<season>.parquet` |
| `python -m fifa_fantasy.model.train [--include-wc]` | latest EPL training Parquet (plus `data/raw/` for WC labels) | `data/models/gbm_<position>_<head>.txt` |
| `python -m fifa_fantasy.optimizer` | latest predictions Parquet (and optional previous recommendation JSON via `--from`) | `results/<host>_recommendation_<backend>_<STAGE>_<UTC-timestamp>.{json,md}` |
| `python -m fifa_fantasy.live` | a recommendation JSON, latest collector and predictions Parquet | `results/<host>_live_<STAGE>_R<n>_<UTC-timestamp>.md` |
| `python -m fifa_fantasy.report` | evaluation JSONs, predictions and signal parquet, market jsonl | `results/report/report_data.json` plus `results/figures/*.svg` |
| `python -m fifa_fantasy.web` | every JSON under `results/`, report dataset, figures, signal parquet | `results/{index,algorithms,intelligence,research}.html` (static; open in a browser) |

Output filenames make every dimension visible: the host that ran the
pipeline, the model backend that produced the predictions
(`heuristic`, `poisson`, `gbm`, `ensemble`, ...), the tournament stage
(`GROUP_MD1`, `R32`, ...),
and a UTC timestamp. Two files per run (json + md) carry the same
data: the JSON is the structured payload for `python -m fifa_fantasy.web`
or any other consumer; the markdown is a human-readable squad table
with a fact-only title line. Both are intentionally LLM-free; the only
content is what the optimizer produced.

See [`docs/tournament.md`](./docs/tournament.md) for what `MD1`, `MD2`,
`R32` and friends mean (matchdays in the group stage and knockout
rounds; `MD` here stands for "matchday", not for the file extension
`.md`).

The wrapper `./scripts/daily-snapshot.sh` chains collector, features,
model and optimizer in order.

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

That executes collector, features, predictor, optimizer in order. Tilt
toward premium attackers with `PREMIUM_BOOST=0.4 ./scripts/daily-snapshot.sh`.

Run **once before lockout** (within an hour of the first MD1 kickoff):

```bash
./scripts/daily-snapshot.sh
python -m fifa_fantasy.live --recommendation results/<host>_recommendation_<backend>_GROUP_MD1_<date>.json
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
python -m fifa_fantasy.live --recommendation results/<host>_recommendation_<backend>_<STAGE>_<date>.json
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
    --from results/<host>_recommendation_<backend>_GROUP_MD1_<date>.json \
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
    --from results/<host>_recommendation_<backend>_R32_<date>.json
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
- [`docs/whitepaper/`](./docs/whitepaper) the full scholarly write-up, one file per section
- [`notebooks/model_evolution.ipynb`](./notebooks/model_evolution.ipynb) executed companion notebook for the Research page
