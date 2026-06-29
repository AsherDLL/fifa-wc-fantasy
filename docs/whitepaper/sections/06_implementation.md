# 06 — Implementation and architecture

Status: **DRAFT**

## 6.1 Repository layout

```
fifa_wc_fantasy/
  data/
    external/                Country and club Elo, football-data caches
    models/                  LightGBM artefacts (text format)
    processed/               Parquet rolls from features/models pipeline
    raw/                     Parquet rolls from collector
    static/                  fifa_rankings.csv (hand-maintained snapshot)
    training/                EPL per-(player, gameweek) Parquet dumps
  docs/                      Markdown documentation
  results/                   Per-run recommendation JSON+MD and HTML reports
  scripts/                   Ad-hoc analysis (transfer plans, dashboards, MC)
  src/fifa_fantasy/
    collector/               FIFA Fantasy API client + Pydantic schemas
    external/                martj42 + football-data fetchers, Elo derivation
    features/                Per-(player, round) feature table construction
    live/                    Captain switcher and sub advisor
    model/                   Heuristic, Poisson, GBM backends with shared
                             prediction contract
    optimizer/               MILP solvers and stage configuration
    training/                EPL data scraper, held-out validation,
                             hyperparameter sweep
    web/                     Static HTML report generator
    scoring.py               Canonical scoring functions
  tests/                     Pytest tests (~50 cases against scoring rules)
```

## 6.2 Pipeline dataflow

```
[ FIFA API ]              [ martj42 results ]    [ football-data CSVs ]
     |                            |                       |
     v                            v                       v
collector                     external                 external
     |                            |                       |
     v                            v                       v
data/raw/                  data/external/         data/external/
*.parquet                  country_elo.csv         fd_matches.parquet
                                                   club_elo.csv
     \                          /                       /
      v                        v                       v
                          features
                              |
                              v
                  data/processed/features_*.parquet
                              |
                              v
                model (heuristic | poisson | gbm)
                              |
                              v
                 data/processed/predictions_*.parquet
                              |
                              v
                         optimizer
                              |
                              v
                 results/<host>_recommendation_*.json
                          results/<host>_recommendation_*.md
                              |
                              v
                          web
                              |
                              v
                       results/index.html
```

The pipeline is idempotent: every stage reads from disk and writes to
disk, so any stage can be re-run with different parameters without
re-running the upstream stages. The collector is the only stage that
hits the network at refresh time; the external module hits the network
on `--no-refresh-cache=false`.

## 6.3 Engineering decisions worth flagging

**Parquet over CSV everywhere.** Faster I/O, smaller files, typed
columns. The only CSVs in the repo are hand-maintained static files
(`fifa_rankings.csv`, `country_elo.csv`).

**Hostname-prefixed timestamps on every result file.** Pattern:
`<host>_recommendation_<backend>_<stage>_<UTC-timestamp>.{json,md}`.
This prevents two collaborators from clobbering each other and makes the
results directory self-documenting.

**JSON is the contract for the web UI.** The recommendation JSON
includes the full squad array with per-player metadata, lineup and
captain decision, transfer block (when applicable), and quantile bands
when the GBM backend was used. The Markdown is just the squad table.

**Deterministic LightGBM.** We set `seed=42`, `bagging_seed=42`,
`feature_fraction_seed=42`, `deterministic=True` in `_shared_params`.
Without this, the held-out RMSE wobbles by ±0.05 per position run-to-run,
which masked the `team_elo_diff` A/B comparison (Section 7.4).

**Two-stage Pydantic validation.** `Raw*` models mirror the API
verbatim; a second normalisation pass produces the typed contract used
downstream. New API fields land as `extra="ignore"` and only break the
parser if a required field changes shape.

**MILP not greedy.** Squad selection has too many constraints (budget,
country cap, position counts, transfer quota) for a greedy heuristic to
return the optimum reliably. PuLP's CBC backend solves the 1488-row,
binary-decision problem in under a second on commodity hardware.

## 6.4 Reproducibility

- `requirements.txt` pins every direct dependency to a known-good
  version
- LightGBM training is deterministic given the seed (Section 6.3)
- Static input snapshots (`fifa_rankings.csv`, the cached martj42 CSV,
  the football-data CSVs) live in the repo or under `data/external/cache/`
- The pre-tournament FIFA Fantasy snapshot is committed so a fresh clone
  can run the full pipeline end-to-end without re-scraping
- Random Monte Carlo simulations in `scripts/md*.py` use
  `np.random.default_rng(seed=42)`

## 6.5 Testing strategy

The bulk of the test surface is in `tests/test_scoring.py`: about 50
parametrised test cases that pin every per-position scoring component
(appearance, goal scored, assist, clean sheet, goals conceded penalty,
saves bonus, tackles, chances created, shots on target, scouting bonus
threshold conditions). Realistic end-to-end scenarios (Mbappé scoring
two, Donnarumma keeping a clean sheet with five saves) are also pinned
to known totals.

The collector, features, and optimiser modules are tested indirectly
via the end-to-end pipeline runs. The held-out validation
(`training/validate.py`) is the model-correctness regression: any change
to a backend that drifts RMSE more than 0.05 in either direction is
flagged in the CI output.
