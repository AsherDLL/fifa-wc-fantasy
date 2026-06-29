# Appendix — Reproducibility, schemas, and artefact formats

Status: **DRAFT**

## A. Full per-(player, round) feature schema

See `src/fifa_fantasy/features/build.py` for the source of truth. The
columns at inference time (after the country Elo join) are:

- Player metadata (16 columns): `player_id`, `first_name`, `last_name`,
  `known_name`, `full_name`, `position`, `squad_id`, `country`,
  `country_abbr`, `price_millions`, `ownership_fraction`, `status`,
  `is_eliminated`, `one_to_watch`, `one_to_watch_text`, plus `total_points`,
  `last_round_points`, `form`, `round_points`.
- Fixture context (9 columns): `fixture_id`, `round_id`, `stage`,
  `is_home`, `kickoff`, `opponent_squad_id`, `opponent_name`,
  `opponent_abbr`, `venue_name`, `venue_city`, `fixture_status`.
- Own-squad strength (6 columns): `squad_total_price`, `squad_avg_price`,
  `squad_top_n_avg_price`, `squad_top_n_rank`, `squad_rank_points`,
  `squad_rank_position`.
- Opponent-squad strength (6 columns): same fields prefixed `opp_`.
- Derived (10+ columns): `strength_diff`, `rank_diff`, `country_elo`,
  `opp_country_elo`, `country_elo_diff`, `country_last10_form`,
  `opp_country_last10_form`, `days_since_prev_match`,
  `days_to_next_match`.

Total: 51 columns at the most recent dump.

## B. LightGBM model artefact format

Per-position, per-head text files under `data/models/`:

```
gbm_{GK,DEF,MID,FWD}_{mean,q10,q50,q90}.txt
```

LightGBM's native text serialisation. Load with `lgb.Booster(model_file=path)`.

## C. Recommendation JSON schema

Per-run output under `results/`:

```json
{
  "stage": "GROUP_MD3",
  "model_backend": "heuristic",
  "model_version": "",
  "host": "linux-mint",
  "generated_at_utc": "2026-06-24T10:00:00Z",
  "horizon_rounds": [3],
  "budget_used": 99.5,
  "budget_total": 100.0,
  "total_horizon_points": 75.4,
  "net_horizon_points": 75.4,
  "squad_player_ids": [...],
  "squad": [
    {
      "player_id": 38,
      "full_name": "Lionel Messi",
      "first_name": "Lionel",
      "last_name": "Messi",
      "known_name": null,
      "position": "FWD",
      "country": "Argentina",
      "country_abbr": "ARG",
      "squad_id": 1,
      "price_millions": 10.0,
      "ownership_fraction": 0.285,
      "status": "playing",
      "is_eliminated": false,
      "one_to_watch": false,
      "one_to_watch_text": null,
      "form": 13.3,
      "role": "Captain",
      "in_starting_xi": true,
      "bench_priority": null,
      "opponent_abbr": "JOR",
      "is_home": true,
      "predicted_points": 8.55,
      "predicted_q10": null,
      "predicted_q50": null,
      "predicted_q90": null
    }
    // ... 14 more
  ],
  "lineup": {
    "round_id": 3,
    "formation": "3-4-3",
    "starter_ids": [...],
    "bench_ids_priority_order": [...],
    "captain_id": 38,
    "vice_captain_id": 517,
    "expected_points": 56.7
  },
  "transfer": {
    "from": "results/<prev>.json",
    "rolled_over_free_transfers": 0,
    "free_transfers_total": 2,
    "n_transfers": 2,
    "n_extra_transfers": 0,
    "transfer_cost_points": 0,
    "transfers_in": [38, 1710],
    "transfers_out": [505, 1338],
    "transfers_in_detail": [...],
    "transfers_out_detail": [...]
  }
}
```

The `predicted_q10`, `q50`, `q90` fields are populated only when the
GBM backend was used. The `transfer` block is absent for fresh squad
selections (e.g. MD1 and R32 Wildcard rounds).

## D. Reproducibility checklist

To reproduce a result file:

1. `git clone` the repo, `cd` into it, `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python -m fifa_fantasy.collector` (refresh FIFA API snapshot)
4. `python -m fifa_fantasy.external` (refresh martj42 + football-data)
5. `python -m fifa_fantasy.features`
6. `python -m fifa_fantasy.model --backend <heuristic|poisson|gbm>`
7. `python -m fifa_fantasy.optimizer --stage <STAGE>`

The result JSON lands under `results/`. Compare against the
hostname-prefixed result file we committed for the same date.

For the held-out validation:

1. Steps 1-2 above
2. `python -m fifa_fantasy.training.vaastav --season 2022-23`
3. Repeat for 2023-24 and 2024-25
4. `python -m fifa_fantasy.training.validate_main`

Output table goes to stdout; sidecar JSON to
`data/training/validation_report.json`.

## E. Versions of every external pin

```
python 3.12.x
pandas, pyarrow      (per requirements.txt)
pydantic >= 2.x
httpx                (HTTP client)
lightgbm             (gradient boosting)
pulp                 (MILP modelling)
jinja2               (HTML report)
pytest               (tests)
```

(Pin exact versions in `requirements.txt` at LaTeX conversion time.)
