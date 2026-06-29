"""Live tournament accuracy metrics for the whitepaper.

Computes three families of accuracy figures after each WC matchday:

1. **Model-level accuracy:** RMSE and rank correlation between each
   backend's predicted_points and the realised round_points for every
   (player, round) row.
2. **Squad-level performance:** what the user actually scored each
   round vs the model's recommended squad's hypothetical score, vs
   a random-pick baseline.
3. **Captain accuracy:** how often the model's top captain pick
   matched the actually-best-performing XI player, and the EV gap
   when they disagreed.

The user has been overriding model recommendations based on Diego
Guajardo's domain input. This module separates "what the model would
have done" from "what was actually picked" so the whitepaper can
report both honestly.

Inputs:
    - Per-round realised points: from `data/raw/players_<date>.parquet`,
      column `round_points` (list[int] indexed by round position).
    - Per-round model predictions: from
      `data/processed/predictions_<date>.parquet`, column
      `predicted_points`, filtered to the round being evaluated.
    - User's actual squad: declared in `results/<host>_recommendation_*.json`
      or in a per-round override file `data/user_squads/round_<n>.json`.

Outputs:
    - `data/evaluation/accuracy_<date>.json`: per-round summary.
    - `data/evaluation/leaderboard.csv`: cumulative table across rounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EVAL_DIR = Path("data/evaluation")
DEFAULT_USER_SQUADS_DIR = Path("data/user_squads")


@dataclass(frozen=True)
class ModelAccuracy:
    """Per-(backend, position) accuracy for one round."""
    round_id: int
    backend: str
    position: str
    n: int
    rmse: float
    mae: float
    spearman_rho: float       # rank-correlation between predicted and realised


@dataclass(frozen=True)
class SquadPerformance:
    """One row per (round, source) where source is one of:
    'user_actual', 'model_<backend>', 'random_baseline'."""
    round_id: int
    source: str
    starter_points: int
    captain_id: int | None
    captain_raw: int
    captain_doubled: int
    total: int                 # starter_points + captain_raw (captain counted twice net)


@dataclass(frozen=True)
class CaptainAnalysis:
    """How the model's captain pick fared vs the actual best XI scorer."""
    round_id: int
    backend: str
    chosen_captain_id: int
    chosen_captain_points: int
    best_xi_player_id: int
    best_xi_player_points: int
    captain_choice_was_optimal: bool
    ev_gap: int                # how much was left on the table


def _round_points_for(player_row, round_id: int) -> int:
    """Extract realised points for a given round_id from the player row.

    round_points is stored as a list indexed by round position (1-based:
    round_id N is at index N-1).
    """
    rp = player_row.get("round_points")
    if rp is None:
        return 0
    rp = list(rp)
    idx = round_id - 1
    if 0 <= idx < len(rp):
        return int(rp[idx])
    return 0


def model_accuracy(predictions: pd.DataFrame,
                   actuals: pd.DataFrame,
                   round_id: int,
                   backend: str) -> list[ModelAccuracy]:
    """Compute per-position RMSE/MAE/Spearman for one backend on one round.

    `predictions` must have columns: player_id, position, predicted_points,
    plus a row per (player, round). `actuals` is the latest
    `players_*.parquet` from the collector, which has `round_points`.
    """
    p = predictions[predictions["round_id"] == round_id].copy()
    if p.empty:
        return []
    a = actuals[["player_id", "round_points"]].copy()
    a["realised"] = a.apply(lambda r: _round_points_for(r, round_id), axis=1)
    a = a[["player_id", "realised"]]
    merged = p.merge(a, on="player_id", how="inner")
    merged = merged[merged["predicted_points"].notna()]

    rows: list[ModelAccuracy] = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        sub = merged[merged["position"] == pos]
        if len(sub) < 2:
            continue
        pred = sub["predicted_points"].astype(float).to_numpy()
        real = sub["realised"].astype(float).to_numpy()
        diff = pred - real
        rmse = float(np.sqrt(np.mean(diff * diff)))
        mae = float(np.mean(np.abs(diff)))
        # Spearman: rank-correlation.
        rank_pred = pd.Series(pred).rank()
        rank_real = pd.Series(real).rank()
        rho = float(rank_pred.corr(rank_real, method="pearson"))
        rows.append(ModelAccuracy(
            round_id=round_id, backend=backend, position=pos,
            n=len(sub), rmse=rmse, mae=mae, spearman_rho=rho,
        ))
    return rows


def squad_performance(squad_player_ids: list[int],
                      starter_ids: list[int],
                      captain_id: int,
                      actuals: pd.DataFrame,
                      round_id: int,
                      source: str) -> SquadPerformance:
    """Compute total points for one squad's lineup on one round.

    Auto-substitution is NOT modelled here; pass the realised starter
    list (which the FIFA Fantasy game determines for the user). Captain
    is counted twice: once via `starter_points`, once again via
    `captain_doubled - captain_raw`.
    """
    a = actuals[actuals["player_id"].isin(squad_player_ids)].copy()
    a["realised"] = a.apply(lambda r: _round_points_for(r, round_id), axis=1)
    starter_points = int(a[a["player_id"].isin(starter_ids)]["realised"].sum())
    cap_row = a[a["player_id"] == captain_id]
    cap_raw = int(cap_row["realised"].iloc[0]) if len(cap_row) else 0
    cap_doubled = cap_raw * 2
    total = starter_points + cap_raw   # captain raw counted again
    return SquadPerformance(
        round_id=round_id, source=source,
        starter_points=starter_points,
        captain_id=captain_id,
        captain_raw=cap_raw,
        captain_doubled=cap_doubled,
        total=total,
    )


def captain_analysis(predictions: pd.DataFrame,
                     actuals: pd.DataFrame,
                     starter_ids: list[int],
                     chosen_captain_id: int,
                     round_id: int,
                     backend: str) -> CaptainAnalysis:
    """How far was the chosen captain from the actual best XI scorer?"""
    a = actuals[actuals["player_id"].isin(starter_ids)].copy()
    a["realised"] = a.apply(lambda r: _round_points_for(r, round_id), axis=1)
    cap_row = a[a["player_id"] == chosen_captain_id]
    cap_pts = int(cap_row["realised"].iloc[0]) if len(cap_row) else 0
    best = a.sort_values("realised", ascending=False).head(1)
    best_id = int(best["player_id"].iloc[0]) if len(best) else chosen_captain_id
    best_pts = int(best["realised"].iloc[0]) if len(best) else cap_pts
    return CaptainAnalysis(
        round_id=round_id, backend=backend,
        chosen_captain_id=chosen_captain_id,
        chosen_captain_points=cap_pts,
        best_xi_player_id=best_id,
        best_xi_player_points=best_pts,
        captain_choice_was_optimal=(chosen_captain_id == best_id),
        ev_gap=(best_pts - cap_pts),
    )


def random_baseline_score(actuals: pd.DataFrame,
                          round_id: int,
                          budget: float = 100.0,
                          n_trials: int = 1000,
                          seed: int = 42) -> dict:
    """Sample N random valid squads, compute median XI score.

    A 'valid' random squad: 2 GK, 5 DEF, 5 MID, 3 FWD; under budget;
    at most 3 per country. We do NOT optimise; just sample. The median
    over N trials is the baseline number we report against.

    Returns dict with `mean`, `median`, `p10`, `p90`, `n_valid_trials`.
    """
    rng = np.random.default_rng(seed)
    a = actuals.copy()
    a["realised"] = a.apply(lambda r: _round_points_for(r, round_id), axis=1)
    eligible = a[(a["is_eliminated"] == False) & (a["status"] == "playing")]
    pos_counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    totals = []
    attempts = 0
    while len(totals) < n_trials and attempts < n_trials * 20:
        attempts += 1
        # Sample by position.
        squad_ids: list[int] = []
        ok = True
        for pos, n in pos_counts.items():
            cand = eligible[eligible["position"] == pos]
            if len(cand) < n:
                ok = False; break
            picked = rng.choice(cand["player_id"].values, size=n, replace=False)
            squad_ids.extend(int(x) for x in picked)
        if not ok:
            continue
        sq = a[a["player_id"].isin(squad_ids)]
        # Budget and country-cap check.
        if float(sq["price_millions"].sum()) > budget:
            continue
        country_counts = sq.groupby("country_abbr").size()
        if (country_counts > 3).any():
            continue
        # Pick a random 3-4-3 XI to keep it simple: top 3 DEF, top 4 MID,
        # top 3 FWD by realised (lookahead, but this is the BASELINE so OK).
        # Use random selection instead to be a true baseline.
        starter_ids = []
        for pos, n in {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}.items():
            cand = sq[sq["position"] == pos]
            picked = rng.choice(cand["player_id"].values, size=n, replace=False)
            starter_ids.extend(int(x) for x in picked)
        starters = sq[sq["player_id"].isin(starter_ids)]
        cap_pid = int(rng.choice(starters["player_id"].values))
        cap_pts = int(starters[starters["player_id"] == cap_pid]["realised"].iloc[0])
        starter_pts = int(starters["realised"].sum())
        totals.append(starter_pts + cap_pts)
    arr = np.array(totals)
    return {
        "n_valid_trials": int(len(totals)),
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "median": float(np.median(arr)) if len(arr) else 0.0,
        "p10": float(np.percentile(arr, 10)) if len(arr) else 0.0,
        "p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
    }


def emit_report(round_id: int,
                model_accuracies: list[ModelAccuracy],
                squad_performances: list[SquadPerformance],
                captain_analyses: list[CaptainAnalysis],
                random_baseline: dict,
                out_dir: Path = DEFAULT_EVAL_DIR) -> Path:
    """Persist the round's full evaluation as JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "round_id": round_id,
        "model_accuracies": [asdict(r) for r in model_accuracies],
        "squad_performances": [asdict(r) for r in squad_performances],
        "captain_analyses": [asdict(r) for r in captain_analyses],
        "random_baseline": random_baseline,
    }
    path = out_dir / f"accuracy_round_{round_id:02d}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path
