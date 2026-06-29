"""With-vs-without-prediction-market comparison framework.

For every model backend and every completed round, compare:
  (a) Model-only predictions (no market signal)
  (b) Model + Benter combiner predictions (with market signal)
against (c) realised round_points.

The output shows whether incorporating prediction-market data via the
combiner improves prediction accuracy (RMSE / MAE / Spearman rank
correlation) compared to the model alone.

This is the empirical answer to the academic question "do prediction
markets add value as a meta-signal in fantasy football prediction?"

Caveat: the combiner is in scaffold mode (β₂ = 0.20 prior, not fitted).
Post-tournament we'll refit β empirically; this run shows the
framework working and gives a preliminary indication. The structural
finding (e.g. Polymarket has signal on top-tier countries) is robust
to the exact β value.

Output:
  - per-round table to stdout
  - data/evaluation/with_vs_without_market.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fifa_fantasy.external.benter_combiner import BenterConfig, combine
from fifa_fantasy.external.prediction_markets import load_history
from fifa_fantasy.model.baseline import heuristic_predict
from fifa_fantasy.model.gbm import DEFAULT_MODELS_DIR, load_models, predict as gbm_predict
from fifa_fantasy.model.monte_carlo import mc_predict
from fifa_fantasy.model.poisson import poisson_predict

OUT_DIR = Path("data/evaluation")

ROUND_PLAN = [
    (1, "features_2026-06-08.parquet", "players_2026-06-23.parquet"),
    (2, "features_2026-06-18.parquet", "players_2026-06-23.parquet"),
    (3, "features_2026-06-23.parquet", "players_2026-06-29.parquet"),
]

BACKENDS = ("heuristic", "poisson", "gbm", "monte_carlo")


@dataclass(frozen=True)
class WvWRow:
    round_id: int
    backend: str
    n: int
    rmse_model: float
    rmse_combined: float
    mae_model: float
    mae_combined: float
    rho_model: float
    rho_combined: float
    delta_rmse: float       # combined - model; negative = combiner helps


def _round_points(raw_players: pd.DataFrame, round_id: int) -> dict[int, int]:
    out = {}
    for r in raw_players.itertuples():
        rp = list(r.round_points) if r.round_points is not None else []
        idx = round_id - 1
        out[int(r.player_id)] = int(rp[idx]) if 0 <= idx < len(rp) else 0
    return out


def _rmse(y, yhat):
    diff = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(y, yhat):
    diff = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(diff)))


def _spearman(y, yhat):
    ya = pd.Series(np.asarray(y, dtype=float)).rank()
    yh = pd.Series(np.asarray(yhat, dtype=float)).rank()
    return float(ya.corr(yh, method="pearson"))


def _predict(features: pd.DataFrame, backend: str) -> pd.DataFrame:
    if backend == "heuristic": return heuristic_predict(features)
    if backend == "poisson":   return poisson_predict(features)
    if backend == "gbm":
        models = load_models(DEFAULT_MODELS_DIR)
        return gbm_predict(features, models)
    if backend == "monte_carlo": return mc_predict(features)
    raise ValueError(backend)


def run() -> dict:
    market_snapshots = load_history()
    print(f"market snapshots loaded: {len(market_snapshots)} rows")
    config = BenterConfig()
    print(f"Benter combiner config: β₀={config.beta_0}, β₁={config.beta_1}, "
          f"β₂={config.beta_2} (scaffold priors)")
    rows: list[WvWRow] = []
    for round_id, feat_name, raw_name in ROUND_PLAN:
        feat_path = Path("data/processed") / feat_name
        raw_path = Path("data/raw") / raw_name
        if not feat_path.exists() or not raw_path.exists():
            print(f"\nROUND {round_id}: skipping (missing files)")
            continue
        feat = pd.read_parquet(feat_path)
        raw = pd.read_parquet(raw_path)
        realised_map = _round_points(raw, round_id)
        print(f"\n=== ROUND {round_id} ===")

        for backend in BACKENDS:
            try:
                preds = _predict(feat.copy(), backend)
            except Exception as e:
                print(f"  {backend}: skipped ({type(e).__name__})"); continue

            in_round = preds[preds["round_id"] == round_id].copy()
            in_round["realised"] = in_round["player_id"].map(realised_map)
            combined = combine(in_round, config=config, market_snapshots=market_snapshots)

            ok = combined["realised"].notna() & combined["predicted_points"].notna()
            c = combined[ok]
            if len(c) < 10: continue
            y = c["realised"].to_numpy()
            model_p = c["predicted_points"].to_numpy()
            comb_p = c["combined_predicted_points"].to_numpy()
            r = WvWRow(
                round_id=round_id, backend=backend, n=len(c),
                rmse_model=_rmse(y, model_p),
                rmse_combined=_rmse(y, comb_p),
                mae_model=_mae(y, model_p),
                mae_combined=_mae(y, comb_p),
                rho_model=_spearman(y, model_p),
                rho_combined=_spearman(y, comb_p),
                delta_rmse=_rmse(y, comb_p) - _rmse(y, model_p),
            )
            rows.append(r)
            verdict = ("IMPROVED" if r.delta_rmse < -0.005
                       else ("WORSE" if r.delta_rmse > 0.005 else "tie"))
            print(f"  {backend:<11} n={r.n:>4}  model RMSE={r.rmse_model:.4f}  "
                  f"combined RMSE={r.rmse_combined:.4f}  ΔRMSE={r.delta_rmse:+.4f}  {verdict}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "with_vs_without_market.json"
    out.write_text(json.dumps({
        "config": {"beta_0": config.beta_0, "beta_1": config.beta_1,
                   "beta_2": config.beta_2},
        "rows": [asdict(r) for r in rows],
    }, indent=2))
    print(f"\nreport -> {out}")

    # Summary by backend.
    print("\n=== SUMMARY: did the combiner help? ===")
    df = pd.DataFrame([asdict(r) for r in rows])
    if df.empty:
        print("  (no data)")
        return {"rows": []}
    summary = df.groupby("backend").agg(
        n_rounds=("round_id", "count"),
        mean_delta_rmse=("delta_rmse", "mean"),
        improved_rounds=("delta_rmse", lambda x: int((x < -0.005).sum())),
    )
    print(summary.to_string())
    return {"rows": [asdict(r) for r in rows]}


if __name__ == "__main__":
    run()
