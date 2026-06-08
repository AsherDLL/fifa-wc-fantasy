"""Small hyperparameter sweep for the LightGBM backend.

Trains a handful of configs on the train split and reports per-position
RMSE on the held-out 2024-25 GW 30-38 rows. Picks the best config by
overall RMSE (weighted by row count).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fifa_fantasy.model.gbm import TrainConfig, predict as gbm_predict, train_all
from fifa_fantasy.training.features import build_training_table
from fifa_fantasy.training.validate import _rmse, load_all_seasons, split_train_holdout


CONFIGS = [
    TrainConfig(n_estimators=200, learning_rate=0.05, num_leaves=15),
    TrainConfig(n_estimators=400, learning_rate=0.05, num_leaves=31),
    TrainConfig(n_estimators=600, learning_rate=0.03, num_leaves=31),
    TrainConfig(n_estimators=800, learning_rate=0.02, num_leaves=63),
    TrainConfig(n_estimators=400, learning_rate=0.05, num_leaves=63,
                min_child_samples=50),
]


@dataclass(frozen=True)
class SweepRow:
    config: TrainConfig
    rmse_per_pos: dict[str, float]
    overall_rmse: float


def sweep() -> list[SweepRow]:
    all_seasons = load_all_seasons()
    train_df, holdout_df = split_train_holdout(all_seasons)
    train_t = build_training_table(train_df)
    holdout_t = build_training_table(holdout_df)

    inf = holdout_t.copy()
    inf["status"] = "playing"
    inf["is_eliminated"] = False
    inf["ownership_fraction"] = 0.10
    inf["rank_diff"] = float("nan")

    rows = []
    for cfg in CONFIGS:
        models = train_all(train_t, cfg)
        preds = gbm_predict(inf, models)["predicted_points"].to_numpy()
        y = inf["target"].astype(float).to_numpy()
        per_pos = {}
        weighted = 0.0
        total = 0
        for pos in ("GK", "DEF", "MID", "FWD"):
            mask = (inf["position"] == pos).to_numpy()
            if not mask.any():
                continue
            r = _rmse(y[mask], preds[mask])
            per_pos[pos] = r
            weighted += (r ** 2) * mask.sum()
            total += int(mask.sum())
        overall = float(np.sqrt(weighted / total)) if total else float("nan")
        rows.append(SweepRow(cfg, per_pos, overall))
    return rows


def main() -> None:
    rows = sweep()
    print(f"{'config':<60} {'GK':>6} {'DEF':>6} {'MID':>6} {'FWD':>6} {'overall':>8}")
    print("-" * 100)
    best = None
    for r in rows:
        cfg = r.config
        cfg_str = (f"n={cfg.n_estimators} lr={cfg.learning_rate} "
                   f"leaves={cfg.num_leaves} mcs={cfg.min_child_samples}")
        per = r.rmse_per_pos
        line = (f"{cfg_str:<60}  "
                f"{per.get('GK', float('nan')):>6.3f} "
                f"{per.get('DEF', float('nan')):>6.3f} "
                f"{per.get('MID', float('nan')):>6.3f} "
                f"{per.get('FWD', float('nan')):>6.3f} "
                f"{r.overall_rmse:>8.4f}")
        print(line)
        if best is None or r.overall_rmse < best.overall_rmse:
            best = r
    print()
    print(f"best config: {best.config}")
    print(f"best overall RMSE: {best.overall_rmse:.4f}")


if __name__ == "__main__":
    main()
