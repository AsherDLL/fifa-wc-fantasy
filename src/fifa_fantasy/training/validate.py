"""Held-out validation of every predictor backend on real labels.

Strategy: use the most recent EPL season (2024-25) split at gameweek
30. Train the GBM on every row from earlier seasons plus
gameweeks 1-29 of 2024-25; predict gameweeks 30-38; measure RMSE per
position against the realised `total_points`.

The heuristic and Poisson backends are pure functions that do not learn
from EPL, so they are evaluated as-is on the same held-out 30-38 rows.
All three numbers come from the same labels and the same rows; the
RMSE comparison is fair.

Output: a small table printed to stdout, plus a JSON sidecar saved
next to the held-out parquet so a UI can read it later.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fifa_fantasy.training.features import build_training_table

DEFAULT_TRAINING_DIR = Path("data/training")
DEFAULT_REPORT = Path("data/training/validation_report.json")


@dataclass(frozen=True)
class PositionRMSE:
    position: str
    n: int
    heuristic_rmse: float
    poisson_rmse: float
    gbm_rmse: float


def load_all_seasons(training_dir: Path = DEFAULT_TRAINING_DIR
                     ) -> pd.DataFrame:
    frames = []
    for path in sorted(training_dir.glob("fpl_player_gameweek_*.parquet")):
        df = pd.read_parquet(path)
        if "season" not in df.columns:
            df["season"] = path.stem.replace("fpl_player_gameweek_", "")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def split_train_holdout(df: pd.DataFrame,
                        holdout_season: str = "2024-25",
                        holdout_gw_min: int = 30,
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = ~(
        (df["season"] == holdout_season)
        & (df["gameweek"] >= holdout_gw_min)
    )
    return df[train_mask].copy(), df[~train_mask].copy()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
) -> list[PositionRMSE]:
    """Train the GBM on train_df, score everything on holdout_df."""
    from fifa_fantasy.model.baseline import heuristic_predict
    from fifa_fantasy.model.gbm import TrainConfig, train_all
    from fifa_fantasy.model.poisson import poisson_predict

    train_t = build_training_table(train_df)
    holdout_t = build_training_table(holdout_df)

    cfg = TrainConfig(n_estimators=400, learning_rate=0.05)
    models = train_all(train_t, cfg)

    # Build the inference-side feature frame holdout_t already has the
    # right columns, but heuristic_predict and poisson_predict also need
    # `position` (already present), `status`, `is_eliminated`,
    # `ownership_fraction`. Fill neutral defaults.
    inf = holdout_t.copy()
    inf["status"] = "playing"
    inf["is_eliminated"] = False
    inf["ownership_fraction"] = 0.10
    inf["rank_diff"] = float("nan")  # heuristic falls back to price signal

    pred_h = heuristic_predict(inf)["predicted_points"].to_numpy()
    pred_p = poisson_predict(inf)["predicted_points"].to_numpy()

    # GBM head returns predicted_points on its own contract; we recreate it.
    from fifa_fantasy.model.gbm import predict as gbm_predict
    pred_g = gbm_predict(inf, models)["predicted_points"].to_numpy()

    y_true = inf["target"].astype(float).to_numpy()
    rows = []
    for pos in ("GK", "DEF", "MID", "FWD"):
        mask = (inf["position"] == pos).to_numpy()
        if not mask.any():
            continue
        rows.append(PositionRMSE(
            position=pos,
            n=int(mask.sum()),
            heuristic_rmse=_rmse(y_true[mask], pred_h[mask]),
            poisson_rmse=_rmse(y_true[mask], pred_p[mask]),
            gbm_rmse=_rmse(y_true[mask], pred_g[mask]),
        ))
    return rows


def save_report(rows: list[PositionRMSE],
                path: Path = DEFAULT_REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"position_rmse": [asdict(r) for r in rows]}
    path.write_text(json.dumps(payload, indent=2))
    return path
