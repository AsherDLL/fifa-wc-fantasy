"""LightGBM predictor: per-position point + quantile heads.

Four position models trained on EPL FPL data (one season is plenty for
shape; more seasons land later). Each position trains:

  - mean head: regression for E[total_points]
  - q10 head: quantile regression at the 10th percentile
  - q50 head: quantile regression at the 50th percentile (median)
  - q90 head: quantile regression at the 90th percentile

Feature columns match the WC inference table so the same model can score
both data sets:

    price_millions, is_home (0/1), strength_diff,
    squad_top_n_avg_price, opp_squad_top_n_avg_price

`rank_diff` is included at WC inference but absent in training; the
inference path drops it before passing rows to the model, so the column
list above is the contract.

Model artefacts persist to `data/models/gbm_<position>_<head>.txt` via
LightGBM's native model format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

DEFAULT_MODELS_DIR = Path("data/models")
POSITIONS = ("GK", "DEF", "MID", "FWD")
HEADS = ("mean", "q10", "q50", "q90")
QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}

# Version string written to predictions and recommendation outputs so
# legacy v1 results (single season, default hyperparameters) can be told
# apart from current v2 results (three seasons, tuned hyperparameters).
# Bump this any time the training data or hyperparameters change in a
# way that changes the squad picks.
GBM_VERSION = "v2"

FEATURE_COLUMNS = [
    "price_millions",
    "is_home",
    "strength_diff",
    "squad_top_n_avg_price",
    "opp_squad_top_n_avg_price",
]
# team_elo_diff was tested as a sixth feature (v3 candidate). With a
# deterministic A/B on EPL 2024-25 GW 30-38: GK -0.018, MID -0.019 (better);
# DEF +0.033, FWD +0.015 (worse). Net wash; the extra feature risks
# distribution shift to international Elo gaps (much larger than EPL gaps)
# at WC inference, so v3 was not shipped. The data and column live on for
# the heuristic and Poisson backends, where the signal is consumed directly.


@dataclass(frozen=True)
class TrainConfig:
    """Defaults picked by held-out RMSE sweep on EPL 2024-25 GW 30-38.

    Lighter than the original (15 leaves, 200 estimators) beat heavier
    configs at every position; the original 31/400 was overfitting.
    """
    num_leaves: int = 15
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_child_samples: int = 30
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    verbose: int = -1


def _shared_params(cfg: TrainConfig) -> dict:
    return {
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_child_samples": cfg.min_child_samples,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "verbose": cfg.verbose,
        "force_row_wise": True,
        # Determinism: without this, bagging RNG drifts run-to-run and the
        # held-out RMSE wobbles ~0.05 per position; that masks small
        # feature-engineering gains. Same seed across heads is fine; the
        # heads have different objectives so they still learn different
        # trees.
        "seed": 42,
        "bagging_seed": 42,
        "feature_fraction_seed": 42,
        "deterministic": True,
    }


def _to_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_home"] = out["is_home"].astype(int)
    return out[FEATURE_COLUMNS]


def train_one(
    train_df: pd.DataFrame,
    position: str,
    cfg: TrainConfig = TrainConfig(),
) -> dict[str, lgb.Booster]:
    """Train mean + three quantile heads for a single position."""
    sub = train_df[train_df["position"] == position].copy()
    if sub.empty:
        raise ValueError(f"no training rows for position {position}")
    X = _to_features(sub)
    y = sub["target"].astype(float)

    out: dict[str, lgb.Booster] = {}
    base = _shared_params(cfg)

    # Mean head: standard regression.
    out["mean"] = lgb.train(
        {**base, "objective": "regression", "metric": "rmse"},
        lgb.Dataset(X, label=y),
        num_boost_round=cfg.n_estimators,
    )
    # Quantile heads.
    for head, alpha in QUANTILES.items():
        out[head] = lgb.train(
            {**base, "objective": "quantile", "alpha": alpha, "metric": "quantile"},
            lgb.Dataset(X, label=y),
            num_boost_round=cfg.n_estimators,
        )
    return out


def train_all(train_df: pd.DataFrame,
              cfg: TrainConfig = TrainConfig(),
              ) -> dict[str, dict[str, lgb.Booster]]:
    return {pos: train_one(train_df, pos, cfg) for pos in POSITIONS}


def save_models(models: dict[str, dict[str, lgb.Booster]],
                out_dir: Path = DEFAULT_MODELS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pos, heads in models.items():
        for head, booster in heads.items():
            booster.save_model(str(out_dir / f"gbm_{pos}_{head}.txt"))


def load_models(in_dir: Path = DEFAULT_MODELS_DIR
                ) -> dict[str, dict[str, lgb.Booster]]:
    out: dict[str, dict[str, lgb.Booster]] = {}
    for pos in POSITIONS:
        heads: dict[str, lgb.Booster] = {}
        for head in HEADS:
            path = in_dir / f"gbm_{pos}_{head}.txt"
            if not path.exists():
                raise FileNotFoundError(f"missing {path}; run model.train first")
            heads[head] = lgb.Booster(model_file=str(path))
        out[pos] = heads
    return out


def predict(features: pd.DataFrame,
            models: dict[str, dict[str, lgb.Booster]],
            ) -> pd.DataFrame:
    """Add predicted_points + quantile columns to a copy of `features`.

    `features` is the per-(player, round) WC table from Phase 2 (it has
    all FEATURE_COLUMNS plus `position`). Players with non-playing status
    or eliminated squads are predicted at 0, mirroring the heuristic.
    """
    out = features.copy()
    out["is_home"] = out["is_home"].astype(int)
    X = out[FEATURE_COLUMNS]

    pred_mean = np.zeros(len(out))
    pred_q10 = np.zeros(len(out))
    pred_q50 = np.zeros(len(out))
    pred_q90 = np.zeros(len(out))
    for pos, heads in models.items():
        mask = (out["position"] == pos).to_numpy()
        if not mask.any():
            continue
        Xp = X[mask]
        pred_mean[mask] = heads["mean"].predict(Xp)
        pred_q10[mask] = heads["q10"].predict(Xp)
        pred_q50[mask] = heads["q50"].predict(Xp)
        pred_q90[mask] = heads["q90"].predict(Xp)

    available = (out["status"] == "playing") & (~out["is_eliminated"].astype(bool))
    out["predicted_points"] = np.where(available, np.clip(pred_mean, 0, None), 0.0)
    out["predicted_q10"] = np.where(available, np.clip(pred_q10, 0, None), 0.0)
    out["predicted_q50"] = np.where(available, np.clip(pred_q50, 0, None), 0.0)
    out["predicted_q90"] = np.where(available, np.clip(pred_q90, 0, None), 0.0)
    return out
