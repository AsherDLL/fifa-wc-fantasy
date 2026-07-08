"""Leak-free walk-forward validation of the GBM on realised WC rounds.

The EPL held-out validation (training/validate.py) proves a change helps
on Premier League labels. It cannot prove the change helps on the World
Cup, where the feature distribution shifts (international Elo gaps far
exceed club gaps; squad prices are compressed). This script measures
that directly.

For each completed WC round k (k >= 2, since round 1 has no prior form):

    train rows = EPL(all) + WC(rounds < k)
    holdout    = WC(round k), realised fantasy points as the label

We train three GBM configurations on the same rows and score them on the
same holdout, so the comparison is clean:

    A  epl_noform    EPL only,  features without form_lag   (the shipped v2)
    B  epl_form      EPL only,  features with form_lag
    C  eplwc_form    EPL + WC(<k), features with form_lag   (the candidate)

Output: per-round, per-position RMSE for each configuration, then the
pooled RMSE across all held-out rounds. Lower is better.

No file writes; prints to stdout. Deterministic (seed pinned in gbm).
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fifa_fantasy.model.gbm import HEADS, POSITIONS, QUANTILES, TrainConfig, _shared_params
from fifa_fantasy.training.features import build_training_table
from fifa_fantasy.training.wc import extract_wc_training_rows

TRAINING_DIR = Path("data/training")

BASE_FEATURES = [
    "price_millions",
    "is_home",
    "strength_diff",
    "squad_top_n_avg_price",
    "opp_squad_top_n_avg_price",
]
FORM_FEATURES = BASE_FEATURES + ["form_lag"]
ALL_FEATURES = FORM_FEATURES + ["start_rate_lag", "team_gc_form"]


def _train_mean_head(train_df: pd.DataFrame, position: str,
                     feature_cols: list[str]) -> lgb.Booster:
    sub = train_df[train_df["position"] == position]
    X = sub[feature_cols].copy()
    X["is_home"] = X["is_home"].astype(int)
    y = sub["target"].astype(float)
    base = _shared_params(TrainConfig())
    return lgb.train(
        {**base, "objective": "regression", "metric": "rmse"},
        lgb.Dataset(X, label=y),
        num_boost_round=TrainConfig().n_estimators,
    )


def _predict(booster: lgb.Booster, holdout: pd.DataFrame,
             feature_cols: list[str]) -> np.ndarray:
    X = holdout[feature_cols].copy()
    X["is_home"] = X["is_home"].astype(int)
    return np.clip(booster.predict(X), 0, None)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run() -> None:
    epl_raw = pd.concat(
        [pd.read_parquet(p) for p in sorted(TRAINING_DIR.glob("fpl_player_gameweek_*.parquet"))],
        ignore_index=True,
    )
    epl = build_training_table(epl_raw)  # has form_lag + target

    wc = extract_wc_training_rows()       # has form_lag + target + gameweek
    if wc.empty:
        raise SystemExit("no completed WC rounds yet")
    rounds = sorted(int(r) for r in wc["gameweek"].unique())
    holdout_rounds = [k for k in rounds if k >= 2]
    print(f"WC rounds present: {rounds}; validating on {holdout_rounds}")

    configs = {
        "A_epl_noform": (BASE_FEATURES, False),
        "B_epl_form": (FORM_FEATURES, False),
        "C_eplwc_form": (FORM_FEATURES, True),
        "D_eplwc_all": (ALL_FEATURES, True),
    }
    # Accumulate squared errors per (config, position) for a pooled RMSE.
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        c: {p: [] for p in POSITIONS} for c in configs
    }

    common = [c for c in epl.columns if c in wc.columns]

    for k in holdout_rounds:
        holdout = wc[wc["gameweek"] == k].copy()
        wc_prior = wc[wc["gameweek"] < k]
        print(f"\n=== holdout WC round {k}  (n={len(holdout)}) ===")
        print(f"{'config':<14} " + " ".join(f"{p:>7}" for p in POSITIONS))
        for cname, (feats, include_wc) in configs.items():
            if include_wc:
                train_df = pd.concat([epl[common], wc_prior[common]], ignore_index=True)
            else:
                train_df = epl
            line = f"{cname:<14} "
            for pos in POSITIONS:
                hpos = holdout[holdout["position"] == pos]
                if hpos.empty:
                    line += f"{'-':>7} "
                    continue
                booster = _train_mean_head(train_df, pos, feats)
                pred = _predict(booster, hpos, feats)
                yt = hpos["target"].astype(float).to_numpy()
                line += f"{_rmse(yt, pred):>7.3f} "
                pooled[cname][pos].append(np.column_stack([yt, pred]))
            print(line)

    print("\n=== POOLED RMSE across all held-out WC rounds ===")
    print(f"{'config':<14} " + " ".join(f"{p:>7}" for p in POSITIONS) + f" {'ALL':>7}")
    for cname in configs:
        line = f"{cname:<14} "
        all_stack = []
        for pos in POSITIONS:
            chunks = pooled[cname][pos]
            if not chunks:
                line += f"{'-':>7} "
                continue
            m = np.vstack(chunks)
            line += f"{_rmse(m[:, 0], m[:, 1]):>7.3f} "
            all_stack.append(m)
        if all_stack:
            m = np.vstack(all_stack)
            line += f"{_rmse(m[:, 0], m[:, 1]):>7.3f}"
        print(line)


if __name__ == "__main__":
    run()
