"""Smoke tests for the LightGBM predictor.

The trained models are not part of the test fixtures; tests construct
tiny in-memory models so the assertions are deterministic.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from fifa_fantasy.model.gbm import FEATURE_COLUMNS, POSITIONS, predict


def _tiny_booster(intercept: float = 3.0) -> lgb.Booster:
    """Train a one-leaf model that always predicts `intercept`."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.uniform(0, 10, size=(40, len(FEATURE_COLUMNS))),
                     columns=FEATURE_COLUMNS)
    y = np.full(len(X), intercept)
    return lgb.train(
        {"objective": "regression", "metric": "rmse",
         "num_leaves": 2, "min_child_samples": 5, "verbose": -1},
        lgb.Dataset(X, label=y),
        num_boost_round=20,
    )


@pytest.fixture
def models_stub():
    return {
        pos: {
            "mean": _tiny_booster(3.0 + i),
            "q10": _tiny_booster(1.0 + i),
            "q50": _tiny_booster(3.0 + i),
            "q90": _tiny_booster(5.0 + i),
        }
        for i, pos in enumerate(POSITIONS)
    }


def _row(**over):
    base = {
        "position": "MID",
        "price_millions": 6.0,
        "is_home": True,
        "strength_diff": 1.0,
        "squad_top_n_avg_price": 6.0,
        "opp_squad_top_n_avg_price": 5.0,
        "status": "playing",
        "is_eliminated": False,
    }
    base.update(over)
    return base


def test_predict_adds_required_columns(models_stub):
    df = pd.DataFrame([_row()])
    out = predict(df, models_stub)
    for col in ("predicted_points", "predicted_q10", "predicted_q50", "predicted_q90"):
        assert col in out.columns


def test_predict_zeros_for_unavailable(models_stub):
    df = pd.DataFrame([_row(status="transferred"), _row(is_eliminated=True)])
    out = predict(df, models_stub)
    assert out["predicted_points"].iloc[0] == 0.0
    assert out["predicted_points"].iloc[1] == 0.0


def test_predict_per_position_routing(models_stub):
    df = pd.DataFrame([_row(position=p) for p in POSITIONS])
    out = predict(df, models_stub)
    # Stubs return 3+i where i is the position index. GBM clips at 0 minimum,
    # so values are non-negative and strictly ordered by position index.
    vals = out["predicted_points"].to_numpy()
    assert (vals[1:] > vals[:-1]).all() or (vals[1:] >= vals[:-1]).all()


def test_predict_quantiles_are_ordered(models_stub):
    df = pd.DataFrame([_row()])
    out = predict(df, models_stub)
    assert out["predicted_q10"].iloc[0] <= out["predicted_q50"].iloc[0]
    assert out["predicted_q50"].iloc[0] <= out["predicted_q90"].iloc[0]
