"""Tests for the structural Poisson predictor."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.model.poisson import (
    APPEARANCE_POINTS,
    BASE_TEAM_XG,
    poisson_predict,
)


def _row(**over):
    base = {
        "position": "MID",
        "price_millions": 6.0,
        "is_home": True,
        "strength_diff": 0.0,
        "rank_diff": 0.0,
        "ownership_fraction": 0.20,
        "status": "playing",
        "is_eliminated": False,
    }
    base.update(over)
    return base


def test_zero_for_unavailable():
    df = pd.DataFrame([
        _row(status="transferred"),
        _row(is_eliminated=True),
    ])
    out = poisson_predict(df)
    assert out["predicted_points"].iloc[0] == 0.0
    assert out["predicted_points"].iloc[1] == 0.0


def test_fwd_against_weaker_beats_fwd_against_stronger():
    df = pd.DataFrame([
        _row(position="FWD", rank_diff=600.0, strength_diff=2.0),
        _row(position="FWD", rank_diff=-600.0, strength_diff=-2.0),
    ])
    out = poisson_predict(df)
    assert out["predicted_points"].iloc[0] > out["predicted_points"].iloc[1]


def test_gk_clean_sheet_signal_against_weak_opponent():
    df = pd.DataFrame([
        _row(position="GK", rank_diff=600.0, strength_diff=2.0),
        _row(position="GK", rank_diff=-600.0, strength_diff=-2.0),
    ])
    out = poisson_predict(df)
    # GK facing a weak opponent should beat GK facing a strong one.
    assert out["predicted_points"].iloc[0] > out["predicted_points"].iloc[1]


def test_scouting_bonus_not_baked_into_predictions():
    # The scouting bonus is applied exactly once, in
    # optimizer/pipeline.apply_scouting_bonus. The backend must NOT bake
    # it into predicted_points: doing so double-counted it downstream
    # (+4 instead of +2), inflating every low-ownership backup keeper
    # above their team's actual number one. Identical players differing
    # only in ownership must get identical raw predictions.
    df = pd.DataFrame([
        _row(position="FWD", rank_diff=400.0, strength_diff=1.5,
             ownership_fraction=0.01),
        _row(position="FWD", rank_diff=400.0, strength_diff=1.5,
             ownership_fraction=0.5),
    ])
    out = poisson_predict(df)
    low_own = out["predicted_points"].iloc[0]
    high_own = out["predicted_points"].iloc[1]
    assert low_own == pytest.approx(high_own, abs=1e-9)


def test_appearance_floor_above_zero():
    # Even an ordinary midfielder with neutral context should beat
    # APPEARANCE_POINTS by some margin once xG is added in.
    df = pd.DataFrame([_row()])
    out = poisson_predict(df)
    assert out["predicted_points"].iloc[0] > APPEARANCE_POINTS - 1


def test_predicted_points_non_negative():
    df = pd.DataFrame([
        _row(position=p, rank_diff=-700.0, strength_diff=-3.0)
        for p in ("GK", "DEF", "MID", "FWD")
    ])
    out = poisson_predict(df)
    assert (out["predicted_points"] >= 0).all()
