"""Tests for the Phase 3a heuristic baseline predictor."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fifa_fantasy.model.baseline import (
    HOME_ADVANTAGE_BETA,
    POINTS_PER_PRICE_UNIT,
    STRENGTH_DIFF_ALPHA,
    STRENGTH_DIFF_SCALE,
    heuristic_predict,
)
from fifa_fantasy.scoring import Position


def _row(**overrides) -> dict:
    """One feature row with sensible defaults; tests override specific fields."""
    base = {
        "position": "MID",
        "price_millions": 6.0,
        "strength_diff": 0.0,
        "is_home": False,
        "status": "playing",
        "is_eliminated": False,
    }
    base.update(overrides)
    return base


def _predict(rows: list[dict]) -> pd.DataFrame:
    return heuristic_predict(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Base formula (no matchup adjustment, no home advantage)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "position, price",
    [("GK", 5.0), ("DEF", 5.0), ("MID", 6.0), ("FWD", 8.0)],
)
def test_base_formula_position_coefficient(position, price):
    df = _predict([_row(position=position, price_millions=price)])
    expected = POINTS_PER_PRICE_UNIT[Position(position)] * price
    assert df["predicted_points"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Matchup adjustment via strength_diff
# ---------------------------------------------------------------------------

def test_positive_strength_diff_boosts_prediction():
    base = _predict([_row(strength_diff=0.0)])["predicted_points"].iloc[0]
    boosted = _predict([_row(strength_diff=3.0)])["predicted_points"].iloc[0]
    assert boosted > base


def test_negative_strength_diff_reduces_prediction():
    base = _predict([_row(strength_diff=0.0)])["predicted_points"].iloc[0]
    weakened = _predict([_row(strength_diff=-3.0)])["predicted_points"].iloc[0]
    assert weakened < base


def test_matchup_saturates_at_alpha():
    # tanh(big) → 1 → adjustment ≈ (1 + alpha).
    big = _predict([_row(strength_diff=20.0)])["predicted_points"].iloc[0]
    coef = POINTS_PER_PRICE_UNIT[Position.MID]
    expected_upper = coef * 6.0 * (1 + STRENGTH_DIFF_ALPHA)
    assert big == pytest.approx(expected_upper, rel=1e-3)


def test_matchup_formula_explicit():
    # Pin the exact value at a moderate strength diff for one row.
    diff = 1.5
    df = _predict([_row(strength_diff=diff)])
    coef = POINTS_PER_PRICE_UNIT[Position.MID]
    expected = (
        coef * 6.0 * (1 + STRENGTH_DIFF_ALPHA * math.tanh(diff / STRENGTH_DIFF_SCALE))
    )
    assert df["predicted_points"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Home advantage
# ---------------------------------------------------------------------------

def test_home_advantage_is_multiplicative():
    away = _predict([_row(is_home=False)])["predicted_points"].iloc[0]
    home = _predict([_row(is_home=True)])["predicted_points"].iloc[0]
    assert home == pytest.approx(away * (1 + HOME_ADVANTAGE_BETA))


# ---------------------------------------------------------------------------
# Availability gating
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["transferred", "injured", "suspended"])
def test_non_playing_status_predicts_zero(status):
    df = _predict([_row(status=status)])
    assert df["predicted_points"].iloc[0] == 0.0


def test_eliminated_squad_predicts_zero():
    df = _predict([_row(is_eliminated=True)])
    assert df["predicted_points"].iloc[0] == 0.0


def test_unavailable_overrides_great_fixture():
    df = _predict(
        [_row(status="transferred", strength_diff=5.0, is_home=True, price_millions=10.0)]
    )
    assert df["predicted_points"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# DataFrame integrity
# ---------------------------------------------------------------------------

def test_input_not_mutated():
    df_in = pd.DataFrame([_row()])
    df_in_copy = df_in.copy()
    heuristic_predict(df_in)
    pd.testing.assert_frame_equal(df_in, df_in_copy)


def test_output_preserves_input_columns():
    df_in = pd.DataFrame([_row()])
    df_out = heuristic_predict(df_in)
    assert set(df_in.columns).issubset(set(df_out.columns))
    assert "predicted_points" in df_out.columns
