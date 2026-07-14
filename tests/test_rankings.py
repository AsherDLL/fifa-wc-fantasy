"""Tests for the FIFA ranking loader and the rank-aware heuristic blend."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fifa_fantasy.collector.rankings import load_rankings
from fifa_fantasy.features.squad import squad_strength
from fifa_fantasy.model.baseline import (
    PRICE_SIGNAL_WEIGHT,
    RANK_DIFF_SCALE,
    STRENGTH_DIFF_ALPHA,
    STRENGTH_DIFF_SCALE,
    STRENGTH_SIGNAL_WEIGHT,
    heuristic_predict,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_load_rankings_missing_file(tmp_path):
    out = load_rankings(tmp_path / "nope.csv")
    assert out.empty
    assert set(out.columns) == {"country", "rank_points", "rank_position"}


def test_load_rankings_strips_comments_and_ranks(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(textwrap.dedent("""
        # header comment one
        # header comment two
        country,rank_points
        Atlantis,2000.0
        Borduria,1500.0
        Carpathia,1000.0
    """).strip())
    df = load_rankings(csv)
    assert len(df) == 3
    by_country = df.set_index("country")
    assert by_country.loc["Atlantis", "rank_position"] == 1
    assert by_country.loc["Borduria", "rank_position"] == 2
    assert by_country.loc["Carpathia", "rank_position"] == 3


def test_load_rankings_drops_blank_rows(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("country,rank_points\nAtlantis,1500.0\nBorduria,\n")
    df = load_rankings(csv)
    assert df["country"].tolist() == ["Atlantis"]


# ---------------------------------------------------------------------------
# squad_strength with rankings
# ---------------------------------------------------------------------------

@pytest.fixture
def small_pool():
    squads = pd.DataFrame([
        {"squad_id": 1, "name": "Atlantis", "abbr": "ATL", "group": "a", "is_eliminated": False},
        {"squad_id": 2, "name": "Borduria", "abbr": "BOR", "group": "a", "is_eliminated": False},
    ])
    players = pd.DataFrame([
        {"player_id": p, "full_name": f"P{p}", "position": "MID", "squad_id": 1,
         "country": "Atlantis", "price_millions": 5.0, "ownership_fraction": 0.1,
         "status": "playing", "is_eliminated": False}
        for p in range(1, 12)
    ] + [
        {"player_id": p, "full_name": f"P{p}", "position": "MID", "squad_id": 2,
         "country": "Borduria", "price_millions": 5.0, "ownership_fraction": 0.1,
         "status": "playing", "is_eliminated": False}
        for p in range(12, 23)
    ])
    return squads, players


def test_squad_strength_attaches_rank_columns_when_provided(small_pool):
    squads, players = small_pool
    rankings = pd.DataFrame({
        "country": ["Atlantis", "Borduria"],
        "rank_points": [2000.0, 1000.0],
        "rank_position": [1, 2],
    })
    out = squad_strength(players, squads, rankings=rankings).set_index("name")
    assert out.loc["Atlantis", "squad_rank_points"] == 2000.0
    assert out.loc["Borduria", "squad_rank_points"] == 1000.0


def test_squad_strength_left_join_keeps_unmapped_squads(small_pool):
    squads, players = small_pool
    rankings = pd.DataFrame({
        "country": ["Atlantis"],
        "rank_points": [2000.0],
        "rank_position": [1],
    })
    out = squad_strength(players, squads, rankings=rankings).set_index("name")
    assert pd.isna(out.loc["Borduria", "squad_rank_points"])


def test_squad_strength_no_rankings_keeps_columns_nan(small_pool):
    squads, players = small_pool
    out = squad_strength(players, squads).set_index("name")
    assert "squad_rank_points" in out.columns
    assert out["squad_rank_points"].isna().all()


# ---------------------------------------------------------------------------
# Heuristic blends signals correctly
# ---------------------------------------------------------------------------

def _row(**overrides):
    row = {
        "position": "MID",
        "price_millions": 6.0,
        "strength_diff": 0.0,
        "rank_diff": 0.0,
        "is_home": False,
        "status": "playing",
        "is_eliminated": False,
    }
    row.update(overrides)
    return row


def test_heuristic_uses_blended_signal_when_rank_present():
    # Player with no price gap but a 500-point rank advantage. Expected
    # matchup multiplier = 1 + alpha * tanh(STRENGTH_SIGNAL_WEIGHT * 500/RANK_DIFF_SCALE).
    df = pd.DataFrame([_row(strength_diff=0.0, rank_diff=500.0)])
    actual = heuristic_predict(df)["predicted_points"].iloc[0]
    z_blended = STRENGTH_SIGNAL_WEIGHT * 500.0 / RANK_DIFF_SCALE
    expected = 0.60 * 6.0 * (1 + STRENGTH_DIFF_ALPHA * np.tanh(z_blended))
    assert actual == pytest.approx(expected, rel=1e-4)


def test_heuristic_falls_back_to_price_only_when_rank_missing():
    df = pd.DataFrame([_row(strength_diff=2.0, rank_diff=float("nan"))])
    actual = heuristic_predict(df)["predicted_points"].iloc[0]
    z_price = 2.0 / STRENGTH_DIFF_SCALE  # full weight, no rank blend
    expected = 0.60 * 6.0 * (1 + STRENGTH_DIFF_ALPHA * np.tanh(z_price))
    assert actual == pytest.approx(expected, rel=1e-4)


def test_heuristic_blends_two_signals_consistently():
    df = pd.DataFrame([_row(strength_diff=2.0, rank_diff=500.0)])
    actual = heuristic_predict(df)["predicted_points"].iloc[0]
    z_blended = (PRICE_SIGNAL_WEIGHT * 2.0 / STRENGTH_DIFF_SCALE
                 + STRENGTH_SIGNAL_WEIGHT * 500.0 / RANK_DIFF_SCALE)
    expected = 0.60 * 6.0 * (1 + STRENGTH_DIFF_ALPHA * np.tanh(z_blended))
    assert actual == pytest.approx(expected, rel=1e-4)


def test_rank_diff_can_swing_prediction_above_price_only():
    base = pd.DataFrame([_row(strength_diff=0.0, rank_diff=float("nan"))])
    pos_rank = pd.DataFrame([_row(strength_diff=0.0, rank_diff=800.0)])
    neg_rank = pd.DataFrame([_row(strength_diff=0.0, rank_diff=-800.0)])
    base_pts = heuristic_predict(base)["predicted_points"].iloc[0]
    pos_pts = heuristic_predict(pos_rank)["predicted_points"].iloc[0]
    neg_pts = heuristic_predict(neg_rank)["predicted_points"].iloc[0]
    assert pos_pts > base_pts > neg_pts
