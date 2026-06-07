"""Tests for the Phase 2 feature builders."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from fifa_fantasy.features.build import (
    build_player_round_features,
    flatten_fixtures,
)
from fifa_fantasy.features.squad import squad_strength


# ---------------------------------------------------------------------------
# Toy data — three squads, a handful of players each, two fixtures over MD1+MD2.
# ---------------------------------------------------------------------------


@pytest.fixture
def squads() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"squad_id": 1, "name": "Atlantis", "abbr": "ATL", "group": "a", "is_eliminated": False},
            {"squad_id": 2, "name": "Borduria", "abbr": "BOR", "group": "a", "is_eliminated": False},
            {"squad_id": 3, "name": "Carpathia", "abbr": "CAR", "group": "a", "is_eliminated": False},
        ]
    )


@pytest.fixture
def players() -> pd.DataFrame:
    # Atlantis: 12 players, expensive top-3. Borduria: 11 cheap. Carpathia: 3 only.
    rows = []
    pid = 0
    for sid, prices in {
        1: [10, 9, 8, 7, 6, 5, 5, 5, 5, 5, 5, 4],
        2: [4] * 11,
        3: [6, 5, 4],
    }.items():
        for price in prices:
            pid += 1
            rows.append(
                {
                    "player_id": pid,
                    "full_name": f"P{pid}",
                    "position": "MID",
                    "squad_id": sid,
                    "country": {1: "Atlantis", 2: "Borduria", 3: "Carpathia"}[sid],
                    "price_millions": price,
                    "ownership_fraction": 0.05,
                    "status": "playing",
                    "is_eliminated": False,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def fixtures() -> pd.DataFrame:
    # Two fixtures: ATL vs BOR in MD1, CAR vs ATL in MD2 (5 days later).
    return pd.DataFrame(
        [
            {
                "fixture_id": 1, "round_id": 1, "stage": "GROUP_MD1",
                "home_squad_id": 1, "home_squad_name": "Atlantis", "home_squad_abbr": "ATL",
                "away_squad_id": 2, "away_squad_name": "Borduria", "away_squad_abbr": "BOR",
                "kickoff": datetime(2026, 6, 11, 18, 0, tzinfo=timezone.utc),
                "venue_name": "Banorte", "venue_city": "Mexico City",
                "status": "scheduled",
            },
            {
                "fixture_id": 2, "round_id": 2, "stage": "GROUP_MD2",
                "home_squad_id": 3, "home_squad_name": "Carpathia", "home_squad_abbr": "CAR",
                "away_squad_id": 1, "away_squad_name": "Atlantis", "away_squad_abbr": "ATL",
                "kickoff": datetime(2026, 6, 16, 18, 0, tzinfo=timezone.utc),
                "venue_name": "Akron", "venue_city": "Guadalajara",
                "status": "scheduled",
            },
        ]
    )


# ---------------------------------------------------------------------------
# squad_strength
# ---------------------------------------------------------------------------

def test_squad_strength_columns(players, squads):
    out = squad_strength(players, squads)
    assert {"squad_total_price", "squad_avg_price", "squad_top_n_avg_price",
            "squad_top_n_rank", "squad_size"} <= set(out.columns)


def test_squad_strength_atlantis_is_strongest(players, squads):
    out = squad_strength(players, squads).set_index("squad_id")
    # Atlantis top 11: 10+9+8+7+6+5+5+5+5+5+5 = 70 → avg 70/11 ≈ 6.36
    assert out.loc[1, "squad_top_n_avg_price"] == pytest.approx(70 / 11)
    # Borduria flat 4 → 4
    assert out.loc[2, "squad_top_n_avg_price"] == 4.0
    # Carpathia has only 3 players → top-11 avg uses all 3 = 5
    assert out.loc[3, "squad_top_n_avg_price"] == 5.0
    # Rank 1 = strongest
    assert out.loc[1, "squad_top_n_rank"] == 1


def test_squad_strength_handles_small_squad(players, squads):
    out = squad_strength(players, squads).set_index("squad_id")
    assert out.loc[3, "squad_size"] == 3


def test_squad_strength_top_n_invalid_raises(players, squads):
    with pytest.raises(ValueError):
        squad_strength(players, squads, top_n=0)


# ---------------------------------------------------------------------------
# flatten_fixtures
# ---------------------------------------------------------------------------

def test_flatten_fixtures_doubles_rows(fixtures):
    flat = flatten_fixtures(fixtures)
    assert len(flat) == 2 * len(fixtures)


def test_flatten_fixtures_is_home_split(fixtures):
    flat = flatten_fixtures(fixtures)
    assert flat["is_home"].sum() == len(fixtures)
    assert (~flat["is_home"]).sum() == len(fixtures)


def test_flatten_fixtures_opponent_correctness(fixtures):
    flat = flatten_fixtures(fixtures)
    atl_md1 = flat.query("squad_id == 1 and round_id == 1").iloc[0]
    assert atl_md1["opponent_squad_id"] == 2
    assert atl_md1["opponent_abbr"] == "BOR"
    assert atl_md1["is_home"] is True or atl_md1["is_home"]


# ---------------------------------------------------------------------------
# build_player_round_features
# ---------------------------------------------------------------------------

def test_build_features_row_count(players, squads, fixtures):
    strength = squad_strength(players, squads)
    feats = build_player_round_features(players, fixtures, strength)
    # Atlantis (12 players) plays MD1+MD2 → 24 rows.
    # Borduria (11 players) plays MD1 only → 11 rows.
    # Carpathia (3 players) plays MD2 only → 3 rows.
    assert len(feats) == 24 + 11 + 3


def test_build_features_strength_diff_sign(players, squads, fixtures):
    strength = squad_strength(players, squads)
    feats = build_player_round_features(players, fixtures, strength)
    # Atlantis players in MD1 face Borduria — positive strength_diff.
    atl_md1 = feats.query("squad_id == 1 and round_id == 1")
    assert (atl_md1["strength_diff"] > 0).all()
    # Borduria players in MD1 face Atlantis — negative.
    bor_md1 = feats.query("squad_id == 2 and round_id == 1")
    assert (bor_md1["strength_diff"] < 0).all()


def test_build_features_is_home(players, squads, fixtures):
    strength = squad_strength(players, squads)
    feats = build_player_round_features(players, fixtures, strength)
    assert feats.query("squad_id == 1 and round_id == 1")["is_home"].all()
    assert (~feats.query("squad_id == 2 and round_id == 1")["is_home"]).all()
    assert feats.query("squad_id == 3 and round_id == 2")["is_home"].all()


def test_build_features_rest_days(players, squads, fixtures):
    strength = squad_strength(players, squads)
    feats = build_player_round_features(players, fixtures, strength)
    # Atlantis plays MD1 (Jun 11) and MD2 (Jun 16) — 5 days apart.
    atl = feats.query("squad_id == 1").sort_values("kickoff")
    md1 = atl[atl["round_id"] == 1].iloc[0]
    md2 = atl[atl["round_id"] == 2].iloc[0]
    assert pd.isna(md1["days_since_prev_match"])
    assert md2["days_since_prev_match"] == pytest.approx(5.0)
    assert md1["days_to_next_match"] == pytest.approx(5.0)
    assert pd.isna(md2["days_to_next_match"])


def test_build_features_opponent_columns_present(players, squads, fixtures):
    strength = squad_strength(players, squads)
    feats = build_player_round_features(players, fixtures, strength)
    expected = {
        "opp_squad_total_price",
        "opp_squad_avg_price",
        "opp_squad_top_n_avg_price",
        "opp_squad_top_n_rank",
    }
    assert expected <= set(feats.columns)
