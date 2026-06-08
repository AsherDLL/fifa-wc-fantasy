"""Tests for the Phase 4.5 pre-lockout polish features."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fifa_fantasy.collector.parse import parse_players, parse_squads
from fifa_fantasy.model.baseline import (
    PREMIUM_PRICE_THRESHOLD,
    heuristic_predict,
)
from fifa_fantasy.optimizer.alternatives import render_alternatives_markdown
from fifa_fantasy.optimizer.compare import diff as diff_rec, render_diff_markdown


# ---------------------------------------------------------------------------
# oneToWatch round-trips through the collector parser
# ---------------------------------------------------------------------------

def test_one_to_watch_round_trips():
    raw_squads = [{
        "id": 1, "name": "Atlantis", "abbr": "ATL", "group": "a",
        "isEliminated": False,
    }]
    raw_players = [
        {"id": 1, "firstName": "Star", "lastName": "Player",
         "knownName": "Star Player", "squadId": 1, "position": "FWD",
         "price": 9.0, "status": "playing", "percentSelected": 10.0,
         "oneToWatch": True, "oneToWatchText": "in red-hot form"},
        {"id": 2, "firstName": "Other", "lastName": "Player",
         "knownName": None, "squadId": 1, "position": "MID",
         "price": 6.0, "status": "playing", "percentSelected": 5.0},
        # oneToWatch fields omitted — should default
    ]
    squads = parse_squads(raw_squads)
    players = parse_players(raw_players, squads)
    assert players[0].one_to_watch is True
    assert players[0].one_to_watch_text == "in red-hot form"
    assert players[1].one_to_watch is False
    assert players[1].one_to_watch_text is None


# ---------------------------------------------------------------------------
# Premium-boost knob
# ---------------------------------------------------------------------------

@pytest.fixture
def features_for_premium() -> pd.DataFrame:
    # Two FWDs: one at $7M (below threshold), one at $11M (above).
    return pd.DataFrame([
        {"position": "FWD", "price_millions": 7.0, "strength_diff": 0.0,
         "is_home": False, "status": "playing", "is_eliminated": False},
        {"position": "FWD", "price_millions": 11.0, "strength_diff": 0.0,
         "is_home": False, "status": "playing", "is_eliminated": False},
    ])


def test_premium_boost_zero_unchanged(features_for_premium):
    a = heuristic_predict(features_for_premium, premium_boost=0.0)
    b = heuristic_predict(features_for_premium)  # default
    pd.testing.assert_series_equal(a["predicted_points"], b["predicted_points"])


def test_premium_boost_only_affects_above_threshold(features_for_premium):
    boost = 0.5
    base = heuristic_predict(features_for_premium, premium_boost=0.0)
    boosted = heuristic_predict(features_for_premium, premium_boost=boost)
    delta = boosted["predicted_points"] - base["predicted_points"]
    # $7M player: below threshold → no boost.
    assert delta.iloc[0] == pytest.approx(0.0)
    # $11M player: 11 - 9 = 2; 0.5 × 2 = 1.0.
    assert delta.iloc[1] == pytest.approx(0.5 * (11.0 - PREMIUM_PRICE_THRESHOLD))


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _write_recommendation(tmp_path: Path, squad: list[int], starters: list[int],
                          captain_id: int, vice_id: int, formation: str,
                          xi_pts: float) -> Path:
    path = tmp_path / "prev.json"
    path.write_text(json.dumps({
        "stage": "GROUP_MD1",
        "horizon_rounds": [1, 2, 3],
        "budget_used": 99.0,
        "budget_total": 100.0,
        "total_horizon_points": 250.0,
        "squad_player_ids": squad,
        "lineup": {
            "round_id": 1, "formation": formation,
            "starter_ids": starters,
            "bench_ids_priority_order": [pid for pid in squad if pid not in starters],
            "captain_id": captain_id, "vice_captain_id": vice_id,
            "expected_points": xi_pts,
        },
    }))
    return path


def test_diff_no_changes(tmp_path):
    squad = list(range(1, 16))
    starters = list(range(1, 12))
    prev = _write_recommendation(tmp_path, squad, starters, 1, 2, "4-4-2", 60.0)
    d = diff_rec(prev, squad, starters, 1, "4-4-2", 60.0)
    assert d.squad_in == []
    assert d.squad_out == []
    assert d.captain_changed is False
    assert d.formation_changed is False
    assert d.md1_expected_delta == pytest.approx(0.0)


def test_diff_squad_change(tmp_path):
    prev_squad = list(range(1, 16))
    new_squad = list(range(2, 17))  # 1 OUT, 16 IN
    starters = list(range(2, 13))
    prev = _write_recommendation(tmp_path, prev_squad, list(range(1, 12)),
                                 1, 2, "4-4-2", 60.0)
    d = diff_rec(prev, new_squad, starters, 2, "4-4-2", 62.0)
    assert d.squad_in == [16]
    assert d.squad_out == [1]
    assert d.captain_changed is True
    assert d.md1_expected_delta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Alternatives report
# ---------------------------------------------------------------------------

@pytest.fixture
def squad_round() -> pd.DataFrame:
    rows = []
    pid = 0
    counts = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    for position, n in counts.items():
        for i in range(n):
            pid += 1
            rows.append({
                "player_id": pid, "full_name": f"{position}{i}",
                "country_abbr": "XXX", "position": position,
                "predicted_points": 1.0 + i + (10.0 if position == "FWD" and i == 0 else 0),
            })
    return pd.DataFrame(rows)


def test_alternatives_lists_top3_captains(squad_round):
    md = render_alternatives_markdown(
        squad_round,
        starter_ids=squad_round["player_id"].head(11).tolist(),
        captain_id=int(squad_round.loc[squad_round["position"] == "FWD"]
                       .nlargest(1, "predicted_points")["player_id"].iloc[0]),
    )
    assert "Top 3 captain candidates" in md
    assert "Gap to #1" in md


def test_alternatives_marks_starters_with_star(squad_round):
    md = render_alternatives_markdown(
        squad_round,
        starter_ids=squad_round["player_id"].head(11).tolist(),
        captain_id=1,
    )
    assert "★" in md
    assert "Bench → starter swap risk" in md
