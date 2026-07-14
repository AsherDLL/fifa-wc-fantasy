"""Tests for the Phase 4.5 pre-lockout polish features."""

from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.collector.parse import parse_players, parse_squads
from fifa_fantasy.model.baseline import (
    PREMIUM_PRICE_THRESHOLD,
    heuristic_predict,
)


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
