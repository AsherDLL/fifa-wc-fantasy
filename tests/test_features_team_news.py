"""Integration test: team-news join produces expected feature columns."""
from __future__ import annotations

import pandas as pd

from fifa_fantasy.features.build import _attach_team_news


def test_attach_team_news_empty_signal_safe():
    """No news -> NaN columns, downstream models keep current behaviour."""
    grid = pd.DataFrame({
        "player_id": [1, 2, 3],
        "fixture_id": [10, 10, 11],
    })
    out = _attach_team_news(grid, None)
    assert "predicted_starting_xi" in out.columns
    assert "xi_confidence" in out.columns
    assert out["predicted_starting_xi"].isna().all()


def test_attach_team_news_starting_and_bench():
    grid = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "fixture_id": [10, 10, 10, 10],
    })
    news = pd.DataFrame([
        {"fixture_id": 10, "player_id": 1, "status": "starting",
         "source_confidence": 0.8, "scraped_at_utc": "2026-06-30T10:00:00Z"},
        {"fixture_id": 10, "player_id": 2, "status": "bench",
         "source_confidence": 0.8, "scraped_at_utc": "2026-06-30T10:00:00Z"},
    ])
    out = _attach_team_news(grid, news)
    assert out.loc[out["player_id"] == 1, "predicted_starting_xi"].iloc[0] == True
    assert out.loc[out["player_id"] == 2, "predicted_starting_xi"].iloc[0] == False
    # Unmatched players (3, 4) are NaN
    assert pd.isna(out.loc[out["player_id"] == 3, "predicted_starting_xi"].iloc[0])
    assert pd.isna(out.loc[out["player_id"] == 4, "predicted_starting_xi"].iloc[0])


def test_attach_team_news_picks_latest_per_player():
    grid = pd.DataFrame({"player_id": [1], "fixture_id": [10]})
    news = pd.DataFrame([
        {"fixture_id": 10, "player_id": 1, "status": "bench",
         "source_confidence": 0.5, "scraped_at_utc": "2026-06-30T08:00:00Z"},
        {"fixture_id": 10, "player_id": 1, "status": "starting",
         "source_confidence": 0.8, "scraped_at_utc": "2026-06-30T12:00:00Z"},
    ])
    out = _attach_team_news(grid, news)
    # Newer record wins.
    assert out["predicted_starting_xi"].iloc[0] == True
    assert out["xi_confidence"].iloc[0] == 0.8
