"""Tests for the player-name matcher."""
from __future__ import annotations

import pandas as pd
import pytest

from fifa_fantasy.external.team_news.matcher import PlayerNameMatcher


@pytest.fixture
def sample_players():
    return pd.DataFrame([
        {"player_id": 38, "full_name": "Lionel Messi",
         "known_name": None, "country_abbr": "ARG", "position": "FWD"},
        {"player_id": 500, "full_name": "Kylian Mbappé",
         "known_name": None, "country_abbr": "FRA", "position": "FWD"},
        {"player_id": 173, "full_name": "Vinícius Júnior",
         "known_name": "Vini Jr", "country_abbr": "BRA", "position": "MID"},
        {"player_id": 491, "full_name": "Jude Bellingham",
         "known_name": None, "country_abbr": "ENG", "position": "MID"},
    ])


def test_matcher_exact_full_name(sample_players):
    m = PlayerNameMatcher(sample_players)
    r = m.match("Lionel Messi", country_abbr="ARG", position="FWD")
    assert r.player_id == 38
    assert r.confidence == 1.0


def test_matcher_accent_insensitive(sample_players):
    m = PlayerNameMatcher(sample_players)
    r = m.match("Kylian Mbappe", country_abbr="FRA", position="FWD")
    assert r.player_id == 500
    assert r.confidence >= 0.85


def test_matcher_last_name_with_country(sample_players):
    m = PlayerNameMatcher(sample_players)
    r = m.match("Bellingham", country_abbr="ENG", position="MID")
    assert r.player_id == 491
    assert r.confidence >= 0.85


def test_matcher_no_match(sample_players):
    m = PlayerNameMatcher(sample_players)
    r = m.match("Someone Unknown", country_abbr="ARG", position="FWD")
    assert r.player_id is None
    assert r.confidence == 0.0


def test_matcher_known_name(sample_players):
    m = PlayerNameMatcher(sample_players)
    r = m.match("Vini Jr", country_abbr="BRA", position="MID")
    assert r.player_id == 173
    assert r.confidence >= 0.9
