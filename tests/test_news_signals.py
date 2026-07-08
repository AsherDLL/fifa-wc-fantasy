"""Tests for the config-driven player-signal extractor."""
from __future__ import annotations

import pandas as pd

from fifa_fantasy.external.news.signals import (
    SignalsConfig, build_name_index, extract_signals, load_signals_config,
)


def _cfg(**over) -> SignalsConfig:
    base = dict(
        proximity_chars=120,
        min_lastname_len=5,
        signals={
            "injury": ("risk", ("injur", "ruled out")),
            "player_of_the_match": ("boost", ("player of the match",)),
        },
    )
    base.update(over)
    return SignalsConfig(**base)


def _players() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": 1, "full_name": "Ousmane Dembele", "known_name": None,
         "last_name": "Dembele", "country_abbr": "FRA", "position": "MID"},
        {"player_id": 2, "full_name": "Orlando Gill", "known_name": None,
         "last_name": "Gill", "country_abbr": "PAR", "position": "GK"},
    ])


def _article(title: str, body: str = "") -> pd.DataFrame:
    return pd.DataFrame([{
        "url": "https://example.test/a", "source_id": "test",
        "source_confidence": 0.5, "title": title, "snippet": "",
        "body_text": body, "published_at_utc": None,
        "collected_at_utc": None,
    }])


def test_signal_fires_when_pattern_near_name():
    art = _article("Dembele suffers hamstring injury in training")
    out = extract_signals(art, _players(), _cfg())
    assert len(out) == 1
    row = out.iloc[0]
    assert row["player_id"] == 1
    assert row["signal"] == "injury"
    assert row["signal_class"] == "risk"
    assert "injury" in row["evidence"].lower()


def test_no_signal_when_pattern_far_from_name():
    body = "Dembele scored twice. " + ("x" * 500) + " An injury update on someone else."
    out = extract_signals(_article("Match report", body), _players(), _cfg())
    assert len(out) == 0


def test_accented_name_matches_ascii_text_and_vice_versa():
    players = pd.DataFrame([{
        "player_id": 3, "full_name": "Kylian Mbappé", "known_name": None,
        "last_name": "Mbappé", "country_abbr": "FRA", "position": "FWD",
    }])
    art = _article("Mbappe named player of the match against Morocco")
    out = extract_signals(art, players, _cfg())
    assert len(out) == 1
    assert out.iloc[0]["signal"] == "player_of_the_match"
    assert out.iloc[0]["signal_class"] == "boost"


def test_short_lastname_requires_full_name():
    # "Gill" (4 chars < min 5) must not fire on the standalone surname,
    # so a town name containing it stays silent.
    out = extract_signals(
        _article("Gillingham injury crisis deepens"), _players(), _cfg())
    assert len(out) == 0
    # The full name still matches.
    out2 = extract_signals(
        _article("Orlando Gill injury scare in training"), _players(), _cfg())
    assert len(out2) == 1
    assert out2.iloc[0]["player_id"] == 2


def test_packaged_config_loads_and_has_potm():
    cfg = load_signals_config()
    assert "player_of_the_match" in cfg.signals
    cls, patterns = cfg.signals["player_of_the_match"]
    assert cls == "boost"
    # Covers the sponsored award name via substring.
    assert any("player of the match" in p for p in patterns)
    assert cfg.proximity_chars > 0


def test_name_index_variants():
    idx = build_name_index(_players(), min_lastname_len=5)
    dembele = next(e for e in idx if e["player_id"] == 1)
    gill = next(e for e in idx if e["player_id"] == 2)
    # Dembele: full name + standalone surname (7 chars >= 5).
    assert len(dembele["patterns"]) == 2
    # Gill: full name only (4-char surname suppressed).
    assert len(gill["patterns"]) == 1
