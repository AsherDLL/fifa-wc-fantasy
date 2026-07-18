"""Hermetic tests for the wc2026_dataset module (no network)."""
from pathlib import Path

import pandas as pd

from fifa_fantasy.external import wc2026_dataset as w
from fifa_fantasy.external.team_news.matcher import PlayerNameMatcher


def test_load_missing_dir_returns_empty(tmp_path: Path):
    df = w.load("matches", out_dir=tmp_path / "nope")
    assert df.empty


def test_team_id_map_joins_on_fifa_code(tmp_path: Path):
    teams = pd.DataFrame({"team_id": [1, 2],
                          "team_name": ["Mexico", "Argentina"],
                          "fifa_code": ["MEX", "ARG"]})
    squads = pd.DataFrame({"squad_id": [10, 2], "name": ["Mexico", "Argentina"],
                           "abbr": ["MEX", "ARG"], "group": ["a", "j"],
                           "is_eliminated": [True, False]})
    raw = tmp_path / "raw"
    raw.mkdir()
    squads.to_parquet(raw / "squads_2026-07-01.parquet", index=False)
    m = w.team_id_map(teams=teams, raw_dir=raw)
    assert m.set_index("team_id")["squad_id"].to_dict() == {1: 10, 2: 2}


def test_match_player_ids_first_last_fallback():
    ours = pd.DataFrame({
        "player_id": [1, 2, 3],
        "full_name": ["Theo Hernandez", "Lucas Hernandez", "Nico O'Reilly"],
        "known_name": [None, None, None],
        "country_abbr": ["FRA", "FRA", "ENG"],
        "position": ["DEF", "DEF", "DEF"],
    })
    matcher = PlayerNameMatcher(ours)
    ext = pd.DataFrame({
        "player_name": ["Théo Bernard François Hernandez", "Nico Oreilly"],
        "abbr": ["FRA", "ENG"],
        "position": ["DEF", "DEF"],
    })
    ids = w.match_player_ids(ext, matcher)
    assert list(ids) == [1, 3]


def test_team_xg_form_is_leak_free(tmp_path: Path, monkeypatch):
    """A round-k row must average only xG from rounds strictly before k."""
    spine = pd.DataFrame({
        "match_id": [1, 2],
        "completed": [True, True],
        "round_id": [1, 2],
        "home_squad_id": [10, 10],
        "away_squad_id": [20, 20],
        "home_xg": [2.0, 0.5],
        "away_xg": [1.0, 1.5],
    })
    monkeypatch.setattr(w, "match_spine", lambda raw_dir: spine)
    monkeypatch.setattr(w, "team_id_map", lambda raw_dir: pd.DataFrame(
        {"team_id": [1, 2], "abbr": ["AAA", "BBB"], "squad_id": [10, 20],
         "country": ["Aland", "Bland"]}))
    fixtures = pd.DataFrame({"round_id": [1, 2, 3]})
    monkeypatch.setattr(w, "_latest", lambda raw_dir, prefix: fixtures)
    out = w.team_xg_form(window=3, raw_dir=tmp_path)
    a = out[out["squad_id"] == 10].set_index("round_id")
    assert 1 not in a.index  # no prior matches before round 1
    assert a.loc[2, "team_xg_form_real"] == 2.0  # only round 1
    assert a.loc[3, "team_xg_form_real"] == 1.25  # rounds 1 and 2
    assert a.loc[3, "team_xga_form_real"] == 1.25
