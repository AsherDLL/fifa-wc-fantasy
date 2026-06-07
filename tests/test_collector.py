"""Tests for the Phase 1 collector.

Two layers:

- Pure parsing functions exercised against small recorded JSON fixtures.
- A single end-to-end test that mocks the HTTP layer with `respx`, runs the
  same pipeline the CLI uses, and verifies on-disk Parquet output.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from fifa_fantasy.collector.api import (
    PLAYERS_PATH,
    ROUNDS_PATH,
    SQUADS_PATH,
    BASE_URL,
    get_json,
    make_client,
)
from fifa_fantasy.collector.parse import (
    parse_fixtures,
    parse_players,
    parse_squads,
)
from fifa_fantasy.collector.persist import save_parquet, save_raw_json
from fifa_fantasy.collector.schemas import Stage
from fifa_fantasy.scoring import Position

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def raw_squads() -> list[dict]:
    return json.loads((FIXTURES_DIR / "squads_sample.json").read_text())


@pytest.fixture
def raw_players() -> list[dict]:
    return json.loads((FIXTURES_DIR / "players_sample.json").read_text())


@pytest.fixture
def raw_rounds() -> list[dict]:
    return json.loads((FIXTURES_DIR / "rounds_sample.json").read_text())


# ---------------------------------------------------------------------------
# parse_squads
# ---------------------------------------------------------------------------

def test_parse_squads_preserves_count(raw_squads):
    squads = parse_squads(raw_squads)
    assert len(squads) == len(raw_squads)


def test_parse_squads_maps_fields(raw_squads):
    squads = {s.squad_id: s for s in parse_squads(raw_squads)}
    mexico = squads[28]
    assert mexico.name == "Mexico"
    assert mexico.abbr == "MEX"
    assert mexico.is_eliminated is False


# ---------------------------------------------------------------------------
# parse_players
# ---------------------------------------------------------------------------

def test_parse_players_joins_country(raw_squads, raw_players):
    squads = parse_squads(raw_squads)
    players = parse_players(raw_players, squads)
    countries = {p.country for p in players}
    assert countries.issubset({"Algeria", "Mexico", "South Africa", "Korea Republic"})


def test_parse_players_normalizes_ownership(raw_squads, raw_players):
    # API reports percentSelected as a percent (e.g. 1.2 for 1.2%);
    # we expose a [0, 1] fraction.
    squads = parse_squads(raw_squads)
    players = parse_players(raw_players, squads)
    for raw, normalized in zip(raw_players, players):
        assert normalized.ownership_fraction == pytest.approx(raw["percentSelected"] / 100.0)


def test_parse_players_position_uses_scoring_enum(raw_squads, raw_players):
    squads = parse_squads(raw_squads)
    players = parse_players(raw_players, squads)
    for p in players:
        assert isinstance(p.position, Position)


def test_parse_players_full_name_prefers_known_name(raw_squads):
    payload = [
        {"id": 1, "firstName": "Cristiano", "lastName": "Ronaldo",
         "knownName": "Cristiano Ronaldo", "squadId": 28, "position": "FWD",
         "price": 10.0, "status": "playing", "percentSelected": 50.0},
        {"id": 2, "firstName": "Lionel", "lastName": "Messi", "knownName": None,
         "squadId": 28, "position": "FWD", "price": 10.0,
         "status": "playing", "percentSelected": 50.0},
    ]
    squads = parse_squads([
        {"id": 28, "name": "Mexico", "abbr": "MEX", "group": "A",
         "isEliminated": False}
    ])
    players = parse_players(payload, squads)
    assert players[0].full_name == "Cristiano Ronaldo"
    assert players[1].full_name == "Lionel Messi"


# ---------------------------------------------------------------------------
# parse_fixtures
# ---------------------------------------------------------------------------

def test_parse_fixtures_flattens_rounds(raw_rounds):
    fixtures = parse_fixtures(raw_rounds)
    expected = sum(len(r["tournaments"]) for r in raw_rounds)
    assert len(fixtures) == expected


def test_parse_fixtures_stage_mapping(raw_rounds):
    fixtures = parse_fixtures(raw_rounds)
    # The fixture sample includes round 1 and round 2 only.
    stages = {f.stage for f in fixtures if f.round_id == 1}
    assert stages == {Stage.GROUP_MD1}
    stages = {f.stage for f in fixtures if f.round_id == 2}
    assert stages == {Stage.GROUP_MD2}


def test_parse_fixtures_preserves_kickoff_timezone(raw_rounds):
    fixtures = parse_fixtures(raw_rounds)
    # API delivers tz-aware ISO strings; we keep them tz-aware.
    assert all(f.kickoff.tzinfo is not None for f in fixtures)


# ---------------------------------------------------------------------------
# persist
# ---------------------------------------------------------------------------

def test_save_parquet_roundtrip(tmp_path, raw_squads):
    squads = parse_squads(raw_squads)
    path = save_parquet(squads, "squads", data_dir=tmp_path)
    assert path.exists()
    df = pd.read_parquet(path)
    assert len(df) == len(squads)
    assert set(df.columns) == {"squad_id", "name", "abbr", "group", "is_eliminated"}


def test_save_raw_json_writes_under_raw_subdir(tmp_path, raw_squads):
    path = save_raw_json(raw_squads, "squads", data_dir=tmp_path)
    assert path.parent == tmp_path / "raw"
    assert path.exists()
    assert json.loads(path.read_text()) == raw_squads


# ---------------------------------------------------------------------------
# End-to-end (mocked HTTP)
# ---------------------------------------------------------------------------

def test_end_to_end_with_mocked_api(tmp_path, raw_squads, raw_players, raw_rounds):
    with respx.mock(base_url=BASE_URL, assert_all_called=True) as mock:
        mock.get(SQUADS_PATH).mock(return_value=httpx.Response(200, json=raw_squads))
        mock.get(PLAYERS_PATH).mock(return_value=httpx.Response(200, json=raw_players))
        mock.get(ROUNDS_PATH).mock(return_value=httpx.Response(200, json=raw_rounds))

        with make_client() as client:
            squads = parse_squads(get_json(client, SQUADS_PATH))
            players = parse_players(get_json(client, PLAYERS_PATH), squads)
            fixtures = parse_fixtures(get_json(client, ROUNDS_PATH))

    squads_path = save_parquet(squads, "squads", data_dir=tmp_path)
    players_path = save_parquet(players, "players", data_dir=tmp_path)
    fixtures_path = save_parquet(fixtures, "fixtures", data_dir=tmp_path)

    assert pd.read_parquet(squads_path).shape[0] == len(raw_squads)
    assert pd.read_parquet(players_path).shape[0] == len(raw_players)
    assert pd.read_parquet(fixtures_path).shape[0] == sum(
        len(r["tournaments"]) for r in raw_rounds
    )
