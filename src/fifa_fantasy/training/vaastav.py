"""Pull historical EPL FPL seasons from the vaastav community mirror.

The mirror at github.com/vaastav/Fantasy-Premier-League republishes
each gameweek's player records as CSV under
`data/<season>/gws/merged_gw.csv`. Column names match the live API
fields closely enough that we can normalise to the same row schema as
`training.fpl.build_player_gameweek_table` and concat the two.

Output: one parquet per season, identical schema to fpl.py output.
"""

from __future__ import annotations

import io
from pathlib import Path

import httpx
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_OUT = Path("data/training")

POSITION_MAP = {"GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD",
                "GKP": "GK"}


def fetch_merged_gw(season: str) -> pd.DataFrame:
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"
    r = httpx.get(url, timeout=60.0, follow_redirects=True)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def normalise(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map vaastav columns to the schema build_player_gameweek_table emits.

    Missing columns are filled with neutral values; some fields the
    vaastav dump does not carry (team_strength) get defaults.
    """
    df = df.copy()
    df["position"] = df["position"].map(POSITION_MAP)
    df = df[df["position"].isin(POSITION_MAP.values())]

    out = pd.DataFrame({
        "player_id": df["element"].astype(int),
        "web_name": df["name"],
        "first_name": df["name"],
        "second_name": "",
        "team_id": df["team"].astype("category").cat.codes.astype(int) + 1,
        "team_name": df["team"],
        "team_strength": 3,  # we do not have per-team strength in the mirror
        "position": df["position"],
        "gameweek": df["GW"].astype(int),
        "fixture_id": df["fixture"].astype(int),
        "is_home": df["was_home"].astype(bool),
        "opponent_team_id": df["opponent_team"].astype(int),
        "opponent_team_name": "",  # not given as a string in the dump
        "opponent_strength": 3,
        "minutes": df["minutes"].astype(int),
        "goals_scored": df["goals_scored"].astype(int),
        "assists": df["assists"].astype(int),
        "clean_sheets": df["clean_sheets"].astype(int),
        "goals_conceded": df["goals_conceded"].astype(int),
        "own_goals": df["own_goals"].astype(int),
        "penalties_saved": df["penalties_saved"].astype(int),
        "penalties_missed": df["penalties_missed"].astype(int),
        "yellow_cards": df["yellow_cards"].astype(int),
        "red_cards": df["red_cards"].astype(int),
        "saves": df["saves"].astype(int),
        "bonus": df["bonus"].astype(int),
        "bps": df["bps"].astype(int),
        "ict_index": df["ict_index"].astype(float),
        "expected_goals": df["expected_goals"].astype(float),
        "expected_assists": df["expected_assists"].astype(float),
        "expected_goals_conceded": df["expected_goals_conceded"].astype(float),
        "tackles": 0,
        "starts": df["starts"].astype(int),
        "price_millions": df["value"] / 10.0,
        "ownership_fraction": None,
        "total_points": df["total_points"].astype(int),
        "kickoff_time": df["kickoff_time"],
        "season": season,
    })
    return out


def save(df: pd.DataFrame, season: str,
         out_dir: Path = DEFAULT_OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fpl_player_gameweek_{season}.parquet"
    df.to_parquet(path, index=False)
    return path
