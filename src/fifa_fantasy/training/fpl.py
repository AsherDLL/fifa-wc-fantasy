"""Scrape one EPL FPL season into a player-gameweek Parquet.

The Fantasy Premier League public API at
`https://fantasy.premierleague.com/api/` exposes two endpoints:

- `bootstrap-static/` : player roster, team roster, gameweek calendar
- `element-summary/<id>/` : per-player history (one row per gameweek)

Same scoring model as the FIFA Fantasy WC game (appearance, goals,
assists, clean sheets, saves, cards, bonus), same auction-priced
market, similar fixture-difficulty mechanic. Good donor data for a
LightGBM trained to map per-(player, fixture) features to fantasy
points.

Output: `data/training/fpl_player_gameweek_<season>.parquet`, one row
per (player, gameweek).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "fifa-fantasy-trainer/0.0.1 (https://github.com/AsherDLL/fifa-wc-fantasy)"
DEFAULT_OUT = Path("data/training")
DEFAULT_WORKERS = 8

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _make_client() -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        timeout=30.0,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def fetch_bootstrap(client: httpx.Client) -> dict[str, Any]:
    r = client.get("/bootstrap-static/")
    r.raise_for_status()
    return r.json()


def fetch_element_history(client: httpx.Client, element_id: int) -> list[dict[str, Any]]:
    r = client.get(f"/element-summary/{element_id}/")
    r.raise_for_status()
    return r.json().get("history", [])


def build_player_gameweek_table(
    bootstrap: dict[str, Any],
    workers: int = DEFAULT_WORKERS,
) -> pd.DataFrame:
    elements = bootstrap["elements"]
    teams = {t["id"]: t for t in bootstrap["teams"]}
    elem_meta = {
        e["id"]: {
            "player_id": e["id"],
            "web_name": e.get("web_name"),
            "first_name": e["first_name"],
            "second_name": e["second_name"],
            "team_id": e["team"],
            "team_name": teams[e["team"]]["name"],
            "team_strength": teams[e["team"]]["strength"],
            "position": POSITION_BY_ELEMENT_TYPE[e["element_type"]],
        }
        for e in elements
    }

    histories: dict[int, list[dict[str, Any]]] = {}
    with _make_client() as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_element_history, client, eid): eid
                for eid in elem_meta
            }
            for fut in as_completed(futures):
                eid = futures[fut]
                histories[eid] = fut.result()

    rows: list[dict[str, Any]] = []
    for eid, history in histories.items():
        meta = elem_meta[eid]
        for h in history:
            opp = teams.get(h["opponent_team"], {})
            rows.append({
                **meta,
                "gameweek": h["round"],
                "fixture_id": h["fixture"],
                "is_home": bool(h["was_home"]),
                "opponent_team_id": h["opponent_team"],
                "opponent_team_name": opp.get("name"),
                "opponent_strength": opp.get("strength"),
                "minutes": h["minutes"],
                "goals_scored": h["goals_scored"],
                "assists": h["assists"],
                "clean_sheets": h["clean_sheets"],
                "goals_conceded": h["goals_conceded"],
                "own_goals": h["own_goals"],
                "penalties_saved": h["penalties_saved"],
                "penalties_missed": h["penalties_missed"],
                "yellow_cards": h["yellow_cards"],
                "red_cards": h["red_cards"],
                "saves": h["saves"],
                "bonus": h["bonus"],
                "bps": h["bps"],
                "ict_index": float(h["ict_index"]),
                "expected_goals": float(h["expected_goals"]),
                "expected_assists": float(h["expected_assists"]),
                "expected_goals_conceded": float(h["expected_goals_conceded"]),
                "tackles": h.get("tackles", 0),
                "starts": h["starts"],
                # FPL's `value` is price * 10 (e.g. 75 = $7.5M).
                "price_millions": h["value"] / 10.0,
                "ownership_fraction": h["selected"] / 100_000_000.0 if False else None,
                "total_points": h["total_points"],
                "kickoff_time": h["kickoff_time"],
            })
    return pd.DataFrame(rows)


def save_table(df: pd.DataFrame, season: str = "current",
               out_dir: Path = DEFAULT_OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fpl_player_gameweek_{season}.parquet"
    df.to_parquet(path, index=False)
    return path
