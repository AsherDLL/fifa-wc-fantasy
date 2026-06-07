"""HTTP layer for the FIFA Fantasy public data endpoints."""

from __future__ import annotations

from typing import Any

import httpx

BASE_URL = "https://play.fifa.com/json/fantasy"
USER_AGENT = "fifa-fantasy-collector/0.0.1 (https://github.com/AsherDLL/fifa-wc-fantasy)"
DEFAULT_TIMEOUT = 30.0

PLAYERS_PATH = "/players.json"
SQUADS_PATH = "/squads.json"
ROUNDS_PATH = "/rounds.json"
CHECKSUMS_PATH = "/checksums.json"


def make_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.Client:
    return httpx.Client(
        base_url=BASE_URL,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )


def get_json(client: httpx.Client, path: str) -> Any:
    response = client.get(path)
    response.raise_for_status()
    return response.json()
