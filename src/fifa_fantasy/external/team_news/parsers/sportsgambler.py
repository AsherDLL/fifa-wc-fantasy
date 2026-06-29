"""Sportsgambler predicted-lineup parser.

Sportsgambler.com publishes structured predicted lineups for major
fixtures. Their pages are HTML with a recognisable layout: two teams,
each with 11 players listed in a defined block.

URL pattern: https://www.sportsgambler.com/lineups/football/<league>/<match-slug>/

This is currently a scaffold; the exact CSS selectors will need
tuning when we have a real WC 2026 match URL to test against. For
now the parser returns None and lets the caller fall back to ESPN.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ...scraping import StealthClient
from ..models import RawLineup, RawPlayerLineup


SOURCE_NAME = "sportsgambler"
SOURCE_CONFIDENCE = 0.50


def fetch_predicted_xi(
    client: StealthClient,
    page_url: str,
    home_team_name: str,
    away_team_name: str,
) -> RawLineup | None:
    """Fetch a Sportsgambler page and extract the predicted XIs.

    Currently a scaffold; returns None until selectors are tuned.
    Document why: Sportsgambler's exact DOM structure varies per
    league and we don't have a stable WC 2026 page yet.
    """
    try:
        response = client.get(page_url)
    except Exception:  # noqa: BLE001
        return None
    if response.status_code != 200:
        return None
    # TODO when we have a real URL: parse the soup and extract lineups.
    # For now, returning None lets the caller cascade to other parsers.
    _ = BeautifulSoup(response.text, "lxml")
    return None
