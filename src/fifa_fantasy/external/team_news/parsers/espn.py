"""ESPN predicted-XI article parser.

ESPN's pre-match articles include a "Predicted lineup:" section in a
fairly standard format. We use the StealthClient to fetch the article
and BeautifulSoup to extract the player names.

Article URL pattern (WC 2026 preview articles):
    https://www.espn.com/soccer/story/_/id/<id>/<slug>-tv-channel-...-predicted-lineups

We do not crawl ESPN to discover URLs; the caller passes in an
(article_url, fixture_id, home_abbr, away_abbr) tuple. Article-URL
discovery is a separate concern (could be RSS, search, manual seed
list).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup

from ...scraping import StealthClient
from ..models import RawLineup, RawPlayerLineup


SOURCE_NAME = "espn"
SOURCE_CONFIDENCE = 0.65


# Common patterns ESPN uses to mark a predicted lineup section.
LINEUP_HEADERS = (
    "predicted lineup",
    "predicted xi",
    "starting xi",
    "lineup:",
)


def _extract_lineup_block(text: str, team_name: str) -> str | None:
    """Find a paragraph that mentions team_name and looks like a lineup."""
    paragraphs = re.split(r"\n\s*\n|<p>|</p>", text, flags=re.IGNORECASE)
    for p in paragraphs:
        lower = p.lower()
        if team_name.lower() in lower and any(h in lower for h in LINEUP_HEADERS):
            return p
    return None


def _player_names_from_block(block: str) -> list[str]:
    """Heuristic extractor: split on common delimiters, drop short tokens."""
    # Remove the predictive-XI header.
    for h in LINEUP_HEADERS:
        block = re.sub(rf"(?i){re.escape(h)}", "", block)
    # Split on comma, semicolon, slash, dash.
    tokens = re.split(r"[,;/–—]|\s\-\s", block)
    names = []
    for t in tokens:
        t = t.strip().strip(".:;")
        # Filter junk: too short, contains digits, contains "predicted".
        if not t or len(t) < 4:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if any(w in t.lower() for w in ("predicted", "lineup", "formation", "starting")):
            continue
        # First-cap heuristic: only keep tokens that start with a capital.
        if not t[0].isupper():
            continue
        names.append(t)
    return names[:11]   # cap at XI


def fetch_predicted_xi(
    client: StealthClient,
    article_url: str,
    home_team_name: str,
    away_team_name: str,
) -> RawLineup | None:
    """Fetch an ESPN preview article and extract the predicted XIs.

    Returns None if the article does not contain a recognisable lineup
    block, the network call fails, or the response is non-200.
    """
    try:
        response = client.get(article_url)
    except Exception:  # noqa: BLE001
        return None
    if response.status_code != 200:
        return None
    soup = BeautifulSoup(response.text, "lxml")
    body_text = soup.get_text(separator="\n")

    home_block = _extract_lineup_block(body_text, home_team_name)
    away_block = _extract_lineup_block(body_text, away_team_name)
    if not home_block or not away_block:
        return None

    home_names = _player_names_from_block(home_block)
    away_names = _player_names_from_block(away_block)
    if not home_names or not away_names:
        return None

    home_lineup = [RawPlayerLineup(name=n, status="starting") for n in home_names]
    away_lineup = [RawPlayerLineup(name=n, status="starting") for n in away_names]

    return RawLineup(
        source=SOURCE_NAME,
        scraped_at_utc=datetime.now(timezone.utc),
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        confidence=SOURCE_CONFIDENCE,
        raw_text=body_text[:2000],
    )
