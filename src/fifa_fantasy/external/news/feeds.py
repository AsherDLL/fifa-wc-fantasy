"""Feed registry loader for the news collector.

The registry itself lives in `config/feeds.json` next to this module:
every feed URL and every gating keyword is configuration, not code, so
the collector can be reused in another project by swapping one JSON
file (or pointing the NEWS_FEEDS_CONFIG env var / --feeds-config flag
at your own). This module only defines the FeedConfig shape and the
loader.

Feed-format notes (why JSON feeds exist alongside RSS), and the log of
feeds we tried and dropped (BBC world-football 404, ESPN RSS behind AWS
WAF, Goal.com 404, reddit.com blocking datacenter IPs), live in the
JSON file's own comment fields so the operational record travels with
the config.

Twitter / X is deliberately NOT included. As of 2026, Nitter's public
instances are dead, twscrape requires authenticated accounts, and the
RSS-Bridge path requires hosting your own instance. The cost is not
worth the marginal lineup signal. Documented in
`docs/whitepaper/sections/11e_team_news_ingestion.md`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "feeds.json"


@dataclass(frozen=True)
class FeedConfig:
    """One feed (RSS by default; JSON for sources that don't expose RSS)."""
    name: str
    url: str
    source_id: str             # short id used in storage
    base_confidence: float = 0.5
    # Keywords (case-insensitive) that gate which articles we fetch.
    keywords: tuple[str, ...] = field(default_factory=tuple)
    # Feed format. "rss" parses RSS 2.0 / Atom XML. "json_espn" parses the
    # ESPN site.api.espn.com news payload (articles -> [{headline, links, ...}]).
    # "json_arctic_shift" parses arctic-shift's Reddit mirror (data -> [{title,
    # selftext, permalink, created_utc, ...}]).
    format: str = "rss"


def load_feeds(config_path: Path | str | None = None) -> tuple[FeedConfig, ...]:
    """Load the feed registry from JSON.

    Resolution order: explicit argument, NEWS_FEEDS_CONFIG env var, the
    packaged default. Each feed's gating keywords are the file's
    shared_keywords plus the feed's optional keywords_extra.
    """
    path = Path(config_path or os.environ.get("NEWS_FEEDS_CONFIG")
                or DEFAULT_CONFIG_PATH)
    raw = json.loads(path.read_text())
    shared = tuple(raw.get("shared_keywords", ()))
    feeds = []
    for f in raw.get("feeds", ()):
        feeds.append(FeedConfig(
            name=f["name"],
            url=f["url"],
            source_id=f["source_id"],
            base_confidence=float(f.get("base_confidence", 0.5)),
            keywords=shared + tuple(f.get("keywords_extra", ())),
            format=f.get("format", "rss"),
        ))
    return tuple(feeds)


# Loaded at import so existing call sites keep working; the collector CLI
# can still pass a different registry via load_feeds(path).
DEFAULT_FEEDS: tuple[FeedConfig, ...] = load_feeds()
