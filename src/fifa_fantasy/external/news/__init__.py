"""News article collection: RSS-first, lean, disk-budgeted.

Two-stage flow:

  1. Poll RSS feeds for new articles. Cheap (small XML payloads).
  2. Filter by keyword/team-name relevance.
  3. Fetch the article's full HTML only when filter matches.
  4. Extract article body text + metadata; discard the raw HTML.
  5. Persist a single parquet row per article (~5-20KB after compression).

A disk budget cap is enforced before every write. When approaching the
cap, oldest articles are pruned. Total disk usage stays bounded.

This module is general-purpose: it produces (url, title, published_at,
source, body_text, snippet) rows. Domain-specific parsers (e.g.
`team_news/parsers/espn.py`) consume the persisted articles for their
own extraction.
"""
from .collector import collect, prune_disk_budget
from .feeds import DEFAULT_FEEDS, FeedConfig
from .store import load_articles

__all__ = ["collect", "prune_disk_budget", "DEFAULT_FEEDS", "FeedConfig",
           "load_articles"]
