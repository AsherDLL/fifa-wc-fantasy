"""Curated free RSS feeds for football / WC 2026 team news.

We intentionally choose RSS feeds (XML, ~5-50KB per fetch) over scraping
HTML index pages (~500KB-2MB). RSS is light, structured, and respected
by every source we've added below.

Twitter / X is deliberately NOT included. As of 2026, Nitter's public
instances are dead, twscrape requires authenticated accounts, and the
public-frontend ecosystem is unstable. The cost (legal, technical) is
not worth the marginal lineup-signal it provides. Documented in
`docs/whitepaper/sections/11e_team_news_ingestion.md`.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeedConfig:
    """One RSS feed."""
    name: str
    url: str
    source_id: str             # short id used in storage
    base_confidence: float = 0.5
    # Keywords (case-insensitive) that gate which articles we fetch.
    keywords: tuple[str, ...] = field(default_factory=tuple)


# WC 2026 + football team-news relevant keywords. Any article whose
# title OR description matches at least one is fetched.
WC_KEYWORDS = (
    "world cup", "fifa", "wc 2026", "wc26",
    "predicted xi", "predicted lineup", "starting xi", "team news",
    "lineup", "line-up", "line up",
    "injury", "doubt", "suspended", "rotation",
    # Country names for the 32 R32 advancers (subset of likely surviving teams).
    "argentina", "france", "spain", "england", "brazil", "portugal",
    "netherlands", "germany", "colombia", "usa", "mexico", "morocco",
    "japan", "norway", "belgium", "switzerland", "ecuador", "egypt",
    "australia", "canada", "uruguay", "croatia", "ghana",
)


DEFAULT_FEEDS: tuple[FeedConfig, ...] = (
    FeedConfig(
        name="BBC Sport Football",
        url="https://feeds.bbci.co.uk/sport/football/rss.xml",
        source_id="bbc",
        base_confidence=0.75,
        keywords=WC_KEYWORDS,
    ),
    FeedConfig(
        name="Sky Sports Football",
        url="https://www.skysports.com/rss/12040",
        source_id="skysports",
        base_confidence=0.70,
        keywords=WC_KEYWORDS,
    ),
    FeedConfig(
        name="Guardian Football",
        url="https://www.theguardian.com/football/rss",
        source_id="guardian",
        base_confidence=0.75,
        keywords=WC_KEYWORDS,
    ),
    # FourFourTwo: very high WC-2026 density (11/30 top items match WC
    # keywords in our probe). Strong replacement for the dropped feeds.
    FeedConfig(
        name="FourFourTwo",
        url="https://www.fourfourtwo.com/feeds/all",
        source_id="fourfourtwo",
        base_confidence=0.65,
        keywords=WC_KEYWORDS,
    ),
    # talkSPORT: high-volume general football news including WC coverage.
    FeedConfig(
        name="talkSPORT",
        url="https://talksport.com/feed/",
        source_id="talksport",
        base_confidence=0.55,
        keywords=WC_KEYWORDS,
    ),
)

# Dropped feeds (kept here for documentation; we do not retry them):
#   - BBC Sport World Football (feeds.bbci.co.uk/sport/football/world/rss.xml):
#     404 as of 2026-06-29. The general BBC Sport Football feed above
#     already includes WC coverage, so the loss is small.
#   - ESPN Soccer (www.espn.com/espn/rss/soccer/news):
#     202 with an AWS WAF JavaScript challenge body. ESPN's RSS endpoint
#     is now protected by AWS WAF. Defeating it requires
#     playwright-stealth or nodriver (see scraping/README.md escalation
#     tier 2). Out of scope for the lean RSS collector.
#   - Goal.com (www.goal.com/feeds/en/news):
#     404 with no working alternative URL. Their RSS endpoint moved or
#     was retired; we could not find a replacement.
