"""Reusability demo #2: scrape startup data for niche research.

Shows that the `scraping` module + the `news` collector work outside
the FIFA fantasy domain. Configures custom RSS feeds + keyword filters
for a tech / startup niche, runs one collection pass, and prints
what was collected.

Adapt the FEEDS list and KEYWORDS below to your own niche.

Run from the repo root:
    .venv/bin/python docs/scraping/startup_research_demo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add src/ to sys.path so this runs as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fifa_fantasy.external.news.collector import collect
from fifa_fantasy.external.news.feeds import FeedConfig
from fifa_fantasy.external.news.store import load_articles
from fifa_fantasy.external.scraping import StealthClient


# Keywords gating which articles get the full-body fetch. Wide enough
# to capture relevant content, narrow enough to skip irrelevant news.
KEYWORDS = (
    "startup", "seed round", "series a", "series b", "series c",
    "funding", "venture capital", "vc", "raises", "valuation",
    "acquisition", "acquired", "ipo", "yc", "y combinator",
    "founder", "ceo", "co-founder", "saas", "ai startup",
    "fintech", "biotech", "deeptech", "climate tech",
)


# Curated free RSS feeds for startup / VC news. All public, no auth.
FEEDS = (
    FeedConfig(
        name="TechCrunch Startups",
        url="https://techcrunch.com/category/startups/feed/",
        source_id="techcrunch_startups",
        base_confidence=0.75,
        keywords=KEYWORDS,
    ),
    FeedConfig(
        name="TechCrunch Venture",
        url="https://techcrunch.com/category/venture/feed/",
        source_id="techcrunch_venture",
        base_confidence=0.75,
        keywords=KEYWORDS,
    ),
    FeedConfig(
        name="Crunchbase News",
        url="https://news.crunchbase.com/feed/",
        source_id="crunchbase",
        base_confidence=0.80,
        keywords=KEYWORDS,
    ),
    FeedConfig(
        name="The Information",
        url="https://www.theinformation.com/feed",
        source_id="theinformation",
        base_confidence=0.85,
        keywords=KEYWORDS,
    ),
    FeedConfig(
        name="VentureBeat",
        url="https://venturebeat.com/feed/",
        source_id="venturebeat",
        base_confidence=0.70,
        keywords=KEYWORDS,
    ),
    FeedConfig(
        name="Hacker News (front page items only)",
        url="https://hnrss.org/frontpage",
        source_id="hn_frontpage",
        base_confidence=0.60,
        keywords=KEYWORDS,
    ),
)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("startup-demo")

    out_dir = Path("data/external/startup_articles_demo")

    client = StealthClient(
        impersonate="chrome124",
        rate_limit_per_second=0.5,
        cache_dir="data/external/cache/scraping_demo",
        cache_ttl_hours=2.0,
        max_retries=2,
    )

    log.info("Running collection pass over %d feeds...", len(FEEDS))
    summary = collect(
        client, feeds=FEEDS,
        out_dir=out_dir,
        budget_mb=2048,            # 2GB budget for the startup-data corpus
        max_items_per_feed=30,
    )

    log.info("=== collection summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)

    log.info("=== sample of stored articles ===")
    df = load_articles(out_dir)
    if df.empty:
        log.info("  (none stored yet)")
    else:
        df = df.sort_values("collected_at_utc", ascending=False).head(10)
        for r in df.itertuples():
            log.info("  [%s] %s", r.source_id, r.title[:100])

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
