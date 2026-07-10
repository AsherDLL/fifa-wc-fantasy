"""CLI: run one news collection pass.

    python -m fifa_fantasy.external.news
    python -m fifa_fantasy.external.news --budget-mb 100 --max-per-feed 15

Outputs: data/external/news_articles/news_<date>.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..scraping import StealthClient
from .collector import DEFAULT_BUDGET_MB, collect
from .feeds import load_feeds
from .store import DEFAULT_DIR


def main() -> int:
    p = argparse.ArgumentParser(prog="fifa_fantasy.external.news")
    p.add_argument("--budget-mb", type=float, default=DEFAULT_BUDGET_MB,
                   help="Total disk-usage cap for news_articles/ in MB")
    p.add_argument("--max-per-feed", type=int, default=25,
                   help="Cap items checked per feed per run")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/external/cache/scraping"))
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--feeds-config", type=Path, default=None,
                   help="Feed registry JSON (default: packaged config/feeds.json"
                        " or NEWS_FEEDS_CONFIG env var)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Use kewpie's coherent, rotating identity pool (modern Chrome/Firefox
    # builds with matching Sec-CH-UA + Accept-Language) rather than pinning a
    # single stale target. Pinning impersonate="chrome124" sent a Chrome TLS
    # handshake with an empty User-Agent, a contradiction WAFs flag.
    client = StealthClient(
        rate_limit_per_second=0.5,    # conservative
        cache_dir=str(args.cache_dir),
        cache_ttl_hours=2.0,
        max_retries=2,
    )
    summary = collect(
        client, feeds=load_feeds(args.feeds_config),
        out_dir=args.out_dir,
        budget_mb=args.budget_mb,
        max_items_per_feed=args.max_per_feed,
    )
    log = logging.getLogger("news")
    log.info("=== summary ===")
    for k, v in summary.items():
        log.info("  %s: %s", k, v)
    client.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
