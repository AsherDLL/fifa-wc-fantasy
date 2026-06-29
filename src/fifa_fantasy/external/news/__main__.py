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
from .feeds import DEFAULT_FEEDS
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
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    client = StealthClient(
        impersonate="chrome124",
        rate_limit_per_second=0.5,    # conservative
        cache_dir=str(args.cache_dir),
        cache_ttl_hours=2.0,
        max_retries=2,
    )
    summary = collect(
        client, feeds=DEFAULT_FEEDS,
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
