"""Lean RSS-first article collector with disk budget enforcement.

For each configured feed:
  1. Fetch the RSS XML (cheap, 5-50KB).
  2. Parse items: title, url, published, summary.
  3. Filter items by keyword match against title + summary.
  4. For each matching item, fetch the article HTML using the stealth
     client (rate-limited, cached).
  5. Extract body text via extractor.extract().
  6. Append (url, source_id, title, snippet, body_text_first_N,
     published_at_utc, collected_at_utc, byte_size) row to today's
     parquet.
  7. If total disk usage exceeds the budget, prune the oldest day(s).

We deliberately store only the first N=8000 chars of body text per
article (~8KB before compression, ~1-3KB after parquet/snappy). This
caps a single article's footprint and keeps the total bounded.

A run with 6 feeds × ~30 items/feed × ~10% match rate = ~18 articles
fetched per tick. At 3KB compressed each = ~50KB/tick. Hourly = ~1.2MB/day.
Weekly = ~8MB. Easily within any reasonable disk budget.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..scraping import StealthClient
from .extractor import extract
from .feeds import DEFAULT_FEEDS, FeedConfig
from .store import (
    DEFAULT_DIR, append_articles, disk_usage_bytes, prune_oldest,
)

log = logging.getLogger(__name__)

MAX_BODY_CHARS = 12000         # cap stored body text per article (~1-4KB after parquet+snappy)
DEFAULT_BUDGET_MB = 2048       # total disk budget for news_articles/ (~2GB)


def _parse_rss(xml_text: str) -> list[dict]:
    """Minimal RSS 2.0 / Atom parser.

    Returns a list of dicts with keys: title, url, published, summary.
    Uses standard library xml.etree for zero-dependency, defensive parsing.
    """
    from xml.etree import ElementTree as ET
    items = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return items

    # RSS 2.0
    for it in root.iter():
        tag = it.tag.split("}")[-1]
        if tag != "item" and tag != "entry":
            continue
        d = {"title": "", "url": "", "published": "", "summary": ""}
        for child in it:
            ctag = child.tag.split("}")[-1]
            text = (child.text or "").strip()
            if ctag == "title":
                d["title"] = text
            elif ctag == "link":
                href = child.attrib.get("href")
                d["url"] = href if href else text
            elif ctag in ("pubDate", "published", "updated"):
                d["published"] = text
            elif ctag in ("description", "summary"):
                d["summary"] = text
        if d["url"]:
            items.append(d)
    return items


def _matches_keywords(item: dict, keywords: tuple[str, ...]) -> bool:
    haystack = (item.get("title", "") + " " + item.get("summary", "")).lower()
    return any(k.lower() in haystack for k in keywords)


def _parse_published(s: str) -> datetime | None:
    if not s:
        return None
    # RFC 2822 (RSS pubDate)
    try:
        from email.utils import parsedate_to_datetime
        d = parsedate_to_datetime(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d
    except (TypeError, ValueError):
        pass
    # ISO 8601 (Atom)
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def prune_disk_budget(out_dir: Path, budget_mb: float) -> int:
    """Prune oldest day-parquets until usage < budget. Returns count pruned."""
    budget_bytes = int(budget_mb * 1024 * 1024)
    pruned = 0
    while disk_usage_bytes(out_dir) > budget_bytes:
        deleted = prune_oldest(out_dir)
        if deleted is None:
            break
        pruned += 1
        log.info("pruned %s (over budget)", deleted)
    return pruned


def collect(
    client: StealthClient,
    feeds: tuple[FeedConfig, ...] = DEFAULT_FEEDS,
    out_dir: Path = DEFAULT_DIR,
    budget_mb: float = DEFAULT_BUDGET_MB,
    max_items_per_feed: int = 25,
    skip_seen_within_hours: float = 12.0,
) -> dict:
    """Run one collection pass.

    Returns a summary dict with counts and disk usage.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "feeds": len(feeds),
        "items_seen": 0,
        "items_matched": 0,
        "items_fetched": 0,
        "items_stored": 0,
        "bytes_added": 0,
        "errors": 0,
    }

    # Build set of URLs seen recently to dedup.
    from .store import load_articles
    recent = load_articles(out_dir, since_days=skip_seen_within_hours / 24.0)
    seen_urls = set(recent["url"]) if not recent.empty else set()

    new_rows: list[dict] = []
    for feed in feeds:
        try:
            rss_resp = client.get(feed.url)
        except Exception as e:  # noqa: BLE001
            log.warning("feed %s fetch failed: %s", feed.name, e)
            summary["errors"] += 1
            continue
        if rss_resp.status_code != 200:
            log.warning("feed %s returned %d", feed.name, rss_resp.status_code)
            summary["errors"] += 1
            continue
        items = _parse_rss(rss_resp.text)[:max_items_per_feed]
        summary["items_seen"] += len(items)

        for item in items:
            if item["url"] in seen_urls:
                continue
            if not _matches_keywords(item, feed.keywords):
                continue
            summary["items_matched"] += 1

            try:
                art_resp = client.get(item["url"])
            except Exception as e:  # noqa: BLE001
                log.debug("article fetch failed: %s -> %s", item["url"], e)
                summary["errors"] += 1
                continue
            summary["items_fetched"] += 1
            if art_resp.status_code != 200 or not art_resp.text:
                continue

            article = extract(art_resp.text, fallback_title=item.get("title", ""))
            if not article.body_text or len(article.body_text) < 200:
                continue
            body = article.body_text[:MAX_BODY_CHARS]
            row = {
                "url": item["url"],
                "source_id": feed.source_id,
                "source_name": feed.name,
                "title": article.title or item.get("title", ""),
                "snippet": article.snippet,
                "body_text": body,
                "byte_size": len(body.encode("utf-8")),
                "author": article.author,
                "published_at_utc": (
                    article.published_at_utc.isoformat()
                    if article.published_at_utc
                    else _parse_published(item.get("published", "")).isoformat()
                    if _parse_published(item.get("published", "")) is not None
                    else None
                ),
                "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_confidence": feed.base_confidence,
            }
            new_rows.append(row)
            summary["items_stored"] += 1
            summary["bytes_added"] += row["byte_size"]
            seen_urls.add(item["url"])

    if new_rows:
        append_articles(new_rows, out_dir)
    prune_disk_budget(out_dir, budget_mb)
    summary["disk_usage_mb"] = disk_usage_bytes(out_dir) / (1024 * 1024)
    summary["budget_mb"] = budget_mb
    return summary
