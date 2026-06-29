"""Unit tests for the news collector pipeline.

Tests use mocked HTTP responses so they run offline.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from fifa_fantasy.external.news.collector import (
    _matches_keywords, _parse_rss, collect, prune_disk_budget,
)
from fifa_fantasy.external.news.extractor import extract
from fifa_fantasy.external.news.feeds import FeedConfig
from fifa_fantasy.external.news.store import (
    append_articles, disk_usage_bytes, load_articles, prune_oldest,
)


SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Sample Feed</title>
    <item>
      <title>France predicted lineup vs Norway</title>
      <link>https://example.com/article1</link>
      <pubDate>Mon, 30 Jun 2026 10:00:00 +0000</pubDate>
      <description>Predicted XI for France ahead of WC 2026 Round of 16.</description>
    </item>
    <item>
      <title>Random unrelated article</title>
      <link>https://example.com/article2</link>
      <pubDate>Mon, 30 Jun 2026 11:00:00 +0000</pubDate>
      <description>Some other sport, no football relevance.</description>
    </item>
  </channel>
</rss>"""

SAMPLE_HTML = """<html>
<head>
  <title>France predicted lineup vs Norway</title>
  <meta property="og:title" content="France predicted lineup vs Norway">
</head>
<body>
  <article>
    <p>France manager Didier Deschamps will rotate ahead of the Round of 16 match against Norway, with several first-team regulars set to be rested following an intense group stage that saw Les Bleus top their group.</p>
    <p>The starting XI is expected to include Mbappé, Dembélé, and Olise leading the line, with the midfield trio anchored by Tchouameni who returns from suspension after missing the final group fixture.</p>
    <p>Norway will rely on Erling Haaland up front for the knockout match, with manager Stale Solbakken expected to deploy a five-at-the-back formation to contain the French attack and look for opportunities on the counter.</p>
  </article>
</body>
</html>"""


def test_parse_rss_extracts_items():
    items = _parse_rss(SAMPLE_RSS)
    assert len(items) == 2
    assert items[0]["url"] == "https://example.com/article1"
    assert "predicted lineup" in items[0]["title"].lower()


def test_matches_keywords_positive():
    item = {"title": "France predicted lineup", "summary": "WC 2026"}
    assert _matches_keywords(item, ("world cup", "predicted")) is True


def test_matches_keywords_negative():
    item = {"title": "NBA news", "summary": "basketball"}
    assert _matches_keywords(item, ("world cup", "football")) is False


def test_extractor_extracts_body():
    art = extract(SAMPLE_HTML)
    assert "Deschamps" in art.body_text
    assert "Mbappé" in art.body_text or "Mbapp" in art.body_text
    assert art.title.lower().startswith("france predicted lineup")


def test_extractor_handles_empty_html():
    art = extract("<html><body></body></html>")
    assert art.body_text == ""
    assert art.snippet == ""


def test_store_append_and_load(tmp_path):
    rows = [{
        "url": "https://x/1", "source_id": "test", "source_name": "Test",
        "title": "T1", "snippet": "s", "body_text": "b" * 500,
        "byte_size": 500, "author": None,
        "published_at_utc": "2026-06-30T10:00:00+00:00",
        "collected_at_utc": "2026-06-30T11:00:00+00:00",
        "source_confidence": 0.5,
    }]
    append_articles(rows, tmp_path)
    df = load_articles(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["url"] == "https://x/1"


def test_store_dedup_by_url(tmp_path):
    row_a = {
        "url": "https://x/1", "source_id": "test", "source_name": "Test",
        "title": "T1", "snippet": "s", "body_text": "first",
        "byte_size": 5, "author": None,
        "published_at_utc": None,
        "collected_at_utc": "2026-06-30T10:00:00+00:00",
        "source_confidence": 0.5,
    }
    row_b = dict(row_a, body_text="second updated",
                 collected_at_utc="2026-06-30T12:00:00+00:00")
    append_articles([row_a], tmp_path)
    append_articles([row_b], tmp_path)
    df = load_articles(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["body_text"] == "second updated"


def test_prune_oldest(tmp_path):
    rows = [{
        "url": "https://x/1", "source_id": "test", "source_name": "Test",
        "title": "T", "snippet": "", "body_text": "b" * 100,
        "byte_size": 100, "author": None, "published_at_utc": None,
        "collected_at_utc": "2026-06-29T10:00:00+00:00",
        "source_confidence": 0.5,
    }]
    # Two separate days
    (tmp_path / "news_2026-06-29.parquet").touch()
    pd.DataFrame(rows).to_parquet(tmp_path / "news_2026-06-29.parquet", index=False)
    pd.DataFrame(rows).to_parquet(tmp_path / "news_2026-06-30.parquet", index=False)
    deleted = prune_oldest(tmp_path)
    assert deleted.name == "news_2026-06-29.parquet"
    assert (tmp_path / "news_2026-06-30.parquet").exists()


def test_disk_budget_enforcement(tmp_path):
    # Create three large-ish parquets, then prune to < 0.001 MB budget.
    big_row = {
        "url": "https://x/", "source_id": "t", "source_name": "T",
        "title": "T", "snippet": "", "body_text": "b" * 10000,
        "byte_size": 10000, "author": None, "published_at_utc": None,
        "collected_at_utc": "2026-06-30T10:00:00+00:00",
        "source_confidence": 0.5,
    }
    for d in ("2026-06-27", "2026-06-28", "2026-06-29"):
        rows = [dict(big_row, url=f"https://x/{d}/{i}") for i in range(20)]
        pd.DataFrame(rows).to_parquet(tmp_path / f"news_{d}.parquet", index=False)
    initial_usage = disk_usage_bytes(tmp_path)
    assert initial_usage > 1000
    pruned = prune_disk_budget(tmp_path, budget_mb=0.001)
    assert pruned >= 1
    assert disk_usage_bytes(tmp_path) < initial_usage


def test_collect_end_to_end_with_mocked_client(tmp_path):
    """Full collect() flow with a mocked StealthClient."""
    client = MagicMock()
    rss_response = MagicMock(text=SAMPLE_RSS, status_code=200)
    article_response = MagicMock(text=SAMPLE_HTML, status_code=200)
    # First .get() returns the RSS; subsequent .get()s return the article.
    client.get.side_effect = [rss_response, article_response]

    feed = FeedConfig(
        name="Test", url="https://example.com/rss",
        source_id="test", base_confidence=0.6,
        keywords=("predicted", "lineup"),
    )
    summary = collect(client, feeds=(feed,), out_dir=tmp_path, budget_mb=10)
    assert summary["items_stored"] == 1
    df = load_articles(tmp_path)
    assert len(df) == 1
    assert df.iloc[0]["source_id"] == "test"
    assert "Deschamps" in df.iloc[0]["body_text"]
