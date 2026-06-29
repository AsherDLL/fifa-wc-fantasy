"""Extract article body text + metadata from HTML.

Heuristic-based; works adequately on most modern article pages. We
prefer the <article>, <main>, or first <div role="main"> as the body
root, then collect all <p> text. Falls back to all <p> on the page if
none of the above are found.

We do NOT use a JS engine; some sites serve their article body via
hydration on the client, and we accept that those will produce
empty/short extractions (they're logged and skipped). The collector
filters on keyword match against the article title before fetching, so
the cost of a failed extraction is small.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class ExtractedArticle:
    title: str
    body_text: str
    snippet: str
    published_at_utc: Optional[datetime] = None
    author: Optional[str] = None
    canonical_url: Optional[str] = None


_BODY_SELECTORS = (
    "article",
    "main",
    'div[role="main"]',
    "div.article-body",
    "div.story-body",
    "div.entry-content",
)


def _select_body(soup: BeautifulSoup) -> BeautifulSoup:
    for sel in _BODY_SELECTORS:
        el = soup.select_one(sel)
        if el is not None:
            return el
    return soup  # whole page fallback


def _collect_paragraphs(root) -> list[str]:
    out = []
    for p in root.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text and len(text) > 20:  # drop nav/footer junk
            out.append(text)
    return out


def _normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract(html: str, fallback_title: str = "") -> ExtractedArticle:
    """Return ExtractedArticle from an HTML page.

    Always returns an object (never None); empty body_text means the
    extractor could not find content. Caller decides whether to store
    or skip.
    """
    soup = BeautifulSoup(html, "lxml")

    # Title: <title>, <meta property="og:title">, or fallback.
    title = fallback_title
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    og_title = soup.select_one('meta[property="og:title"]')
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    # Body.
    body_root = _select_body(soup)
    paragraphs = _collect_paragraphs(body_root)
    body_text = "\n\n".join(paragraphs)
    body_text = _normalize_whitespace(body_text)

    # First 280 chars as snippet for index / preview.
    snippet = body_text[:280] + ("..." if len(body_text) > 280 else "")

    # Published date (best-effort).
    published_at = None
    time_tag = soup.find("time")
    if time_tag and time_tag.get("datetime"):
        try:
            published_at = datetime.fromisoformat(
                time_tag["datetime"].replace("Z", "+00:00"))
        except ValueError:
            pass

    # Author (best-effort).
    author = None
    author_meta = soup.select_one('meta[name="author"]')
    if author_meta and author_meta.get("content"):
        author = author_meta["content"].strip()

    canonical = soup.select_one('link[rel="canonical"]')
    canonical_url = canonical["href"] if canonical and canonical.get("href") else None

    return ExtractedArticle(
        title=_normalize_whitespace(title),
        body_text=body_text,
        snippet=snippet,
        published_at_utc=published_at,
        author=author,
        canonical_url=canonical_url,
    )
