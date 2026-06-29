"""Session warm-up: visit a homepage before deep-fetching to look human.

Some sites set first-party cookies on the homepage that they then check
on every subsequent request. A scraper that lands on a deep URL with no
cookies looks suspicious. Warm-up visits the homepage, collects cookies,
and reuses them for the rest of the session.
"""
from __future__ import annotations

from urllib.parse import urlparse


def homepage_url(deep_url: str) -> str:
    """Return the scheme://host/ for the given URL."""
    p = urlparse(deep_url)
    return f"{p.scheme}://{p.netloc}/"
