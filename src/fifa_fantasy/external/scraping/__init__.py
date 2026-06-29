"""Reusable stealth scraping module.

A small, focused HTTP client wrapper that handles the four most common
bot-detection mechanisms (TLS fingerprinting, header order, rate limiting,
IP reputation) with sensible defaults. Sufficient for the majority of
modern websites; escalation paths documented in `README.md` for the
sites this module cannot defeat.

Public API:

    from fifa_fantasy.external.scraping import StealthClient

    client = StealthClient(
        impersonate="chrome124",
        rate_limit_per_host=1.0,        # 1 request per second
        cache_dir="data/cache",
        cache_ttl_hours=6,
    )
    response = client.get("https://example.com")
    print(response.text, response.from_cache)
"""
from .client import CachedResponse, StealthClient

__all__ = ["StealthClient", "CachedResponse"]
