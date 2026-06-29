"""Unit tests for the reusable scraping module.

We avoid hitting the real network in unit tests by mocking the underlying
curl_cffi session. The smoke test in test_scraping_client_live.py (gated
by a network marker) exercises the real path.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fifa_fantasy.external.scraping import StealthClient
from fifa_fantasy.external.scraping.cache import DiskCache
from fifa_fantasy.external.scraping.rate_limit import PerHostRateLimiter
from fifa_fantasy.external.scraping.retry import retry_with_backoff


# ----- rate limiter -----

def test_rate_limiter_serializes_same_host(tmp_path):
    limiter = PerHostRateLimiter(requests_per_second=10.0)  # 100ms interval
    start = time.monotonic()
    for _ in range(3):
        limiter.acquire("https://example.com/")
    elapsed = time.monotonic() - start
    # 3 requests at 100ms each = at least 200ms of total sleep
    assert elapsed >= 0.18


def test_rate_limiter_different_hosts_independent():
    limiter = PerHostRateLimiter(requests_per_second=1.0)
    start = time.monotonic()
    limiter.acquire("https://a.example/")
    limiter.acquire("https://b.example/")  # different host, no wait
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


# ----- cache -----

def test_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path, ttl_hours=1.0)
    assert cache.get("GET", "https://x.example/", None) is None
    cache.put("GET", "https://x.example/", None,
              body=b"hello", status_code=200,
              response_headers={"Content-Type": "text/plain"})
    hit = cache.get("GET", "https://x.example/", None)
    assert hit is not None
    assert hit.body == b"hello"
    assert hit.status_code == 200
    assert hit.from_cache is True


def test_cache_ttl_expiry(tmp_path):
    cache = DiskCache(tmp_path, ttl_hours=0.0)  # 0 hours = always expired
    cache.put("GET", "https://x.example/", None,
              body=b"hello", status_code=200,
              response_headers={})
    # Even an immediate read is expired (TTL = 0).
    assert cache.get("GET", "https://x.example/", None) is None


# ----- retry -----

def test_retry_returns_on_success():
    attempts = []
    def fn():
        attempts.append(1)
        return "ok"
    assert retry_with_backoff(fn, max_attempts=3) == "ok"
    assert len(attempts) == 1


def test_retry_exhausts_and_raises():
    def fn():
        raise ConnectionError("boom")
    with pytest.raises(ConnectionError):
        retry_with_backoff(fn, max_attempts=2, base_delay_s=0.01,
                          is_retryable=lambda e: isinstance(e, ConnectionError))


# ----- StealthClient end-to-end with mocked transport -----

def test_stealth_client_cache_hit_short_circuits(tmp_path):
    """Cache hit returns immediately without calling the underlying transport."""
    cache_dir = tmp_path / "cache"
    client = StealthClient(impersonate="chrome124",
                           rate_limit_per_second=100.0,
                           cache_dir=cache_dir, cache_ttl_hours=1.0)
    # Mock the underlying transport.
    with patch.object(client, "_do_get") as mock_get:
        mock_resp = MagicMock(text="hello", content=b"hello",
                              status_code=200, headers={}, url="https://x/",
                              from_cache=False)
        mock_get.return_value = mock_resp
        # First call: cache miss, transport called.
        r1 = client.get("https://x/")
        assert r1.from_cache is False
        assert mock_get.call_count == 1
        # Second call: cache hit, transport NOT called again.
        r2 = client.get("https://x/")
        assert r2.from_cache is True
        assert mock_get.call_count == 1


def test_stealth_client_retries_5xx(tmp_path):
    client = StealthClient(impersonate="chrome124",
                           rate_limit_per_second=100.0,
                           cache_dir=None,
                           max_retries=3)
    responses = [
        MagicMock(text="", content=b"", status_code=503, headers={}, url="https://x/", from_cache=False),
        MagicMock(text="", content=b"", status_code=503, headers={}, url="https://x/", from_cache=False),
        MagicMock(text="ok", content=b"ok", status_code=200, headers={}, url="https://x/", from_cache=False),
    ]
    with patch.object(client, "_do_get", side_effect=responses) as mock_get:
        r = client.get("https://x/")
    assert r.status_code == 200
    assert mock_get.call_count == 3
