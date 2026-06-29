"""StealthClient: the public face of the scraping module.

Combines:
- curl_cffi for TLS/JA3 fingerprint impersonation (defeats Cloudflare's
  pre-application TLS check)
- PerHostRateLimiter for ethical/safe pacing
- DiskCache for cheap re-runs and ethical avoidance of re-fetching
- ProxyRotator for IP rotation when sites IP-block
- retry_with_backoff for transient failures

Usage:
    from fifa_fantasy.external.scraping import StealthClient

    client = StealthClient(
        impersonate="chrome124",
        rate_limit_per_second=1.0,
        cache_dir="data/cache",
        cache_ttl_hours=6,
    )
    r = client.get("https://example.com")
    print(r.text)

The StealthClient is sync and thread-safe. For async / massive parallel
scraping, use the underlying primitives directly with asyncio.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

try:
    from curl_cffi import requests as cc_requests
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "curl_cffi is required for StealthClient. Install with "
        "`pip install curl_cffi`."
    ) from e

from .cache import DiskCache
from .proxies import ProxyConfig, ProxyRotator
from .rate_limit import PerHostRateLimiter
from .retry import RETRY_STATUSES, retry_with_backoff
from .session import homepage_url

log = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    """Wrapped response with the same minimal surface as curl_cffi.Response."""
    text: str
    content: bytes
    status_code: int
    headers: dict
    url: str
    from_cache: bool

    def json(self):
        import json
        return json.loads(self.text)


class StealthClient:
    """A reusable, anti-bot-detection-aware HTTP GET client.

    Defaults are conservative and ethical. Override only when you know
    your target's policies and have permission.
    """

    def __init__(
        self,
        impersonate: str = "chrome124",
        rate_limit_per_second: float = 1.0,
        cache_dir: Path | str | None = None,
        cache_ttl_hours: float = 6.0,
        max_retries: int = 3,
        proxy_rotator: ProxyRotator | None = None,
        default_timeout_s: float = 30.0,
        warm_session: bool = False,
    ):
        self.impersonate = impersonate
        self.rate_limiter = PerHostRateLimiter(rate_limit_per_second)
        self.cache: DiskCache | None = (
            DiskCache(cache_dir, ttl_hours=cache_ttl_hours) if cache_dir else None
        )
        self.max_retries = max_retries
        self.proxy_rotator = proxy_rotator or ProxyRotator.from_env()
        self.timeout_s = default_timeout_s
        self.warm_session = warm_session
        self._warmed_hosts: set[str] = set()
        self._cc_session = cc_requests.Session(impersonate=impersonate)

    def _do_get(self, url: str, headers: Mapping[str, str] | None) -> CachedResponse:
        proxy = self.proxy_rotator.pick()
        proxies = {"http": proxy.url, "https": proxy.url} if proxy else None
        r = self._cc_session.get(
            url,
            headers=dict(headers) if headers else None,
            timeout=self.timeout_s,
            proxies=proxies,
        )
        return CachedResponse(
            text=r.text,
            content=r.content,
            status_code=r.status_code,
            headers=dict(r.headers),
            url=str(r.url),
            from_cache=False,
        )

    def _warm(self, url: str, headers: Mapping[str, str] | None) -> None:
        if not self.warm_session:
            return
        from urllib.parse import urlparse
        host = urlparse(url).netloc
        if not host or host in self._warmed_hosts:
            return
        home = homepage_url(url)
        try:
            self.rate_limiter.acquire(home)
            self._do_get(home, headers)
            self._warmed_hosts.add(host)
            log.debug("warmed session for %s", host)
        except Exception as e:  # noqa: BLE001
            log.debug("warmup failed for %s: %s", host, e)

    def get(self, url: str,
            headers: Mapping[str, str] | None = None,
            bypass_cache: bool = False) -> CachedResponse:
        """GET a URL with stealth, caching, rate limit, and retries."""
        # Cache hit short-circuits all of the above.
        if self.cache is not None and not bypass_cache:
            hit = self.cache.get("GET", url, headers)
            if hit is not None:
                log.debug("cache hit: %s (stored %s)", url, hit.stored_at_utc)
                return CachedResponse(
                    text=hit.body.decode("utf-8", errors="replace"),
                    content=hit.body,
                    status_code=hit.status_code,
                    headers=hit.headers,
                    url=hit.url,
                    from_cache=True,
                )

        self._warm(url, headers)

        def _attempt() -> CachedResponse:
            self.rate_limiter.acquire(url)
            return self._do_get(url, headers)

        def _retry_check(value_or_exc) -> bool:
            if isinstance(value_or_exc, Exception):
                name = type(value_or_exc).__name__.lower()
                return any(s in name for s in ("connection", "timeout", "transport"))
            return value_or_exc.status_code in RETRY_STATUSES

        resp = retry_with_backoff(
            _attempt,
            max_attempts=self.max_retries,
            is_retryable=_retry_check,
        )

        # Cache 2xx responses.
        if self.cache is not None and 200 <= resp.status_code < 300:
            self.cache.put(
                "GET", url, headers,
                body=resp.content,
                status_code=resp.status_code,
                response_headers=resp.headers,
            )
        return resp

    def close(self) -> None:
        try:
            self._cc_session.close()
        except Exception:  # noqa: BLE001
            pass
