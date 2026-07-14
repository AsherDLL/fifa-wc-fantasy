"""StealthClient: the public face of the scraping module.

Combines:
- curl_cffi for TLS/JA3/JA4 fingerprint impersonation
- A rotating identity pool (modern Chrome + Firefox builds) so the
  fleet of hosts does not look like one client
- Per-host sticky identity so cookies, ETags and bot-mgr scoring
  survive across requests to the same host
- Coherent navigation headers (Sec-CH-UA, Sec-Fetch-*, Accept-Language)
  matched to the impersonated browser
- PerHostRateLimiter for ethical/safe pacing
- DiskCache for cheap re-runs and ethical avoidance of re-fetching
- ProxyRotator for IP rotation when sites IP-block
- retry_with_backoff for transient failures
- Body-fingerprint block detection (Cloudflare interstitial, AWS WAF
  challenge, DataDome / Akamai pages) with identity rotation on block

Usage:
    from fifa_fantasy.external.scraping import StealthClient

    client = StealthClient(
        rate_limit_per_second=1.0,
        cache_dir="data/cache",
        cache_ttl_hours=6,
    )
    r = client.get("https://example.com")
    print(r.text)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

try:
    from curl_cffi import requests as cc_requests
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "curl_cffi is required for StealthClient. Install with "
        "`pip install curl_cffi`."
    ) from e

from .cache import DiskCache
from .identity import DEFAULT_POOL, Identity, pick_for_host
from .proxies import ProxyRotator
from .rate_limit import PerHostRateLimiter
from .retry import RETRY_STATUSES, detect_block, retry_with_backoff

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


class StealthClient:
    """Reusable, anti-bot-detection-aware HTTP GET client.

    Defaults are conservative and ethical. Override only when you know
    your target's policies and have permission.
    """

    def __init__(
        self,
        rate_limit_per_second: float = 1.0,
        cache_dir: Path | str | None = None,
        cache_ttl_hours: float = 6.0,
        max_retries: int = 3,
        proxy_rotator: ProxyRotator | None = None,
        default_timeout_s: float = 30.0,
        identity_pool: tuple[Identity, ...] = DEFAULT_POOL,
    ):
        self.rate_limiter = PerHostRateLimiter(rate_limit_per_second)
        self.cache: DiskCache | None = (
            DiskCache(cache_dir, ttl_hours=cache_ttl_hours) if cache_dir else None
        )
        self.max_retries = max_retries
        self.proxy_rotator = proxy_rotator or ProxyRotator.from_env()
        self.timeout_s = default_timeout_s
        self.identity_pool = identity_pool
        # Per-host: identity index in the pool (so blocks can rotate by ++).
        self._host_identity_idx: dict[str, int] = {}
        # Per-host curl_cffi session, keyed by (host, identity index).
        self._sessions: dict[tuple[str, int], cc_requests.Session] = {}

    # ----- identity / session management -----

    def _identity_for(self, host: str) -> tuple[Identity, int]:
        if host not in self._host_identity_idx:
            # Stable initial pick so resuming a run hits the same identity.
            chosen = pick_for_host(host, self.identity_pool)
            idx = self.identity_pool.index(chosen)
            self._host_identity_idx[host] = idx
        idx = self._host_identity_idx[host]
        return self.identity_pool[idx], idx

    def _session_for(self, host: str, idx: int) -> cc_requests.Session:
        key = (host, idx)
        s = self._sessions.get(key)
        if s is None:
            ident = self.identity_pool[idx]
            s = cc_requests.Session(impersonate=ident.impersonate)
            self._sessions[key] = s
        return s

    def _rotate_identity(self, host: str) -> None:
        """Advance the host's identity to the next pool entry. Called when
        we suspect we have been block-scored by the previous one."""
        current = self._host_identity_idx.get(host, 0)
        self._host_identity_idx[host] = (current + 1) % len(self.identity_pool)
        log.info("rotated identity for %s -> %s", host,
                 self.identity_pool[self._host_identity_idx[host]].name)

    # ----- HTTP plumbing -----

    def _do_get(self, url: str, headers: Mapping[str, str] | None) -> CachedResponse:
        host = urlparse(url).netloc or url
        ident, idx = self._identity_for(host)
        session = self._session_for(host, idx)

        # Build coherent headers: identity defaults + caller overrides.
        merged: dict[str, str] = ident.navigation_headers() if ident.user_agent else {}
        if headers:
            merged.update(headers)

        proxy = self.proxy_rotator.pick()
        proxies = {"http": proxy.url, "https": proxy.url} if proxy else None
        r = session.get(
            url,
            headers=merged if merged else None,
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

    def get(self, url: str,
            headers: Mapping[str, str] | None = None,
            bypass_cache: bool = False) -> CachedResponse:
        """GET a URL with stealth, caching, rate limit, and retries."""
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

        host = urlparse(url).netloc or url

        def _attempt() -> CachedResponse:
            self.rate_limiter.acquire(url)
            return self._do_get(url, headers)

        def _retry_check(value_or_exc) -> bool:
            if isinstance(value_or_exc, Exception):
                name = type(value_or_exc).__name__.lower()
                return any(s in name for s in ("connection", "timeout", "transport"))
            # Status-code retry path.
            if value_or_exc.status_code in RETRY_STATUSES:
                return True
            # Body-fingerprint block detection: a 200 with a Cloudflare
            # interstitial page is still a block. Rotate identity and
            # retry, but only if we have multiple identities to try.
            block = detect_block(value_or_exc.status_code, value_or_exc.headers,
                                 value_or_exc.text or "")
            if block is not None:
                log.warning("block detected on %s: %s", url, block)
                if len(self.identity_pool) > 1:
                    self._rotate_identity(host)
                    return True
            return False

        resp = retry_with_backoff(
            _attempt,
            max_attempts=self.max_retries,
            is_retryable=_retry_check,
        )

        # Cache 2xx responses only when they look real (no block fingerprint).
        if (self.cache is not None
                and 200 <= resp.status_code < 300
                and detect_block(resp.status_code, resp.headers, resp.text or "") is None):
            self.cache.put(
                "GET", url, headers,
                body=resp.content,
                status_code=resp.status_code,
                response_headers=resp.headers,
            )
        return resp

    def close(self) -> None:
        for s in self._sessions.values():
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        self._sessions.clear()
