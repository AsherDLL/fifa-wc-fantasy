"""Browser identity rotation pool.

A single curl_cffi `impersonate=` string is no longer enough in 2026.
Cloudflare, Akamai, DataDome and AWS WAF all cross-check the TLS
fingerprint against the HTTP headers (User-Agent, Sec-CH-UA, Accept-
Language). An old or commonly-impersonated browser plus a mismatched
Accept-Language is a known scraper tell.

This module defines a pool of `Identity` records, each pinning a
modern curl_cffi impersonate target to a coherent set of client-hint
headers. The `StealthClient` picks one identity per host (stable
across that host's requests so cookies and ETags work) and rotates
across hosts so the whole fleet does not look like one client.

References:
- curl_cffi v0.15.0 (Apr 2026): chrome142/145/146, firefox144/147.
- Cloudflare residential-proxy detection paper, 2026 NDSS.
- ScrapFly: "Accept-Language must match exit-country to avoid blocks".
- W3C UA Client Hints spec: Sec-CH-UA-Full-Version-List.

We intentionally do NOT depend on `browserforge` or `fake-useragent`
(both archived / heavyweight). The Sec-CH-UA payloads below are hand-
authored from the current Chrome / Firefox / Safari release notes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class Identity:
    """One coherent browser identity.

    Fields:
        name: short label for logging.
        impersonate: curl_cffi browser-impersonation target.
        user_agent: full UA string. Pinned even though curl_cffi sets one,
            so we control what the upstream sees if a caller overrides.
        accept_language: locale chain. Drives content negotiation.
        sec_ch_ua: low-entropy Sec-CH-UA header value.
        sec_ch_ua_full_version_list: high-entropy version list. Only
            populated when the impersonated browser sends client hints
            (i.e. Chromium-family).
        sec_ch_ua_platform: "Windows" | "macOS" | "Linux" | "Android".
        sec_ch_ua_mobile: "?0" or "?1".
        sends_client_hints: whether to emit Sec-CH-UA-* at all.
    """
    name: str
    impersonate: str
    user_agent: str
    accept_language: str
    sec_ch_ua: str
    sec_ch_ua_full_version_list: str
    sec_ch_ua_platform: str
    sec_ch_ua_mobile: str = "?0"
    sends_client_hints: bool = True

    def navigation_headers(self) -> dict[str, str]:
        """Headers a real browser sends on a top-level navigation GET.

        Sec-Fetch-* are mandatory in 2026 Chrome navigation requests
        and their absence is itself a fingerprint tell.
        """
        h = {
            "User-Agent": self.user_agent,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8,"
                "application/signed-exchange;v=b3;q=0.7"
            ),
            "Accept-Language": self.accept_language,
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }
        if self.sends_client_hints:
            h["sec-ch-ua"] = self.sec_ch_ua
            h["sec-ch-ua-mobile"] = self.sec_ch_ua_mobile
            h["sec-ch-ua-platform"] = f'"{self.sec_ch_ua_platform}"'
            if self.sec_ch_ua_full_version_list:
                h["sec-ch-ua-full-version-list"] = self.sec_ch_ua_full_version_list
        return h


# A small, curated pool. Each entry pairs an up-to-date impersonate
# target with the exact Sec-CH-UA payload that real browser version
# emits. Adding a Firefox identity dilutes the "everyone is Chrome"
# fleet-level tell.
DEFAULT_POOL: tuple[Identity, ...] = (
    Identity(
        name="chrome146-macos",
        impersonate="chrome146",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/146.0.0.0 Safari/537.36"
        ),
        accept_language="en-US,en;q=0.9",
        sec_ch_ua='"Google Chrome";v="146", "Not.A/Brand";v="99", "Chromium";v="146"',
        sec_ch_ua_full_version_list=(
            '"Google Chrome";v="146.0.7390.55", "Not.A/Brand";v="99.0.0.0", '
            '"Chromium";v="146.0.7390.55"'
        ),
        sec_ch_ua_platform="macOS",
    ),
    Identity(
        name="chrome145-windows",
        impersonate="chrome145",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        accept_language="en-US,en;q=0.9",
        sec_ch_ua='"Chromium";v="145", "Not.A/Brand";v="24", "Google Chrome";v="145"',
        sec_ch_ua_full_version_list=(
            '"Chromium";v="145.0.7339.81", "Not.A/Brand";v="24.0.0.0", '
            '"Google Chrome";v="145.0.7339.81"'
        ),
        sec_ch_ua_platform="Windows",
    ),
    Identity(
        name="chrome142-linux",
        impersonate="chrome142",
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/142.0.0.0 Safari/537.36"
        ),
        accept_language="en-US,en;q=0.9",
        sec_ch_ua='"Not_A Brand";v="8", "Chromium";v="142", "Google Chrome";v="142"',
        sec_ch_ua_full_version_list=(
            '"Not_A Brand";v="8.0.0.0", "Chromium";v="142.0.7204.49", '
            '"Google Chrome";v="142.0.7204.49"'
        ),
        sec_ch_ua_platform="Linux",
    ),
    Identity(
        name="firefox147-windows",
        impersonate="firefox147",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:147.0) "
            "Gecko/20100101 Firefox/147.0"
        ),
        accept_language="en-US,en;q=0.5",
        sec_ch_ua="",
        sec_ch_ua_full_version_list="",
        sec_ch_ua_platform="",
        sends_client_hints=False,  # Firefox does not emit client hints
    ),
)


def pick_for_host(host: str, pool: tuple[Identity, ...] = DEFAULT_POOL) -> Identity:
    """Pick a stable identity for the given host.

    Deterministic so cookies and ETags survive across requests to the
    same host. The hash is SHA256(host) interpreted as an int; we mod
    by the pool size. This spreads our hosts across the pool without
    runtime randomness (important for resumable workflows).
    """
    digest = hashlib.sha256(host.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(pool)
    return pool[idx]


def pick_for_url(url: str, pool: tuple[Identity, ...] = DEFAULT_POOL) -> Identity:
    host = urlparse(url).netloc or url
    return pick_for_host(host, pool)
