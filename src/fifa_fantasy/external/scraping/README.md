# `fifa_fantasy.external.scraping` - Reusable stealth scraping module

A small, focused Python HTTP client with TLS-fingerprint impersonation,
per-host rate limiting, on-disk caching, retries, and optional proxy
rotation. Designed to be **reusable across projects**: no domain-specific
code, no FIFA-fantasy dependencies, just a clean public API.

## Why this exists

Modern websites use a four-layer bot-detection stack:

1. **TLS fingerprinting (JA3/JA4)** - identifies the TLS client BEFORE
   the HTTP request reaches the application. Python's `requests` library
   has a trivially-identifiable fingerprint.
2. **Browser fingerprinting** - Canvas, WebGL, navigator props.
3. **Behavioral analysis** - mouse movement, click timing, scroll.
4. **IP reputation** - datacenter vs residential vs known-bot pools.

This module addresses **layer 1** (TLS) via `curl_cffi` browser
impersonation, and gives you knobs for **layer 4** (proxy rotation).
**Layers 2 and 3 require a real browser**; for those, escalate to
`playwright-stealth` or `nodriver` (see Escalation below).

## Quick start

```python
from fifa_fantasy.external.scraping import StealthClient

client = StealthClient(
    impersonate="chrome124",          # JA3/JA4 match real Chrome
    rate_limit_per_second=1.0,        # ethical pacing
    cache_dir="data/cache",
    cache_ttl_hours=6,
    max_retries=3,
    warm_session=False,               # set True to visit homepage first
)

response = client.get("https://example.com")
print(response.status_code, response.from_cache, response.text[:200])
```

## Configuration

| Argument | Default | Purpose |
|---|---|---|
| `impersonate` | `"chrome124"` | Which browser's TLS fingerprint to mimic. Options: `chrome120`, `chrome124`, `firefox120`, `safari17`, etc. (See `curl_cffi` docs for the full list.) |
| `rate_limit_per_second` | `1.0` | Max requests/sec per host. 0.5 = 1 req per 2s. |
| `cache_dir` | `None` | If set, GET responses are cached on disk. Set to `None` to disable. |
| `cache_ttl_hours` | `6.0` | Entries older than this are treated as missing. |
| `max_retries` | `3` | Total attempts (1 = no retry) on retryable errors. |
| `proxy_rotator` | env var | Pluggable proxy pool. Defaults to reading `SCRAPING_PROXY_URL`. |
| `default_timeout_s` | `30.0` | Per-request timeout. |
| `warm_session` | `False` | If True, visit `https://<host>/` once to collect cookies before deep URLs. |

## When this module is enough vs when to escalate

The defenses each layer:

| Site type | Tool sufficient? |
|---|---|
| Open public APIs (JSON, no cookies) | Plain `httpx` or stdlib; you don't need this module |
| TLS-fingerprinted sites (most modern news, ESPN, BBC, Reddit) | yes **This module (`StealthClient`)** |
| JS-challenged sites (Cloudflare Turnstile, DataDome interactive) | no Escalate to `playwright-stealth` |
| Heavy fingerprinting (Akamai Bot Manager, PerimeterX advanced mode) | no Escalate to `nodriver` or paid services |
| Behind-login content | This module supports cookies; escalate to playwright if login requires JS |

### Escalation path

When `StealthClient` returns 403 / Cloudflare interstitial / blank
response repeatedly, escalate in this order:

1. **Add proxy rotation**: set `SCRAPING_PROXY_URL` to a residential
   proxy. This addresses IP-based blocking; TLS-fingerprint and JS
   challenges remain.
2. **Switch to `playwright-stealth`**: Python package that runs a real
   Chromium with anti-detection patches. Slower but defeats most
   browser-fingerprinting checks.
3. **Switch to `nodriver`**: CDP-based Chrome control without
   WebDriver. Currently the strongest open-source option for
   Cloudflare-protected sites (per the 2026 anti-detect benchmark
   referenced in the whitepaper).
4. **Paid services**: ScrapFly, BrightData Web Unlocker. Last resort.

Each step is more expensive (CPU, latency, monetary). Use the simplest
tier that works for your target.

## Ethics

See `ethics.md`. Highlights:

- **Default rate limit is 1 req/s per host**. Do not override without
  good reason and permission.
- **Respect `robots.txt`** for crawlers. We do not auto-enforce; you
  must check it yourself.
- **Do not scrape behind-login content** without permission.
- **Identify yourself**. Even when impersonating a browser, when in
  doubt, set a custom `User-Agent` that lets the site contact you.
- **Don't republish copyrighted content**. Scrape for analysis and
  derived data, not to mirror.

## Reusability across projects

This module has zero dependencies on the rest of `fifa_fantasy/`. To
reuse it in another project:

```bash
# Option 1: copy the directory.
cp -r src/fifa_fantasy/external/scraping/ /path/to/new_project/src/scraping/

# Option 2: install as a git submodule (or vendored package).
git submodule add ...
```

Then `from scraping import StealthClient` works the same way.

Two demonstrations of out-of-project use:

1. `docs/scraping/example_other_project.py` - minimal 30-line script
   that fetches the Hacker News front page using just `StealthClient`.
2. `docs/scraping/startup_research_demo.py` - configures the `news`
   collector with custom feeds (TechCrunch, Crunchbase, The Information,
   VentureBeat, Hacker News) and keyword filters for startup / VC
   research. Demonstrates that the `news` package is reusable outside
   the FIFA fantasy domain.

## Architecture

```
StealthClient.get(url, headers)
  │
  ├─► DiskCache.get(method, url, headers)
  │      └─► hit? return CachedResponse(from_cache=True)
  │
  ├─► (optional) warm session by visiting homepage
  │
  ├─► retry_with_backoff(
  │      attempts=max_retries,
  │      retry_on=429/5xx + network errors
  │   ):
  │      └─► PerHostRateLimiter.acquire(url)
  │             └─► (sleep if needed)
  │      └─► curl_cffi.Session.get(url, impersonate=chrome124, proxies=...)
  │
  └─► DiskCache.put(...)   # cache 2xx responses
```

## Sources

The design follows current best practices for 2026 web scraping:

- [ScrapFly: 11 Best Anti-Bot Bypass Tools](https://scrapfly.io/blog/posts/best-anti-bot-bypass-tools)
- [BrightData: Web Scraping with curl_cffi](https://brightdata.com/blog/web-data/web-scraping-with-curl-cffi)
- [Cloudflare JA3/JA4 fingerprinting docs](https://developers.cloudflare.com/bots/additional-configurations/ja3-ja4-fingerprint/)
- [Anti-detect browser benchmark 2026](https://ianlpaterson.com/blog/anti-detect-browser-benchmark-patchright-nodriver-curl-cffi/)
