# 11e — Team-news ingestion: reusable stealth scraping + predicted-XI signal

Status: **DRAFT** (scaffold shipped; empirical impact measured as data accumulates)

This section documents the team-news ingestion pipeline that addresses
the gap identified in Section 03c.6: expert practitioners read manager
press conferences ~24 hours before kickoff to detect rotations,
suspensions, and form-driven lineup changes, but our prior pipeline
had no such input.

## 11e.1 The architectural split

The implementation deliberately separates two concerns into two
packages:

1. **`src/fifa_fantasy/external/scraping/`** — a **reusable** stealth
   HTTP client module. No domain-specific code. The user can extract
   this directory and drop it into an unrelated project. The
   `README.md` and `ethics.md` are written for an external audience.
2. **`src/fifa_fantasy/external/team_news/`** — the FIFA fantasy
   application that consumes the scraping module. Contains the
   per-source parsers (ESPN, Sportsgambler, soccerdata proxy),
   name-matching logic against the FIFA player_id catalog, and the
   per-fixture predicted-XI persistence layer.

The split is the contribution. The user can:
- Take the `scraping/` directory verbatim to another scraping project
- Replace the `team_news/parsers/` with parsers for other domains
- Keep the `scraping/` core stable while the application evolves

## 11e.2 Anti-bot detection landscape (2026 snapshot)

Cited from sources surveyed during the implementation: ScrapFly,
AlterLab, BrightData, Cloudflare developer docs, Ian L. Paterson's
2026 anti-detect benchmark.

Modern bot detection operates in four layers:

| Layer | What it checks | Bypass technique |
|---|---|---|
| 1. TLS fingerprinting (JA3/JA4) | TLS ClientHello signature | `curl_cffi` (browser impersonation) |
| 2. Browser fingerprinting | Canvas, WebGL, navigator.webdriver | `playwright-stealth`, `nodriver` |
| 3. Behavioral analysis | Mouse, click timing, scroll | Headless browsers with human-pattern emulation |
| 4. IP reputation | Datacenter vs residential | Residential proxy rotation |

JA3 (Salesforce, 2017) fingerprints the TLS ClientHello as an MD5
hash. JA4 (FoxIO, 2023) is the successor: multi-part, survives
Chrome 110's extension-order randomization. Cloudflare uses JA4 by
default in 2026. Python's `requests` library produces a fingerprint
that does not match any real browser; `curl_cffi` produces one that
does.

The benchmark by Ian L. Paterson (2026) tested seven anti-detection
tools against 31 Cloudflare-protected target sites with 651 verdicts.
Top results:
- **nodriver**: zero blocked targets (best open-source result)
- **curl_cffi**: comparable to Chromium forks with 49 source-level
  patches, in a 21-line Python wrapper
- **playwright-stealth**: most of the targets passed, some still
  blocked on advanced sites

For our team-news targets (ESPN, BBC, Sportsgambler, soccerdata),
`curl_cffi` is sufficient. We document escalation to
`playwright-stealth` and `nodriver` in `scraping/README.md` for
future sources that need more.

## 11e.3 The `StealthClient` design

Single public class. Sync interface (thread-safe). Five integrated
components:

```python
from fifa_fantasy.external.scraping import StealthClient

client = StealthClient(
    impersonate="chrome124",
    rate_limit_per_second=1.0,
    cache_dir="data/external/cache/scraping",
    cache_ttl_hours=6,
    max_retries=3,
)
response = client.get("https://www.espn.com/soccer/...")
# response.text, response.status_code, response.from_cache (bool)
```

Internal flow on `get()`:

```
get(url, headers)
  ├─ DiskCache.get() → hit? return CachedResponse(from_cache=True)
  ├─ (optional) warm session by visiting homepage
  ├─ retry_with_backoff(max_attempts=3, retry on 429/5xx + network errors):
  │   ├─ PerHostRateLimiter.acquire(url) → (sleep if needed)
  │   └─ curl_cffi.Session.get(impersonate=chrome124, proxies=...)
  └─ DiskCache.put() → cache 2xx responses
```

Components implemented:

- `client.py` — public class
- `rate_limit.py` — per-host token bucket, thread-safe
- `cache.py` — disk-backed with SHA256 key, JSON metadata, TTL
- `retry.py` — exponential backoff with ±25% jitter
- `session.py` — optional homepage warm-up
- `proxies.py` — env-var-driven proxy rotator (random or round-robin)

## 11e.4 The team-news application

The application layer (`src/fifa_fantasy/external/team_news/`)
consumes the scraping module to fetch predicted XIs from:

1. **`soccerdata` PyPI library** (primary): a community-maintained
   package that scrapes ESPN, FBref, Sofascore, WhoScored, and
   Understat. Confidence 0.75. Currently scaffolded; full
   integration deferred until soccerdata's WC 2026 coverage is
   confirmed.
2. **ESPN preview articles** (fallback): paragraph-level extraction
   from pre-match preview pages. Requires the caller to provide
   article URLs (we don't crawl ESPN to discover them). Confidence
   0.65.
3. **Sportsgambler** (scaffold; CSS selectors pending tuning).
   Confidence 0.50.

Each parser returns a `RawLineup` (Pydantic model: source-as-scraped
player names, no FIFA ids). The `PlayerNameMatcher` resolves names
to FIFA player_ids using a four-tier confidence system (exact full
name 1.00, known name 0.95, last name in country+position 0.85,
last name in country 0.70, last name only 0.50).

The resolved `PredictedXI` records persist as
`data/external/team_news/team_news_<utc-iso>.parquet` with one row
per (fixture, player, status) tuple.

## 11e.5 Feature pipeline integration

`features/build.py` gets a new `_attach_team_news(grid, news_table)`
join function. When news is available, the per-(player, round)
feature grid gains two columns:

- `predicted_starting_xi`: bool — True (confirmed start), False
  (confirmed bench), NaN (no news)
- `xi_confidence`: float in [0, 1] — per-source reliability

NaN-safe: when no news is available for a player, the columns are
NaN and downstream models keep current behaviour.

## 11e.6 Model integration

**Heuristic backend** (`baseline.py`): the availability check gains
a clause. A player marked `predicted_starting_xi == False` is zeroed
exactly like a `transferred` or `eliminated` player. NaN means
unknown and keeps the player available.

**Monte Carlo backend** (`monte_carlo.py`): per-simulation per-player
points are scaled by `xi_confidence` when the player is marked
`starting`. This dampens the contribution of low-confidence
predictions without zeroing them out.

**GBM backend**: deferred. Adding `xi_confidence` as a feature
requires retraining on EPL data that has lineup info attached, which
the current training pipeline does not have. Noted in Section 11.

## 11e.7 Validation

Following the discipline established in Section 5c.6:
- Held-out RMSE on EPL 2024-25 GW 30-38 with team-news = empty: must
  not regress vs the baseline (it can't: empty news means no-op
  fallback).
- Live WC backtest with team-news enabled: must not regress; aim to
  improve.

Initial backtest result (cumulative MD1-MD3, before team-news data
exists):

| Backend | Pre-news | Post-wiring (no data yet) | Δ |
|---|---|---|---|
| Monte Carlo | 212 | 212 | 0 |
| Heuristic v1 | 211 | 211 | 0 |
| GBM | 161 | 161 | 0 |
| Poisson | 72 | 72 | 0 |

No regression. The team-news data path is in place but no scraped
data has flowed through it yet (soccerdata WC coverage is
scaffolded; ESPN parser needs seed URLs). When data starts flowing,
the backtest will re-run and the empirical impact is reported here.

## 11e.8 Docker daemon integration

`docker/snapshot_loop.py` gains a fourth scheduled tick: `news_tick`
runs every 6 hours (configurable via `NEWS_INTERVAL_HOURS`). Calls
the team_news CLI with a 3-day fixture window. Data persists to the
mounted volume; the next FIFA tick's features build picks it up
automatically.

## 11e.9 Reusability demonstration

`docs/scraping/example_other_project.py` is a 30-line script that
uses the `scraping` module to fetch the Hacker News front page and
print the top 10 stories. It runs as a stand-alone script with no
FIFA-fantasy context.

```
$ .venv/bin/python docs/scraping/example_other_project.py
Top 10 HN stories (fresh):
  1. GLM 5.2 beats Claude in our benchmarks
  2. HackerRank open sourced its ATS...
  ...
```

The demo proves the architectural split is real: the `scraping/`
package functions outside the FIFA fantasy domain. A user can copy
the directory to a new project and the example works unchanged.

## 11e.10 Limitations and follow-up work

- **Coverage**: ESPN/Sportsgambler/soccerdata don't all have predicted
  XIs for every WC 2026 fixture, especially for matches between
  smaller nations. Expected hit rate ~60-80% for top-tier matches,
  lower for cross-region knockouts.
- **Name matching false negatives**: African and South American
  players with multiple given names + nicknames (e.g. "Júnior",
  "Filho") may not match cleanly. Unmatched names are logged for
  manual review.
- **Confidence-decay over time**: A 24-hour-old predicted XI is more
  reliable than a 6-hour-old one (closer to kickoff). We store
  `scraped_at_utc` and let consumers compute decay if desired.
- **Anti-bot escalation**: when ESPN/BBC start blocking us via JS
  challenges (they don't currently, but may), escalate to
  `playwright-stealth`. Documented in scraping/README.md.

## 11e.11 Sources for the bibliography

- [ScrapFly: 11 Best Anti-Bot Bypass Tools for Web Scraping in 2026](https://scrapfly.io/blog/posts/best-anti-bot-bypass-tools)
- [AlterLab: Playwright Anti-Bot Detection: What Works (2026)](https://alterlab.io/blog/playwright-bot-detection-what-actually-works-in-2026)
- [ScrapFly: How to Bypass Cloudflare When Web Scraping in 2026](https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping)
- [Ian L. Paterson: Anti-detect browser benchmark 2026 (Patchright, NoDriver, curl_cffi)](https://ianlpaterson.com/blog/anti-detect-browser-benchmark-patchright-nodriver-curl-cffi/)
- [Cloudflare developer docs: JA3/JA4 fingerprint](https://developers.cloudflare.com/bots/additional-configurations/ja3-ja4-fingerprint/)
- [BrightData: Web Scraping With curl_cffi and Python in 2026](https://brightdata.com/blog/web-data/web-scraping-with-curl-cffi)
- [probberechts/soccerdata GitHub: Football data scraper](https://github.com/probberechts/soccerdata)
- [API-Football](https://www.api-football.com/) and [Highlightly Football API](https://highlightly.net/football-api/) for paid-tier alternatives
