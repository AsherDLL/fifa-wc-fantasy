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

## 11e.10b The RSS-first news collector (lean architecture)

After the initial scraper-and-parser architecture was built (above), the
realisation that scraping HTML index pages would blow up disk usage led
to a second, leaner collector: `src/fifa_fantasy/external/news/`. The
key insight: RSS feeds are XML (5-50KB per fetch), already structured,
and produced by every major football news site. Polling them is an
order of magnitude cheaper than crawling HTML pages.

The architecture:

```
news.collector.collect(client, feeds, out_dir, budget_mb)
  for each feed in feeds:
    fetch feed RSS XML (cheap, 5-50KB)
    parse items: title, url, published, summary
    filter items by keyword match against WC + team names
    for each matching item:
      fetch full article HTML (one per article, capped at MAX_BODY_CHARS=8000)
      extract body text via news.extractor.extract()
      append row to today's parquet (snappy-compressed)
    dedup by URL within 12-hour window
  after writing: enforce disk budget (default 200MB cap)
    if over budget, delete oldest day's parquet until under cap
```

Six free feeds are curated in `news/feeds.py`:
- BBC Sport Football (`feeds.bbci.co.uk/sport/football/rss.xml`)
- BBC Sport World Football
- Sky Sports Football
- ESPN Soccer
- Guardian Football
- Goal.com Latest

Twitter/X is **deliberately excluded**. As of 2026, Nitter's public
instance network has collapsed (X blocked their guest-account tokens);
twscrape requires authenticated accounts which the project does not
have; the legal/technical cost of pursuing X data is not worth the
marginal lineup-signal it provides over the RSS feeds above. This is
documented in `news/feeds.py`'s module docstring.

### 11e.10b.2 ESPN JSON API bypass

ESPN's RSS endpoints are protected by AWS WAF (return 202 with a
JavaScript challenge body that requires browser JS execution to
solve). Defeating that would mean escalating to playwright-stealth
or nodriver (escalation tier 2 per scraping/README.md).

We discovered a free, public alternative: ESPN's JSON API at
`site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/news` is
not WAF-protected. It returns ~6 articles per call with title, URL,
summary, and publication timestamp.

The catch: ESPN's article PAGES are also WAF-protected. We can get
the article *list* via JSON but cannot fetch the full body. The
collector handles this by treating JSON-feed summaries as
sufficient body text when the format is `"json_espn"`: ~80-200 chars
per article is short but informative for team-news context (the
summary often mentions the matchup, key players, and tactical
storyline).

This pattern (use the JSON-feed summary, skip the article page)
generalises: any source that exposes a JSON list with reasonable
summaries can be added with `format="json_espn"` even if their
article pages are heavily protected.

### 11e.10b.3 Substacks and tactical blogs

Through a probe of 14 candidate sources (tactical analysis newsletters,
mainstream football blogs, fan-oriented Substacks), three high-quality
free RSS feeds were added:

- **Bob Sturm's World Cup Journal** (`bobsturm.substack.com/feed`):
  15/30 top items match WC keywords - the highest WC density of any
  free source tested. Match-by-match journal of WC 2026.
- **Daily Mail Football** (`dailymail.co.uk/sport/football/index.rss`):
  150 items/pull, 15/30 WC density. High volume + high WC content
  (tournament is in the host nations' news cycle).
- **Gegenpressing** (`gegenpressing.substack.com/feed`): 4/30 WC density
  but high-quality tactical analysis on the items that do surface.

Sources tested but not added:
- The Athletic (paywall, 404 on RSS)
- The Times (empty RSS payload, paywall)
- Cultured Football (DNS error)
- Eurosport (404)
- Football.London (403)
- Onefootball (404)
- Mirror Football (403)

Three RSS sources rotated in over the WC 2026 collection window have
produced a stable 60+ articles per collection tick with zero errors
on the working feeds.

### 11e.10b.4 Reddit via arctic-shift (the .json endpoint is dead)

Reddit deprecated the unauthenticated `<url>.json` suffix on
2026-05-30. Subsequent probes also confirmed:

- `www.reddit.com/r/<sub>/new/.rss` returns 403 to datacenter IPs
  even with realistic User-Agent and curl_cffi Chrome 146 impersonation
- `old.reddit.com` mirrors return 403 (same WAF tier)
- Subreddit-specific `/new/.rss` endpoints return 429 with empty body

The remaining free path is `arctic-shift`
([github.com/ArthurHeitmann/arctic_shift](https://github.com/ArthurHeitmann/arctic_shift)),
a research-grade Reddit mirror exposing:

- `arctic-shift.photon-reddit.com/api/posts/search?subreddit=<sub>&limit=...`
- `arctic-shift.photon-reddit.com/api/comments/search?...`

No authentication, ~2000 req/min soft cap, full post JSON including
`selftext`. Empirically returns 10KB+ of typed data per call. Lags
Reddit by hours to days, acceptable for our team-news context.

The collector exposes this as `format="json_arctic_shift"` and adds
two feeds: `r/soccer` and `r/worldcup`. The arctic-shift parser maps
`permalink -> canonical reddit URL`, `created_utc -> ISO timestamp`,
`selftext or title -> body`. Reddit short posts (e.g. match-thread
headers) are persisted with title-only bodies; long selftexts (tactical
posts, predicted-XI threads) are stored verbatim up to 12000 chars.

An alternative path is registering a Reddit script OAuth app and
using `asyncpraw`. This requires the operator to create a throwaway
Reddit account and provision `REDDIT_CLIENT_ID/SECRET/USERNAME/PASSWORD`
env vars. Not implemented because arctic-shift gives us what we need
without managing OAuth refresh tokens; documented here as the
escalation when arctic-shift coverage proves insufficient.

### 11e.10b.5 Fingerprint upgrades to the StealthClient (2026 sweep)

The original `StealthClient` (Section 11e.3) pinned every host to
`impersonate="chrome124"`. By mid-2026 that has three problems:

1. `chrome124` is on every public scraping tutorial. It is a known
   fingerprint default in Cloudflare Bot Management's defensive
   corpus (per ScrapFly's June 2026 review).
2. curl_cffi v0.15.0 added `chrome142/145/146`, `firefox144/147`, and
   `safari260`. Real-Chrome users are mostly on those builds; staying
   on 124 produces a JA4_R that no real user emits.
3. Single-identity pinning across all hosts means our fleet is one
   client. A site with multi-host telemetry sees one "user" hitting
   ESPN, BBC, Reddit, and Substacks simultaneously, which is itself a
   bot signal.

The upgraded `StealthClient` (`src/fifa_fantasy/external/scraping/`):

- **Identity pool** (`identity.py`): four curated identities
  (`chrome146`/macOS, `chrome145`/Windows, `chrome142`/Linux,
  `firefox147`/Windows), each with a coherent UA + Sec-CH-UA +
  Sec-CH-UA-Full-Version-List + Accept-Language + Sec-Fetch-*
  navigation header set, matching the W3C UA Client Hints spec.
- **Per-host sticky binding**: hash the hostname, pick an identity
  from the pool deterministically. Same host always uses the same
  identity (so cookies and ETags survive); different hosts use
  different ones.
- **Block-aware retry** (`retry.py`): inspect both response headers
  (`x-amzn-waf-action: challenge`, `cf-mitigated: challenge`,
  `x-dd-b`, `x-datadome-cid`) and the first 16KB of the body for
  WAF challenge signatures (Cloudflare "Just a moment...",
  AWS WAF integration JS, DataDome captcha redirect, Akamai
  `_abck` cookie page, Imperva incident ID). When a block is
  detected the client rotates to the next identity in the pool
  and retries.

Empirical improvement (one collection tick, 11 feeds): items_stored
**68 -> 100**, errors **1 -> 0**. The previously-flaky FourFourTwo
feed (intermittent 403) stopped firing 403s after the upgrade,
suggesting at least one host was scoring our previous fingerprint.

What this does NOT defeat:

- Cloudflare Turnstile interactive challenges: no pure-Python
  solver exists in 2026. Escalation tier 2 (`nodriver` or
  `patchright`) is required.
- AWS WAF JavaScript proof-of-work challenges (ESPN article
  pages). We work around this by fetching the ESPN JSON API
  (11e.10b.2) which is not WAF-protected, rather than solving
  the PoW. `xKiian/awswaf` exists as a pure-Python PoW reimpl
  but bit-rots fast; out of scope.
- IP reputation when scraping from a datacenter range (as we are).
  Residential proxies via `SCRAPING_PROXY_URL` are supported but
  cost money; not enabled by default.

ALPS TLS extension drift (curl_cffi emits codepoint 17513 where
Chrome 133+ emits 17613) is documented but not patched, because it
only matters for sites cross-checking JA4_c entropy; we have no
empirical evidence any of our current feeds do so.

### Disk-usage budget

The collector enforces an upper bound on `data/external/news_articles/`
via `prune_disk_budget()`. Default 200MB; per-article cap 8KB raw text
(~1-3KB compressed in parquet). Empirical footprint estimate for the
remaining tournament:

- 6 feeds × ~20 items/feed × ~15% keyword-match rate × 3KB per stored
  article × ~6h cadence = **~50KB per tick, ~200KB per day**.
- Over the remaining ~3 weeks of tournament: **~4-6MB total**. Well
  under the 200MB cap.

The cap exists as a safety: if a parser flaw or feed-flood causes
runaway growth, the oldest data is dropped automatically rather than
the disk filling. This is more defensive than tuned.

### Tests

`tests/test_news_collector.py` (10 tests): RSS parsing, keyword
matching, body extraction, dedup-by-URL, day-file rotation, disk
budget enforcement, and end-to-end with a mocked StealthClient. All
pass.

### Operational status

The collector is built and tested but not yet run live in the
production daemon. A first live run will:
1. Verify the listed RSS endpoints are reachable
2. Confirm keyword-match rate is ~10-20% (per the estimate)
3. Validate disk-usage tracking against the budget

After that, the Docker daemon picks it up automatically via the
updated `news_tick()` (every 6h by default).

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

Industry / blog references:

- [ScrapFly: 11 Best Anti-Bot Bypass Tools for Web Scraping in 2026](https://scrapfly.io/blog/posts/best-anti-bot-bypass-tools)
- [ScrapFly: How to Bypass Anti-Bot Protection (2026)](https://scrapfly.io/blog/posts/how-to-bypass-anti-bot-protection)
- [AlterLab: Playwright Anti-Bot Detection: What Works (2026)](https://alterlab.io/blog/playwright-bot-detection-what-actually-works-in-2026)
- [ScrapFly: How to Bypass Cloudflare When Web Scraping in 2026](https://scrapfly.io/blog/posts/how-to-bypass-cloudflare-anti-scraping)
- [Ian L. Paterson: Anti-detect browser benchmark 2026 (Patchright, NoDriver, curl_cffi)](https://ianlpaterson.com/blog/anti-detect-browser-benchmark-patchright-nodriver-curl-cffi/)
- [Cloudflare developer docs: JA3/JA4 fingerprint](https://developers.cloudflare.com/bots/additional-configurations/ja3-ja4-fingerprint/)
- [Cloudflare Engineering: ML-based residential-proxy bot detection](https://blog.cloudflare.com/residential-proxy-bot-detection-using-machine-learning/)
- [John Althouse (FoxIO): JA4+ Network Fingerprinting](https://medium.com/foxio/ja4-network-fingerprinting-9376fe9ca637)
- [BrightData: Web Scraping With curl_cffi and Python in 2026](https://brightdata.com/blog/web-data/web-scraping-with-curl-cffi)
- [probberechts/soccerdata GitHub: Football data scraper](https://github.com/probberechts/soccerdata)
- [lexiforest/curl_cffi GitHub (Chrome 142/145/146 + HTTP/3 support, April 2026)](https://github.com/lexiforest/curl_cffi)
- [ultrafunkamsterdam/nodriver (CDP Chrome, Cloudflare Turnstile click)](https://github.com/ultrafunkamsterdam/nodriver)
- [Kaliiiiiiiiii-Vinyzu/patchright (Playwright drop-in with anti-detect patches)](https://github.com/Kaliiiiiiiiii-Vinyzu/patchright)
- [ArthurHeitmann/arctic_shift (Pushshift-class Reddit mirror, no auth)](https://github.com/ArthurHeitmann/arctic_shift)
- [praw-dev/asyncpraw (async Reddit OAuth client, 2026)](https://github.com/praw-dev/asyncpraw)
- [xKiian/awswaf (pure-Python AWS WAF PoW solver)](https://github.com/xKiian/awswaf)
- [riboseinc/country_to_locales_mapping (ISO 3166 -> Accept-Language)](https://github.com/riboseinc/country_to_locales_mapping)
- [API-Football](https://www.api-football.com/) and [Highlightly Football API](https://highlightly.net/football-api/) for paid-tier alternatives

Peer-reviewed academic references:

- Annamalai, M. et al. (2025). **Beyond the Crawl: Unmasking Browser
  Fingerprinting in Real User Interactions.** Proceedings of the
  ACM Web Conference (WWW). arXiv:2502.01608. Shows that automated
  crawlers miss 45% of fingerprinting sites real users encounter,
  motivating our identity-rotation pool.
- Bacis, E., Bilogrevic, I. et al. (2024). **Assessing Web
  Fingerprinting Risk.** WWW Companion. arXiv:2403.15607. First
  large-scale entropy estimate from tens of millions of real Chrome
  browsers in the wild.
- Xue, D., Stanley, M., Kumar, R., Ensafi, R. (2025). **The
  Discriminative Power of Cross-Layer RTTs in Fingerprinting Proxy
  Traffic.** NDSS Symposium. 92-98% accuracy fingerprinting
  Shadowsocks / OpenVPN traffic at <1.5% FPR. Bears on our proxy
  rotation discussion.
- (2026). **Beyond RTT: An Adversarially Robust Two-Tiered Approach
  for Residential Proxy Detection.** NDSS Symposium 2026.
  Demonstrates that RTT-based detection breaks under trivial
  adversarial padding; informs our decision not to rely on a single
  proxy provider.
- Teoh, M. et al. (2025). **Are CAPTCHAs Still Bot-hard? Generalized
  Visual CAPTCHA Solving with Agentic Vision-Language Model.**
  USENIX Security. 70.6% solve rate on in-the-wild CAPTCHAs;
  motivates escalation to VLM-class solvers if CAPTCHA-protected
  sources become necessary in future work.
- Searles, A. et al. (2024). **The Matter of CAPTCHAs.** WWW
  Symposium. 15,000 image-CAPTCHAs x 20 schemes against 14
  open-source solvers.
