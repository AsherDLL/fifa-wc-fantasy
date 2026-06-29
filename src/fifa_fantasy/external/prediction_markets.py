"""Polymarket and Kalshi WC 2026 contract scraper.

Implements the data side of the Section 11b innovation roadmap: ingest
prediction-market contract prices, derive implied probabilities, persist
hourly snapshots for the Benter-combiner training data.

Polymarket:
- REST endpoint: https://clob.polymarket.com/markets
- Per-market: https://clob.polymarket.com/markets/<condition_id>
- Polymarket lists binary YES/NO contracts. The YES contract's last
  trade price (in dollars) is the implied probability of the event.

Kalshi:
- REST endpoint: https://trading-api.kalshi.com/trade-api/v2/markets
- Per-market detail: same path + /<ticker>
- Kalshi contracts settle at $1 (YES) or $0 (NO). The YES bid/ask
  midpoint is the implied probability.

Contracts we want for WC 2026:
1. Match outcome (home win / draw / away win) for each fixture
2. Total goals over/under contracts per fixture
3. Top-scorer-of-tournament (one per elite forward)
4. Country-to-reach-stage contracts (per country, per knockout round)

Output:
- data/external/prediction_markets/<provider>_<utc_iso>.jsonl
  One line per contract per snapshot. Provider field for source.
- data/external/prediction_markets/index.parquet
  Rolled-up time series suitable for joining into the feature table.

Notes on rate limits and authentication:
- Polymarket REST endpoints under /markets are public, no auth needed
- Kalshi requires an API key for full access; the public endpoints
  expose a subset. We document both; user can plug in a Kalshi API
  key via env var if available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

DEFAULT_DIR = Path("data/external/prediction_markets")

POLYMARKET_BASE = "https://clob.polymarket.com"
POLYMARKET_GAMMA = "https://gamma-api.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"  # public path

# Search tags that should surface WC 2026 contracts.
POLYMARKET_WC_TAGS = ("world cup 2026", "fifa world cup 2026", "wc 2026", "fifa world cup")
KALSHI_WC_SERIES = ("WORLDCUP", "FIFAWC", "WC2026")

# Known Polymarket event ids for WC 2026. Tag-based search misses them;
# fetching the event directly is the reliable path. Append new ids as
# they appear (e.g. per-fixture markets when posted).
POLYMARKET_WC_EVENT_IDS = (
    "30615",  # World Cup Winner (60 markets, one per country)
)


@dataclass
class MarketSnapshot:
    """One observation of one contract at one timestamp."""
    provider: str
    snapshot_utc: str
    contract_id: str
    title: str
    yes_price: float | None      # implied probability of YES outcome
    no_price: float | None
    volume_24h: float | None
    metadata: dict


def fetch_polymarket_wc_markets(client: httpx.Client) -> list[dict]:
    """Pull WC 2026 markets from Polymarket's gamma API.

    Two-pronged strategy:

    1. Fetch the known WC event ids directly (POLYMARKET_WC_EVENT_IDS).
       Each event has a `markets` array; flatten and return.
    2. Page sports-tagged events as a discovery layer for new WC events
       (per-fixture markets posted close to match time).
    """
    out: list[dict] = []
    # Direct event id lookups (reliable).
    for eid in POLYMARKET_WC_EVENT_IDS:
        try:
            r = client.get(f"{POLYMARKET_GAMMA}/events/{eid}")
            r.raise_for_status()
            ev = r.json()
            for m in (ev.get("markets") or []):
                m["_event_title"] = ev.get("title")
                m["_event_id"] = ev.get("id")
                out.append(m)
        except (httpx.HTTPError, ValueError):
            continue
    # Discovery: page sports events for any new WC event we don't know about.
    for offset in (0, 100, 200):
        params = {"limit": 100, "active": "true", "closed": "false",
                  "tag": "sports", "offset": offset}
        try:
            r = client.get(f"{POLYMARKET_GAMMA}/events", params=params)
            r.raise_for_status()
            events = r.json()
        except (httpx.HTTPError, ValueError):
            continue
        if not isinstance(events, list):
            continue
        for ev in events:
            title = (ev.get("title") or "").lower()
            desc = (ev.get("description") or "").lower()
            if any(t in title or t in desc for t in POLYMARKET_WC_TAGS):
                for m in (ev.get("markets") or []):
                    m["_event_title"] = ev.get("title")
                    m["_event_id"] = ev.get("id")
                    out.append(m)
        if len(events) < 100:
            break
    # Dedupe by conditionId.
    seen, deduped = set(), []
    for m in out:
        cid = m.get("conditionId") or m.get("id") or m.get("question_id")
        if cid and cid not in seen:
            seen.add(cid)
            deduped.append(m)
    return deduped


def fetch_kalshi_wc_markets(client: httpx.Client,
                            include_unopened: bool = False) -> list[dict]:
    """Pull WC 2026 markets from Kalshi's elections API (public endpoint).

    Strategy: list all series, filter to WC-related tickers, then pull
    open markets under each series. Kalshi WC contracts are typically
    `unopened` until match-day (no prices available); we skip those by
    default to keep snapshots small and only include markets with real
    last_price or bid/ask data.
    """
    relevant: list[str] = []
    try:
        r = client.get(f"{KALSHI_BASE}/series", params={"limit": 1000})
        r.raise_for_status()
        ss = r.json().get("series", [])
        for s in ss:
            t = (s.get("ticker") or "").upper()
            title = (s.get("title") or "").lower()
            if any(k in t for k in ("WC", "FIFA")) or "world cup" in title or "fifa" in title:
                relevant.append(s.get("ticker"))
    except (httpx.HTTPError, ValueError):
        pass

    out: list[dict] = []
    statuses = ("open", "unopened") if include_unopened else ("open",)
    for ticker in relevant:
        for status in statuses:
            try:
                r = client.get(
                    f"{KALSHI_BASE}/markets",
                    params={"series_ticker": ticker, "status": status, "limit": 500},
                )
                if r.status_code != 200:
                    continue
                ms = r.json().get("markets", [])
                for m in ms:
                    m["_series_ticker"] = ticker
                out.extend(ms)
            except (httpx.HTTPError, ValueError):
                continue
    return out


def polymarket_to_snapshot(market: dict, now: str) -> MarketSnapshot | None:
    """Translate a Polymarket market record to our MarketSnapshot dataclass.

    Polymarket binary markets expose `outcomePrices` as a JSON-encoded
    list of two strings (YES, NO prices in USD between 0 and 1).
    """
    cid = market.get("conditionId") or market.get("id") or ""
    title = market.get("question") or market.get("title") or ""
    yes_price = no_price = None
    op = market.get("outcomePrices")
    if isinstance(op, str):
        try:
            import json
            parsed = json.loads(op)
            if isinstance(parsed, list) and len(parsed) >= 2:
                yes_price = float(parsed[0])
                no_price = float(parsed[1])
        except (ValueError, json.JSONDecodeError):
            pass
    elif isinstance(op, list) and len(op) >= 2:
        try:
            yes_price = float(op[0]); no_price = float(op[1])
        except (TypeError, ValueError):
            pass
    vol = market.get("volume24hr") or market.get("volume24Hr")
    try:
        vol = float(vol) if vol is not None else None
    except (TypeError, ValueError):
        vol = None
    if not cid:
        return None
    return MarketSnapshot(
        provider="polymarket",
        snapshot_utc=now,
        contract_id=str(cid),
        title=title,
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=vol,
        metadata={"slug": market.get("slug"), "end_date": market.get("endDate")},
    )


def kalshi_to_snapshot(market: dict, now: str) -> MarketSnapshot | None:
    """Translate a Kalshi market record to our MarketSnapshot dataclass.

    Kalshi exposes `yes_bid`, `yes_ask`, `no_bid`, `no_ask`, `volume_24h`,
    `last_price`. We use last_price (in cents) divided by 100 as the
    implied probability, falling back to the bid/ask midpoint.
    """
    ticker = market.get("ticker") or ""
    title = market.get("title") or market.get("subtitle") or ""
    last = market.get("last_price")
    yes_bid = market.get("yes_bid"); yes_ask = market.get("yes_ask")
    yes_price = no_price = None
    if last is not None:
        try:
            yes_price = float(last) / 100.0
        except (TypeError, ValueError):
            pass
    if yes_price is None and yes_bid is not None and yes_ask is not None:
        try:
            yes_price = (float(yes_bid) + float(yes_ask)) / 200.0
        except (TypeError, ValueError):
            pass
    if yes_price is not None:
        no_price = 1.0 - yes_price
    vol = market.get("volume_24h") or market.get("volume24h")
    try:
        vol = float(vol) if vol is not None else None
    except (TypeError, ValueError):
        vol = None
    if not ticker:
        return None
    return MarketSnapshot(
        provider="kalshi",
        snapshot_utc=now,
        contract_id=ticker,
        title=title,
        yes_price=yes_price,
        no_price=no_price,
        volume_24h=vol,
        metadata={"event_ticker": market.get("event_ticker"),
                  "close_time": market.get("close_time")},
    )


def take_snapshot(out_dir: Path = DEFAULT_DIR,
                  timeout: float = 30.0) -> tuple[list[MarketSnapshot], Path]:
    """Pull WC 2026 markets from both providers and persist as JSONL.

    Returns the list of snapshots taken and the path to the JSONL file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    snaps: list[MarketSnapshot] = []
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        for raw in fetch_polymarket_wc_markets(client):
            snap = polymarket_to_snapshot(raw, now)
            if snap:
                snaps.append(snap)
        for raw in fetch_kalshi_wc_markets(client):
            snap = kalshi_to_snapshot(raw, now)
            if snap:
                snaps.append(snap)
    path = out_dir / f"snapshot_{now}.jsonl"
    with open(path, "w") as f:
        import json
        for s in snaps:
            f.write(json.dumps(asdict(s)) + "\n")
    return snaps, path


def load_history(in_dir: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Load all persisted snapshots into a single DataFrame.

    Index: (provider, contract_id). Columns include snapshot_utc,
    yes_price, no_price, volume_24h. One row per contract per snapshot.
    """
    if not in_dir.exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(in_dir.glob("snapshot_*.jsonl")):
        try:
            df = pd.read_json(path, lines=True)
            frames.append(df)
        except (ValueError, FileNotFoundError):
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def implied_match_outcome(snapshots: pd.DataFrame,
                          country_a: str,
                          country_b: str
                          ) -> dict[str, float] | None:
    """Find the latest implied (P(A wins), P(B wins), P(draw)) for a fixture.

    Scans the contract titles for country names; returns None when no
    matching markets are found. This is a best-effort name match; in
    production it should be backed by a curated contract_id-to-fixture
    table.
    """
    if snapshots.empty:
        return None
    latest = snapshots.sort_values("snapshot_utc").groupby("contract_id").tail(1)
    a_low = country_a.lower(); b_low = country_b.lower()
    rows = latest[
        latest["title"].str.lower().str.contains(a_low, na=False)
        & latest["title"].str.lower().str.contains(b_low, na=False)
    ]
    if rows.empty:
        return None
    out: dict[str, float] = {}
    for r in rows.itertuples():
        title_lo = (r.title or "").lower()
        yes = float(r.yes_price) if r.yes_price is not None else None
        if yes is None:
            continue
        if "draw" in title_lo:
            out["P_draw"] = yes
        elif a_low in title_lo and "win" in title_lo:
            out["P_A_wins"] = yes
        elif b_low in title_lo and "win" in title_lo:
            out["P_B_wins"] = yes
    return out or None
