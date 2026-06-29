"""Differentiated-cadence snapshot loop for FIFA Fantasy WC 2026.

Different data sources update at different rates. Polling everything on
the same schedule wastes API quota for slow-moving data and misses
intraday movement on fast-moving data. This loop runs three independent
schedules within a single process:

  - FIFA Fantasy API + features + models + optimizer:  every 12 hours
    (ownership and prices mostly update overnight + post-match)
  - Polymarket + Kalshi prediction markets:            every 3 hours
    (intraday market reactions to news; training data for the Benter
    combiner accumulates over many snapshots)
  - martj42 international Elo:                         once daily
    (only changes when a country plays a match)

The optimizer recommendation is regenerated only on the FIFA-data tick,
since the prediction-market data does not currently feed the optimizer
(it's collected for the post-tournament Benter combiner training).

Configurable via env vars:

    WC_END_DATE                   default 2026-07-18 (loop exits)
    FIFA_INTERVAL_HOURS           default 12
    MARKETS_INTERVAL_HOURS        default 3
    ELO_INTERVAL_HOURS            default 24
    STAGE                         default R32

Errors in any tick are logged and the next tick still runs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_date(name: str, default: str) -> datetime:
    raw = os.environ.get(name, default)
    return datetime.fromisoformat(raw).replace(tzinfo=timezone.utc)


def log(msg: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{now}] {msg}", flush=True)


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except FileNotFoundError as e:
        log(f"  ERROR: {e}")
        return 1


def fifa_tick() -> None:
    """Heavy tick: full FIFA-side refresh + model run + optimizer."""
    log("=== FIFA TICK ===")
    run(["python", "-m", "fifa_fantasy.collector"])
    run(["python", "-m", "fifa_fantasy.features"])
    for backend in ("heuristic", "poisson", "gbm"):
        run(["python", "-m", "fifa_fantasy.model", "--backend", backend])
    run(["python", "-m", "fifa_fantasy.optimizer",
         "--stage", os.environ.get("STAGE", "R32")])
    run(["python", "-m", "fifa_fantasy.web"])


def markets_tick() -> None:
    """Light tick: pull Polymarket + Kalshi snapshots only."""
    log("=== MARKETS TICK ===")
    run(["python", "-m", "fifa_fantasy.external",
         "--skip-international", "--skip-football-data"])


def news_tick() -> None:
    """Team-news tick: scrape predicted XIs for upcoming fixtures."""
    log("=== NEWS TICK ===")
    run(["python", "-m", "fifa_fantasy.external.team_news",
         "--fixtures-ahead",
         os.environ.get("NEWS_FIXTURES_AHEAD", "3")])


def elo_tick() -> None:
    """Daily tick: refresh martj42 Elo only."""
    log("=== ELO TICK ===")
    run(["python", "-m", "fifa_fantasy.external",
         "--skip-football-data", "--skip-markets"])


def main() -> int:
    end = env_date("WC_END_DATE", "2026-07-18")
    fifa_interval = env_int("FIFA_INTERVAL_HOURS", 12) * 3600
    markets_interval = env_int("MARKETS_INTERVAL_HOURS", 3) * 3600
    elo_interval = env_int("ELO_INTERVAL_HOURS", 24) * 3600
    news_interval = env_int("NEWS_INTERVAL_HOURS", 6) * 3600
    log(f"loop start; end={end.isoformat()} "
        f"fifa={fifa_interval//3600}h markets={markets_interval//3600}h "
        f"elo={elo_interval//3600}h news={news_interval//3600}h")

    next_fifa = next_markets = next_elo = next_news = 0.0
    sleep_resolution = 60

    while datetime.now(timezone.utc) < end:
        now = time.time()
        if now >= next_fifa:
            try: fifa_tick()
            except Exception as e: log(f"fifa tick crashed: {e}")  # noqa: BLE001
            next_fifa = now + fifa_interval
        if now >= next_markets:
            try: markets_tick()
            except Exception as e: log(f"markets tick crashed: {e}")  # noqa: BLE001
            next_markets = now + markets_interval
        if now >= next_elo:
            try: elo_tick()
            except Exception as e: log(f"elo tick crashed: {e}")  # noqa: BLE001
            next_elo = now + elo_interval
        if now >= next_news:
            try: news_tick()
            except Exception as e: log(f"news tick crashed: {e}")  # noqa: BLE001
            next_news = now + news_interval

        next_event = min(next_fifa, next_markets, next_elo, next_news)
        sleep_for = max(sleep_resolution, min(next_event - now, sleep_resolution))
        time.sleep(sleep_for)

    log("WC end reached; exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
