"""Differentiated-cadence snapshot loop for FIFA Fantasy WC 2026.

Different data sources update at different rates. Polling everything on
the same schedule wastes API quota for slow-moving data and misses
intraday movement on fast-moving data. This loop runs four independent
schedules within a single process:

  - FIFA Fantasy API + features + models + optimizer:  every 12 hours
    (ownership and prices mostly update overnight + post-match; the
    GBM retrains on EPL + completed WC rounds via --include-wc each
    tick before scoring)
  - Polymarket + Kalshi prediction markets:            every 3 hours
    (intraday market reactions to news; training data for the Benter
    combiner accumulates over many snapshots)
  - RSS news + team-news lineup extraction:            every 6 hours
    (article cache under a disk budget, then predicted-XI extraction)
  - martj42 international Elo:                         every 24 hours
    (only changes when a country plays a match)

The optimizer recommendation is regenerated only on the FIFA-data tick,
since the prediction-market data does not currently feed the optimizer
(it's collected for the post-tournament Benter combiner training).

The dashboard is a served portal, not just a file: a background HTTP
thread serves the results/ directory and renders the page fresh on each
request. The backend ticks below keep the underlying results current; a
manual browser refresh always shows the latest. There is no auto-refresh
and no fast regeneration loop.

Configurable via env vars:

    WC_END_DATE                   default 2026-07-18 (loop exits)
    FIFA_INTERVAL_HOURS           default 12
    MARKETS_INTERVAL_HOURS        default 3
    ELO_INTERVAL_HOURS            default 24
    NEWS_INTERVAL_HOURS           default 6
    NEWS_BUDGET_MB                default 2048 (news article disk cap)
    NEWS_MAX_PER_FEED             default 20
    NEWS_FIXTURES_AHEAD           default 3
    STAGE                         default R32 in code; compose sets it
                                  (currently QF; advance per bracket)
    GBM_INCLUDE_WC                default 1 (retrain with --include-wc)
    OPTIMIZER_BACKEND             default ensemble (the predictions the
                                  optimizer consumes)
    WEB_PORT                      default 8770 (dashboard HTTP port)

Errors in any tick are logged and the next tick still runs.
"""
from __future__ import annotations

import http.server
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


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
    """Heavy tick: full FIFA-side refresh + retrain + model run + optimizer."""
    log("=== FIFA TICK ===")
    run(["python", "-m", "fifa_fantasy.collector"])
    run(["python", "-m", "fifa_fantasy.features"])
    # Retrain the GBM on EPL + every completed WC round before scoring.
    # Without this the model froze on the pre-tournament EPL-only fit and
    # never learned the WC scale, badly under-predicting in-form forwards
    # (see docs section 11f). The collector just refreshed round_points so
    # the WC labels are current. A retrain is a few seconds; if it fails we
    # still score with the previous models rather than crash the tick.
    if os.environ.get("GBM_INCLUDE_WC", "1") == "1":
        run(["python", "-m", "fifa_fantasy.model.train", "--include-wc"])
    # Run every backend so the record has each for comparison, then run the
    # ensemble last. The optimizer reads the last-written predictions, so the
    # final recommendation on the dashboard is the ensemble (poisson GK,
    # heuristic DEF, gbm MID/FWD). The routing follows the per-position
    # held-out RMSE winners (docs 11f); the ensemble's squad-level backtest
    # win is in-sample and labeled as such in 11f.4. Set OPTIMIZER_BACKEND
    # to pin a different final model.
    final_backend = os.environ.get("OPTIMIZER_BACKEND", "ensemble")
    for backend in ("heuristic", "poisson", "gbm", "ensemble"):
        run(["python", "-m", "fifa_fantasy.model", "--backend", backend])
    if final_backend != "ensemble":
        run(["python", "-m", "fifa_fantasy.model", "--backend", final_backend])
    run(["python", "-m", "fifa_fantasy.optimizer",
         "--stage", os.environ.get("STAGE", "R32")])
    web_tick()


def web_tick() -> None:
    """Write a static results/index.html fallback from the current results.

    The served portal renders on request, so this is only a convenience
    snapshot: it lets the page be opened directly as a file (file://) and
    keeps a committable copy current. Called on the FIFA tick, not on a
    fast loop; there is no auto-refresh.
    """
    run(["python", "-m", "fifa_fantasy.web"])


def serve_results(port: int) -> None:
    """Serve results/ over HTTP, rendering the dashboard fresh on each request.

    The backend ticks keep producing results; a request for the index
    re-reads results/ and renders the latest, so a manual browser refresh
    always shows the newest recommendation with no polling and no
    auto-refresh. Other paths (json, md) are served as static files.

    Best-effort: a bind failure or a missing web dependency is logged and
    the loop continues; the static index.html remains available.
    """
    try:
        from fifa_fantasy.web.render import build_html
    except Exception as e:  # noqa: BLE001
        log(f"web server not started: cannot import renderer ({e})")
        return

    class DashboardHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory="results", **kwargs)

        def do_GET(self):  # noqa: N802
            if self.path.split("?")[0] in ("/", "/index.html"):
                try:
                    html, _, _ = build_html(Path("results"), refresh_seconds=0)
                    body = html.encode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    self.send_error(500, f"render failed: {exc}")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            return super().do_GET()

        def log_message(self, *args):  # keep the loop's stdout clean
            pass

    try:
        httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), DashboardHandler)
    except OSError as e:
        log(f"web server not started on :{port} ({e})")
        return
    log(f"dashboard serving on :{port} (renders on request)")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def markets_tick() -> None:
    """Light tick: pull Polymarket + Kalshi snapshots only."""
    log("=== MARKETS TICK ===")
    run(["python", "-m", "fifa_fantasy.external",
         "--skip-international", "--skip-football-data"])


def news_tick() -> None:
    """News tick: collect RSS articles, then run team-news lineup extraction.

    Two-stage: the news module fills the article cache from RSS feeds
    under a disk-budget cap; the team_news module then extracts
    predicted XIs from those cached articles. Splitting them keeps the
    RSS poller cheap and the lineup extractor independent.
    """
    log("=== NEWS TICK ===")
    # 1. Collect articles from RSS feeds (cheap, ~50KB per tick).
    run(["python", "-m", "fifa_fantasy.external.news",
         "--budget-mb", os.environ.get("NEWS_BUDGET_MB", "2048"),
         "--max-per-feed", os.environ.get("NEWS_MAX_PER_FEED", "20")])
    # 2. Extract predicted XIs (uses the article cache + curated seed URLs).
    run(["python", "-m", "fifa_fantasy.external.team_news",
         "--fixtures-ahead",
         os.environ.get("NEWS_FIXTURES_AHEAD", "3")])
    # 3. Extract per-player signals (injury/suspension/rotation/potm) from
    # the article cache using the config-driven vocabulary. Intel table
    # only; nothing feeds the models without passing validation first.
    run(["python", "-m", "fifa_fantasy.external.news.signals",
         "--days", os.environ.get("NEWS_SIGNALS_DAYS", "3")])


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
    web_port = env_int("WEB_PORT", 8770)
    log(f"loop start; end={end.isoformat()} "
        f"fifa={fifa_interval//3600}h markets={markets_interval//3600}h "
        f"elo={elo_interval//3600}h news={news_interval//3600}h "
        f"web_port={web_port} (renders on request)")

    # The dashboard is served on request; the backend ticks below keep the
    # underlying results current. No auto-refresh, no fast web loop.
    serve_results(web_port)

    next_fifa = next_markets = next_elo = next_news = 0.0
    sleep_resolution = 30

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

        time.sleep(sleep_resolution)

    log("WC end reached; exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
