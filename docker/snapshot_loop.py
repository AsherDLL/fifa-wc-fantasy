"""Daily snapshot loop for the FIFA Fantasy WC 2026 prediction system.

Runs the full data refresh + model pipeline once per scheduled tick
through the end of the tournament (2026-07-18). Designed to run as a
long-lived process inside a Docker container with `data/` and `results/`
mounted from the host.

What it does each tick:
    1. fifa_fantasy.collector       -> FIFA API snapshot (players, squads, fixtures)
    2. fifa_fantasy.external        -> martj42 Elo, football-data, Polymarket, Kalshi
    3. fifa_fantasy.features        -> per-(player, round) features
    4. fifa_fantasy.model x3        -> heuristic, Poisson, GBM predictions
    5. fifa_fantasy.optimizer       -> per-stage recommendation JSON+MD
    6. fifa_fantasy.web             -> regenerate the static HTML report

Configurable via env vars:

    SNAPSHOT_INTERVAL_HOURS    default 6 (every 6 hours)
    WC_END_DATE                default 2026-07-18 (loop exits after this)
    STAGE                      default GROUP_MD1 (currently used by optimizer)

Errors are logged to stderr; the loop continues on the next tick. To
inspect a tick's output, mount data/ and results/ from the host.
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


def run(cmd: list[str], *, allow_failure: bool = False) -> int:
    log("$ " + " ".join(cmd))
    try:
        rc = subprocess.call(cmd)
    except FileNotFoundError as e:
        log(f"  ERROR: {e}")
        return 1
    if rc != 0:
        log(f"  exit code {rc}")
        if not allow_failure:
            return rc
    return rc


def tick() -> None:
    """One pass through the refresh pipeline. Errors in one stage do not
    block subsequent stages."""
    log("=== TICK START ===")
    run(["python", "-m", "fifa_fantasy.collector"], allow_failure=True)
    run(["python", "-m", "fifa_fantasy.external",
         "--skip-football-data"], allow_failure=True)
    run(["python", "-m", "fifa_fantasy.features"], allow_failure=True)
    for backend in ("heuristic", "poisson", "gbm"):
        run(["python", "-m", "fifa_fantasy.model",
             "--backend", backend], allow_failure=True)
    stage = os.environ.get("STAGE", "GROUP_MD1")
    run(["python", "-m", "fifa_fantasy.optimizer",
         "--stage", stage], allow_failure=True)
    run(["python", "-m", "fifa_fantasy.web"], allow_failure=True)
    log("=== TICK END ===")


def main() -> int:
    interval_h = env_int("SNAPSHOT_INTERVAL_HOURS", 6)
    end = env_date("WC_END_DATE", "2026-07-18")
    log(f"snapshot loop starting; interval={interval_h}h end={end.isoformat()}")
    while datetime.now(timezone.utc) < end:
        try:
            tick()
        except Exception as e:  # noqa: BLE001
            log(f"tick crashed: {type(e).__name__}: {e}")
        log(f"sleeping {interval_h} hours")
        time.sleep(interval_h * 3600)
    log("WC end reached; exiting cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
