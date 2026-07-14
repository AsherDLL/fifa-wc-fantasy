"""CLI: refresh predicted XIs for upcoming WC 2026 fixtures.

    python -m fifa_fantasy.external.team_news --fixtures-ahead 3
    python -m fifa_fantasy.external.team_news --espn-seed-file urls.txt

Outputs `data/external/team_news/team_news_<utc-iso>.parquet`. The
features pipeline reads the latest of these and joins it into the
per-(player, round) feature table.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..scraping import StealthClient
from .matcher import PlayerNameMatcher
from .models import PredictedXI
from .parsers import espn, soccerdata_proxy
from .store import DEFAULT_DIR, persist

log = logging.getLogger(__name__)


def _upcoming_fixtures(raw_dir: Path, days_ahead: float = 3.0) -> pd.DataFrame:
    files = sorted(raw_dir.glob("fixtures_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no fixtures_*.parquet in {raw_dir}")
    fix = pd.read_parquet(files[-1])
    fix["kickoff"] = pd.to_datetime(fix["kickoff"], utc=True)
    now = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(now) + pd.Timedelta(days=days_ahead)
    return fix[(fix["kickoff"] >= now) & (fix["kickoff"] <= cutoff)]


def _raw_to_predicted(raw, fixture, matcher) -> PredictedXI:
    """Resolve names in a RawLineup to FIFA ids and produce a PredictedXI."""
    home_abbr = fixture["home_squad_abbr"]
    away_abbr = fixture["away_squad_abbr"]
    home_starting = []
    away_starting = []
    home_bench = []
    away_bench = []
    unmatched = []

    for p in raw.home_lineup:
        m = matcher.match(p.name, country_abbr=home_abbr, position=p.position)
        if m.player_id is None:
            unmatched.append(p.name)
            continue
        if p.status == "bench":
            home_bench.append(m.player_id)
        else:
            home_starting.append(m.player_id)

    for p in raw.away_lineup:
        m = matcher.match(p.name, country_abbr=away_abbr, position=p.position)
        if m.player_id is None:
            unmatched.append(p.name)
            continue
        if p.status == "bench":
            away_bench.append(m.player_id)
        else:
            away_starting.append(m.player_id)

    return PredictedXI(
        source=raw.source,
        scraped_at_utc=raw.scraped_at_utc,
        fixture_id=int(fixture["fixture_id"]),
        home_squad_abbr=home_abbr,
        away_squad_abbr=away_abbr,
        home_starting_player_ids=home_starting,
        away_starting_player_ids=away_starting,
        home_bench_player_ids=home_bench,
        away_bench_player_ids=away_bench,
        source_confidence=raw.confidence,
        unmatched_names=unmatched,
        raw_text=raw.raw_text,
    )


def main() -> int:
    p = argparse.ArgumentParser(prog="fifa_fantasy.external.team_news")
    p.add_argument("--fixtures-ahead", type=float, default=3.0,
                   help="Look this many days ahead for fixtures to scrape")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--espn-seed-file", type=Path, default=None,
                   help="Optional file with ESPN article URLs (one per line, "
                        "format: <url><TAB><home_abbr><TAB><away_abbr>)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--dry-run", action="store_true",
                   help="Do not persist; print what would happen")
    p.add_argument("--cache-dir", type=Path,
                   default=Path("data/external/cache/scraping"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    upcoming = _upcoming_fixtures(args.raw_dir, args.fixtures_ahead)
    log.info("upcoming fixtures in next %.1f days: %d",
             args.fixtures_ahead, len(upcoming))
    if upcoming.empty:
        log.info("no upcoming fixtures; exiting cleanly")
        return 0

    matcher = PlayerNameMatcher.from_latest(args.raw_dir)
    # Coherent, rotating identity pool from kewpie (not a single pinned,
    # empty-UA target). See news/__main__ for the rationale.
    client = StealthClient(
        rate_limit_per_second=0.5,        # be conservative for ethics
        cache_dir=str(args.cache_dir),
        cache_ttl_hours=3.0,
    )

    records: list[PredictedXI] = []

    # 1. soccerdata library (preferred when it has coverage).
    raws = soccerdata_proxy.fetch_predicted_xis(upcoming)
    for raw in raws:
        # Match the raw to a fixture by team names.
        fixture_match = upcoming[
            (upcoming["home_squad_name"].str.lower() == raw.home_team_name.lower())
            & (upcoming["away_squad_name"].str.lower() == raw.away_team_name.lower())
        ]
        if fixture_match.empty:
            log.info("soccerdata raw could not be matched to fixture: %s vs %s",
                     raw.home_team_name, raw.away_team_name)
            continue
        records.append(_raw_to_predicted(raw, fixture_match.iloc[0], matcher))

    # 2. ESPN seed file (manual URL provision).
    if args.espn_seed_file and args.espn_seed_file.exists():
        for line in args.espn_seed_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            url, home_abbr, away_abbr = parts[0], parts[1], parts[2]
            fixture_match = upcoming[
                (upcoming["home_squad_abbr"] == home_abbr)
                & (upcoming["away_squad_abbr"] == away_abbr)
            ]
            if fixture_match.empty:
                log.info("seed URL fixture %s vs %s not upcoming; skipping",
                         home_abbr, away_abbr)
                continue
            fixture = fixture_match.iloc[0]
            raw = espn.fetch_predicted_xi(
                client, url,
                home_team_name=fixture["home_squad_name"],
                away_team_name=fixture["away_squad_name"],
            )
            if raw is None:
                log.info("ESPN parser returned nothing for %s", url)
                continue
            records.append(_raw_to_predicted(raw, fixture, matcher))

    log.info("scraped %d PredictedXI records", len(records))
    if args.dry_run:
        for r in records:
            log.info("  fixture %s %s vs %s: %d starting, %d bench, %d unmatched",
                     r.fixture_id, r.home_squad_abbr, r.away_squad_abbr,
                     len(r.home_starting_player_ids) + len(r.away_starting_player_ids),
                     len(r.home_bench_player_ids) + len(r.away_bench_player_ids),
                     len(r.unmatched_names))
        return 0
    out_path = persist(records, args.out_dir)
    log.info("persisted -> %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
