"""Persist PredictedXI records to disk."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .models import PredictedXI

DEFAULT_DIR = Path("data/external/team_news")


def persist(records: list[PredictedXI], out_dir: Path = DEFAULT_DIR) -> Path:
    """Persist a list of PredictedXI to a single timestamped parquet.

    Schema (one row per (fixture, player, status)):
        scraped_at_utc, source, fixture_id, squad_abbr, opponent_abbr,
        is_home, player_id, status, source_confidence
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    path = out_dir / f"team_news_{ts}.parquet"

    rows = []
    for r in records:
        for pid in r.home_starting_player_ids:
            rows.append({
                "scraped_at_utc": r.scraped_at_utc,
                "source": r.source,
                "fixture_id": r.fixture_id,
                "squad_abbr": r.home_squad_abbr,
                "opponent_abbr": r.away_squad_abbr,
                "is_home": True,
                "player_id": pid,
                "status": "starting",
                "source_confidence": r.source_confidence,
            })
        for pid in r.away_starting_player_ids:
            rows.append({
                "scraped_at_utc": r.scraped_at_utc,
                "source": r.source,
                "fixture_id": r.fixture_id,
                "squad_abbr": r.away_squad_abbr,
                "opponent_abbr": r.home_squad_abbr,
                "is_home": False,
                "player_id": pid,
                "status": "starting",
                "source_confidence": r.source_confidence,
            })
        for pid in r.home_bench_player_ids:
            rows.append({
                "scraped_at_utc": r.scraped_at_utc,
                "source": r.source,
                "fixture_id": r.fixture_id,
                "squad_abbr": r.home_squad_abbr,
                "opponent_abbr": r.away_squad_abbr,
                "is_home": True,
                "player_id": pid,
                "status": "bench",
                "source_confidence": r.source_confidence,
            })
        for pid in r.away_bench_player_ids:
            rows.append({
                "scraped_at_utc": r.scraped_at_utc,
                "source": r.source,
                "fixture_id": r.fixture_id,
                "squad_abbr": r.away_squad_abbr,
                "opponent_abbr": r.home_squad_abbr,
                "is_home": False,
                "player_id": pid,
                "status": "bench",
                "source_confidence": r.source_confidence,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
    else:
        # Write an empty marker so the daemon's tick is observable.
        pd.DataFrame(columns=["scraped_at_utc", "source", "fixture_id",
                              "squad_abbr", "opponent_abbr", "is_home",
                              "player_id", "status", "source_confidence"]).to_parquet(path, index=False)
    return path


def load_latest(dir_: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Load the most recent team-news parquet (empty if none exists)."""
    if not dir_.exists():
        return pd.DataFrame()
    files = sorted(dir_.glob("team_news_*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


def load_all(dir_: Path = DEFAULT_DIR) -> pd.DataFrame:
    """Concatenate every persisted parquet for full history."""
    if not dir_.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(p) for p in sorted(dir_.glob("team_news_*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
