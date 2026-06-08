"""CLI entry point.

    python -m fifa_fantasy.features
    python -m fifa_fantasy.features --raw-dir data/raw --out-dir data/processed

Reads the most recent `players_<date>.parquet`, `squads_<date>.parquet`,
and `fixtures_<date>.parquet` from `--raw-dir`, builds the per-(player,
round) feature table, and writes `features_<UTC-date>.parquet` to
`--out-dir`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fifa_fantasy.collector.rankings import DEFAULT_PATH as RANKINGS_PATH
from fifa_fantasy.collector.rankings import load_rankings

from .build import build_player_round_features
from .squad import squad_strength

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUT_DIR = Path("data/processed")


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.features")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rankings", type=Path, default=RANKINGS_PATH,
                        help="path to FIFA ranking CSV (data/static/fifa_rankings.csv)")
    args = parser.parse_args()

    players = pd.read_parquet(_latest(args.raw_dir, "players"))
    squads = pd.read_parquet(_latest(args.raw_dir, "squads"))
    fixtures = pd.read_parquet(_latest(args.raw_dir, "fixtures"))
    # Phase 1 serializes datetimes via Pydantic's JSON mode, so kickoff lands
    # as ISO strings. Restore tz-aware datetime for time-delta arithmetic.
    fixtures["kickoff"] = pd.to_datetime(fixtures["kickoff"], utc=True)
    rankings = load_rankings(args.rankings)

    strength = squad_strength(players, squads, rankings=rankings)
    features = build_player_round_features(players, fixtures, strength)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = args.out_dir / f"features_{date}.parquet"
    features.to_parquet(path, index=False)
    print(f"features: {len(features):5d} rows × {len(features.columns)} cols  → {path}")


if __name__ == "__main__":
    main()
