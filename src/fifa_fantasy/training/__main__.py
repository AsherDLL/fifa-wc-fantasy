"""CLI: scrape one EPL FPL season into Parquet.

    python -m fifa_fantasy.training                    # default workers=8
    python -m fifa_fantasy.training --season 2024-25 --workers 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .fpl import (
    DEFAULT_OUT,
    DEFAULT_WORKERS,
    build_player_gameweek_table,
    fetch_bootstrap,
    _make_client,
    save_table,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.training")
    parser.add_argument("--season", default="current",
                        help="label written into the output filename")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    with _make_client() as client:
        bootstrap = fetch_bootstrap(client)
    df = build_player_gameweek_table(bootstrap, workers=args.workers)
    path = save_table(df, season=args.season, out_dir=args.out_dir)
    print(f"player_gameweek rows: {len(df):>6,}")
    print(f"unique players:       {df['player_id'].nunique():>6,}")
    print(f"gameweeks:            {df['gameweek'].nunique():>6}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
