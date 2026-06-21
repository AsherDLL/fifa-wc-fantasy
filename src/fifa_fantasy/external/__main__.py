"""CLI for refreshing external datasets.

    python -m fifa_fantasy.external                          # refresh both
    python -m fifa_fantasy.external --skip-football-data
    python -m fifa_fantasy.external --skip-international
    python -m fifa_fantasy.external --no-refresh-cache       # reuse cached files
"""
from __future__ import annotations

import argparse

from .football_data import LEAGUES, refresh_all
from .international_elo import refresh as refresh_elo


def main() -> None:
    p = argparse.ArgumentParser(prog="fifa_fantasy.external")
    p.add_argument("--skip-international", action="store_true")
    p.add_argument("--skip-football-data", action="store_true")
    p.add_argument("--no-refresh-cache", action="store_true",
                   help="reuse any cached CSVs instead of re-downloading")
    p.add_argument("--seasons", nargs="+", default=["2223", "2324", "2425"])
    p.add_argument("--leagues", nargs="+", default=list(LEAGUES))
    args = p.parse_args()

    do_refresh = not args.no_refresh_cache

    if not args.skip_international:
        snap = refresh_elo(refresh_cache=do_refresh)
        print(f"international_elo: {len(snap):4d} countries -> data/external/country_elo.csv")
        top = snap.sort_values("elo", ascending=False).head(10)[["country_name", "elo", "last10_form"]]
        print("  top 10:")
        for r in top.itertuples(index=False):
            print(f"    {r.country_name:<24} elo={r.elo:7.1f}  last10_form={r.last10_form:.2f}")

    if not args.skip_football_data:
        matches = refresh_all(
            seasons=tuple(args.seasons),
            leagues=tuple(args.leagues),
            refresh=do_refresh,
        )
        print(f"football_data: {len(matches):5d} matches -> data/external/fd_matches.parquet")


if __name__ == "__main__":
    main()
