"""CLI entry point.

    python -m fifa_fantasy.collector          # fetch players + squads + fixtures
    python -m fifa_fantasy.collector --only players
    python -m fifa_fantasy.collector --data-dir /tmp/out
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import (
    PLAYERS_PATH,
    ROUNDS_PATH,
    SQUADS_PATH,
    get_json,
    make_client,
)
from .parse import parse_fixtures, parse_players, parse_squads
from .persist import DEFAULT_DATA_DIR, save_parquet, save_raw_json

ALL_TARGETS = ("squads", "players", "fixtures")


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.collector")
    parser.add_argument(
        "--only",
        choices=ALL_TARGETS,
        action="append",
        help="restrict to a subset (repeatable); default = all three",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"output directory (default {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    targets = tuple(args.only) if args.only else ALL_TARGETS

    with make_client() as client:
        # squads is always fetched if any of {squads, players} is requested,
        # because parse_players joins on the squad table.
        squads = None
        if "squads" in targets or "players" in targets:
            raw_squads = get_json(client, SQUADS_PATH)
            save_raw_json(raw_squads, "squads", args.data_dir)
            squads = parse_squads(raw_squads)
            if "squads" in targets:
                path = save_parquet(squads, "squads", args.data_dir)
                print(f"squads:    {len(squads):4d} rows  → {path}")

        if "players" in targets:
            raw_players = get_json(client, PLAYERS_PATH)
            save_raw_json(raw_players, "players", args.data_dir)
            players = parse_players(raw_players, squads or [])
            path = save_parquet(players, "players", args.data_dir)
            print(f"players:   {len(players):4d} rows  → {path}")

        if "fixtures" in targets:
            raw_rounds = get_json(client, ROUNDS_PATH)
            save_raw_json(raw_rounds, "rounds", args.data_dir)
            fixtures = parse_fixtures(raw_rounds)
            path = save_parquet(fixtures, "fixtures", args.data_dir)
            print(f"fixtures:  {len(fixtures):4d} rows  → {path}")


if __name__ == "__main__":
    main()
