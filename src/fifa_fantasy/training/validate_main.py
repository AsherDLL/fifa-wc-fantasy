"""CLI: run held-out validation across all three backends.

    python -m fifa_fantasy.training.validate_main
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .validate import (
    DEFAULT_REPORT,
    DEFAULT_TRAINING_DIR,
    evaluate,
    load_all_seasons,
    save_report,
    split_train_holdout,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.training.validate")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--holdout-season", default="2024-25")
    parser.add_argument("--holdout-gw-min", type=int, default=30)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    all_seasons = load_all_seasons(args.training_dir)
    if all_seasons.empty:
        raise SystemExit(f"no training data found under {args.training_dir}")

    train_df, holdout_df = split_train_holdout(
        all_seasons,
        holdout_season=args.holdout_season,
        holdout_gw_min=args.holdout_gw_min,
    )
    print(f"training rows: {len(train_df):,}  "
          f"holdout rows: {len(holdout_df):,}  "
          f"({args.holdout_season} GW>={args.holdout_gw_min})")

    rows = evaluate(train_df, holdout_df)
    print()
    print(f"{'pos':<4} {'n':>6}   {'heuristic':>10} {'poisson':>10} {'gbm':>10}")
    for r in rows:
        print(f"{r.position:<4} {r.n:>6}   "
              f"{r.heuristic_rmse:>10.3f} {r.poisson_rmse:>10.3f} {r.gbm_rmse:>10.3f}")

    path = save_report(rows, args.report)
    print(f"\nreport -> {path}")


if __name__ == "__main__":
    main()
