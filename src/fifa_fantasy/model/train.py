"""Train the per-position LightGBM heads from the FPL Parquet.

    python -m fifa_fantasy.model.train                       # uses latest data/training/*.parquet
    python -m fifa_fantasy.model.train --training data/training/fpl_player_gameweek_2024-25.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fifa_fantasy.training.features import build_training_table

from .gbm import (
    DEFAULT_MODELS_DIR,
    POSITIONS,
    TrainConfig,
    save_models,
    train_all,
)

DEFAULT_TRAINING_DIR = Path("data/training")


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.model.train")
    parser.add_argument("--training", type=Path, default=None,
                        help="player_gameweek Parquet (default: latest under data/training/)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    src = args.training or _latest(DEFAULT_TRAINING_DIR, "fpl_player_gameweek")
    raw = pd.read_parquet(src)
    print(f"training source: {src}  rows={len(raw):,}")

    train_df = build_training_table(raw)
    print(f"training rows after dropping DNPs: {len(train_df):,}")
    for pos in POSITIONS:
        n = (train_df["position"] == pos).sum()
        print(f"  {pos}: {n:,} rows")

    cfg = TrainConfig(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    models = train_all(train_df, cfg)
    save_models(models, args.out_dir)
    print(f"saved models -> {args.out_dir}/")


if __name__ == "__main__":
    main()
