"""Train the per-position LightGBM heads.

    python -m fifa_fantasy.model.train                       # EPL only
    python -m fifa_fantasy.model.train --include-wc          # EPL + realised WC rows so far
    python -m fifa_fantasy.model.train --training <path>     # explicit FPL parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fifa_fantasy.training.features import build_training_table
from fifa_fantasy.training.wc import extract_wc_training_rows

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
                        help="EPL player_gameweek Parquet (default: latest under data/training/)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--include-wc", action="store_true",
                        help="append realised WC player-round rows from data/raw/")
    args = parser.parse_args()

    src = args.training or _latest(DEFAULT_TRAINING_DIR, "fpl_player_gameweek")
    raw = pd.read_parquet(src)
    train_df = build_training_table(raw)
    train_df["source"] = "epl"
    print(f"EPL source: {src}  ({len(train_df):,} rows after DNP drop)")

    if args.include_wc:
        wc = extract_wc_training_rows()
        if wc.empty:
            print("WC training rows: 0 (no completed rounds yet)")
        else:
            print(f"WC training rows: {len(wc):,}")
            common = [c for c in train_df.columns if c in wc.columns]
            train_df = pd.concat([train_df[common], wc[common]], ignore_index=True)

    print(f"total training rows: {len(train_df):,}")
    for pos in POSITIONS:
        n = (train_df["position"] == pos).sum()
        print(f"  {pos}: {n:,} rows")

    cfg = TrainConfig(n_estimators=args.n_estimators, learning_rate=args.learning_rate)
    models = train_all(train_df, cfg)
    save_models(models, args.out_dir)
    print(f"saved models -> {args.out_dir}/")


if __name__ == "__main__":
    main()
