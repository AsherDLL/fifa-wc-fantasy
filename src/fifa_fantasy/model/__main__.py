"""CLI: run the baseline predictor over the latest features Parquet.

    python -m fifa_fantasy.model
    python -m fifa_fantasy.model --features-dir data/processed --out-dir data/processed
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .baseline import heuristic_predict

DEFAULT_DIR = Path("data/processed")


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.model")
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    features = pd.read_parquet(_latest(args.features_dir, "features"))
    predictions = heuristic_predict(features)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = args.out_dir / f"predictions_{date}.parquet"
    predictions.to_parquet(path, index=False)
    pp = predictions["predicted_points"]
    print(
        f"predictions: {len(predictions):5d} rows  → {path}\n"
        f"  predicted_points: min={pp.min():.2f} mean={pp.mean():.2f} max={pp.max():.2f}"
    )


if __name__ == "__main__":
    main()
