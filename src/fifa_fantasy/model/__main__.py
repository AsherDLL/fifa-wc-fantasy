"""CLI: run the chosen predictor backend over the latest features Parquet.

    python -m fifa_fantasy.model                              # heuristic backend (default)
    python -m fifa_fantasy.model --backend gbm                # LightGBM (must be trained first)
    python -m fifa_fantasy.model --backend gbm --models-dir data/models

The GBM backend reads `data/models/gbm_<position>_<head>.txt`. Train
those with `python -m fifa_fantasy.model.train`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .baseline import DEFAULT_PREMIUM_BOOST, heuristic_predict
from .gbm import DEFAULT_MODELS_DIR, GBM_VERSION, load_models, predict as gbm_predict
from .poisson import poisson_predict

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
    parser.add_argument("--backend", choices=("heuristic", "gbm", "poisson"), default="heuristic")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="LightGBM model artefacts (used only for --backend gbm)")
    parser.add_argument(
        "--premium-boost",
        type=float,
        default=DEFAULT_PREMIUM_BOOST,
        help=("heuristic-only knob: non-linear ceiling for $9M+ players "
              "(default 0.0). Try 0.3-0.6 to tilt toward premium picks."),
    )
    args = parser.parse_args()

    features = pd.read_parquet(_latest(args.features_dir, "features"))

    if args.backend == "heuristic":
        predictions = heuristic_predict(features, premium_boost=args.premium_boost)
        backend_label = "heuristic"
    elif args.backend == "gbm":
        models = load_models(args.models_dir)
        predictions = gbm_predict(features, models)
        backend_label = "gbm"
    else:
        predictions = poisson_predict(features)
        backend_label = "poisson"

    # Stamp the backend name into the predictions table so downstream
    # tools (optimizer filename, web report) can read which model produced
    # this snapshot without an extra sidecar file. For the GBM, also
    # stamp the version so legacy outputs can be told apart from current
    # ones.
    predictions["model_backend"] = backend_label
    # Only the GBM has versioned artefacts. Other backends are formulas
    # and do not need a version stamp.
    predictions["model_version"] = (
        GBM_VERSION if backend_label == "gbm" else ""
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = args.out_dir / f"predictions_{date}.parquet"
    predictions.to_parquet(path, index=False)
    pp = predictions["predicted_points"]
    print(
        f"predictions: {len(predictions):5d} rows  ({backend_label})  -> {path}\n"
        f"  predicted_points: min={pp.min():.2f} mean={pp.mean():.2f} max={pp.max():.2f}"
    )


if __name__ == "__main__":
    main()
