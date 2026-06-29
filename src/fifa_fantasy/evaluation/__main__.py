"""CLI: compute per-round accuracy after a matchday completes.

Usage:
    python -m fifa_fantasy.evaluation --round 3
    python -m fifa_fantasy.evaluation --round 4 --user-squad data/user_squads/round_04.json

The `--user-squad` JSON is a small file the user produces after each
round documenting their actual squad. Schema:

    {
      "round_id": 4,
      "squad_player_ids": [...15 ints...],
      "starter_ids":      [...11 ints...],
      "captain_id": 38,
      "vice_captain_id": 501,
      "formation": "3-4-3"
    }

Without --user-squad the eval module just computes the model-only
accuracy and random baseline, skipping the squad-performance comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dataclasses import asdict

from .accuracy import (
    captain_analysis,
    DEFAULT_EVAL_DIR,
    emit_report,
    model_accuracy,
    random_baseline_score,
    squad_performance,
)


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.evaluation")
    parser.add_argument("--round", type=int, required=True,
                        help="round id to evaluate (1=MD1, 4=R32, etc.)")
    parser.add_argument("--actuals", type=Path,
                        default=None,
                        help="players_<date>.parquet with realised round_points "
                             "(defaults to latest under data/raw/)")
    parser.add_argument("--user-squad", type=Path, default=None,
                        help="optional JSON declaring the user's actual squad")
    parser.add_argument("--predictions-dir", type=Path,
                        default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--baseline-trials", type=int, default=1000)
    args = parser.parse_args()

    actuals_path = args.actuals or _latest(Path("data/raw"), "players")
    actuals = pd.read_parquet(actuals_path)
    print(f"actuals: {actuals_path} ({len(actuals)} rows)")

    # Per-backend accuracy from per-backend predictions files.
    # We expect one predictions parquet per backend, named with the backend
    # suffix. If only the latest predictions_*.parquet is present, we
    # treat it as the single backend in use (heuristic by default).
    backend_files: dict[str, Path] = {}
    for path in sorted(args.predictions_dir.glob("predictions_*.parquet")):
        try:
            df_sample = pd.read_parquet(path).head(1)
        except Exception:
            continue
        if df_sample.empty:
            continue
        if "model_backend" in df_sample.columns:
            backend = str(df_sample.iloc[0]["model_backend"])
        else:
            # Pre-versioned predictions; infer from filename pattern.
            backend = "unknown"
        backend_files.setdefault(backend, path)
    print(f"backends found: {list(backend_files.keys())}")

    model_rows = []
    for backend, path in backend_files.items():
        preds = pd.read_parquet(path)
        rows = model_accuracy(preds, actuals, args.round, backend)
        model_rows.extend(rows)
        print(f"  {backend}: {len(rows)} per-position rows")

    captain_rows = []
    squad_rows = []

    if args.user_squad is not None and args.user_squad.exists():
        user = json.loads(args.user_squad.read_text())
        perf = squad_performance(
            squad_player_ids=user["squad_player_ids"],
            starter_ids=user["starter_ids"],
            captain_id=user["captain_id"],
            actuals=actuals,
            round_id=args.round,
            source="user_actual",
        )
        squad_rows.append(perf)
        # captain analysis on user's chosen captain
        captain_rows.append(captain_analysis(
            predictions=pd.read_parquet(next(iter(backend_files.values()))),
            actuals=actuals,
            starter_ids=user["starter_ids"],
            chosen_captain_id=user["captain_id"],
            round_id=args.round,
            backend="user_actual",
        ))

    random_b = random_baseline_score(actuals, args.round,
                                     n_trials=args.baseline_trials)
    print(f"random baseline: mean={random_b['mean']:.1f} median={random_b['median']:.1f} "
          f"p10={random_b['p10']:.1f} p90={random_b['p90']:.1f}")

    out_path = emit_report(args.round, model_rows, squad_rows,
                           captain_rows, random_b, args.out_dir)
    print(f"report -> {out_path}")

    if model_rows:
        df = pd.DataFrame([asdict(r) for r in model_rows])
        print("\nPer-(backend, position) accuracy:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
