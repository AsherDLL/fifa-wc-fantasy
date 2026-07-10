"""Generate the report dataset and figures the dashboard pages consume.

    python -m fifa_fantasy.report [--root DIR] [--skip-figures]

Writes results/report/report_data.json plus results/figures/*.svg. Run by
the snapshot loop's FIFA tick between the optimizer and the page writer;
also runnable by hand after any evaluation artifact changes. Matplotlib
is imported through report.figures with the Agg backend, so no display is
needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data, figures
from .registry import MODEL_REGISTRY


def generate(root: Path, skip_figures: bool = False) -> dict:
    """Assemble the dataset, write it, and render every figure that has data."""
    report_dir = root / "results" / "report"
    figures_dir = root / "results" / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)

    bundle = data.assemble(root)
    written: list[str] = []

    if not skip_figures:
        eval_dir = root / "data" / "evaluation"
        processed_dir = root / "data" / "processed"
        signals_dir = root / "data" / "external" / "player_signals"
        markets_dir = root / "data" / "external" / "prediction_markets"

        calib = data.load_calibration(processed_dir)
        signals = data.load_signals(signals_dir)
        market_history = data.load_market_history(markets_dir)

        jobs = [
            figures.fig_backtest_cumulative(
                bundle["backtest"], figures_dir / "fig_backtest_cumulative.svg"),
            figures.fig_backtest_rounds(
                bundle["backtest"], figures_dir / "fig_backtest_rounds.svg"),
            figures.fig_holdout_rmse(
                bundle["validation"], bundle["routing"],
                figures_dir / "fig_holdout_rmse.svg"),
            figures.fig_walkforward(
                bundle["walkforward"], figures_dir / "fig_walkforward.svg"),
            figures.fig_calibration(
                calib, figures_dir / "fig_calibration.svg"),
            figures.fig_gk_sweep(
                bundle["gk_sweep"], figures_dir / "fig_gk_sweep.svg"),
            figures.fig_market_negative(
                data.load_market_negative(eval_dir),
                figures_dir / "fig_market_negative.svg"),
            figures.fig_market_odds(
                market_history, figures_dir / "fig_market_odds.svg"),
            figures.fig_signal_volume(
                data.signal_volume_by_day(signals),
                figures_dir / "fig_signal_volume.svg"),
        ]
        for record in MODEL_REGISTRY:
            jobs.append(figures.render_formula(
                record.formula_mathtext,
                figures_dir / f"formula_{record.key}.svg"))
        written = [str(p) for p in jobs if p is not None]

    bundle["figures_written"] = sorted(Path(p).name for p in written)
    out_path = report_dir / "report_data.json"
    out_path.write_text(json.dumps(bundle, indent=1, default=str))
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build report_data.json and the dashboard figures.")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="repo root (default: current directory)")
    parser.add_argument("--skip-figures", action="store_true",
                        help="write only report_data.json")
    args = parser.parse_args()

    bundle = generate(args.root, skip_figures=args.skip_figures)
    n_figs = len(bundle.get("figures_written", []))
    print(f"report_data.json written; {n_figs} figures; "
          f"official={bundle['official_backend']} "
          f"stage={bundle['official_stage']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
