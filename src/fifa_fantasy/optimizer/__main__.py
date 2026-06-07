"""CLI entry point.

    python -m fifa_fantasy.optimizer
    python -m fifa_fantasy.optimizer --stage GROUP_MD1 --predictions-dir data/processed

Default behaviour for the pre-tournament selection: optimize the 15-player
squad over MD1+MD2+MD3 (sum of effective points after scouting bonus),
then solve the MD1 lineup + captain.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fifa_fantasy.collector.schemas import Stage

from .pipeline import aggregate_to_player, apply_scouting_bonus
from .solvers import solve_lineup, solve_squad
from .stage_config import DEFAULT_ROUND_HORIZON, STAGE_CONFIGS

DEFAULT_DIR = Path("data/processed")


def _latest(dir_: Path, prefix: str) -> Path:
    matches = sorted(dir_.glob(f"{prefix}_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {prefix}_*.parquet under {dir_}")
    return matches[-1]


def _select_round(predictions: pd.DataFrame, round_id: int, player_ids: list[int]) -> pd.DataFrame:
    rows = predictions[
        (predictions["round_id"] == round_id)
        & (predictions["player_id"].isin(player_ids))
    ]
    return rows[["player_id", "full_name", "position", "country", "country_abbr",
                 "price_millions", "predicted_points", "is_home", "opponent_abbr"]]


def main() -> None:
    parser = argparse.ArgumentParser(prog="fifa_fantasy.optimizer")
    parser.add_argument("--stage", default=Stage.GROUP_MD1.value,
                        choices=[s.value for s in Stage])
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()

    stage = Stage(args.stage)
    config = STAGE_CONFIGS[stage]
    horizon = DEFAULT_ROUND_HORIZON[stage]
    first_round = horizon[0]

    predictions = pd.read_parquet(_latest(args.predictions_dir, "predictions"))
    predictions = apply_scouting_bonus(predictions)
    player_table = aggregate_to_player(predictions, horizon)

    squad = solve_squad(player_table, config)
    print(f"Stage: {stage.value}   horizon: {list(horizon)}")
    print(f"Budget: ${squad.budget_used:.1f}M / ${config.budget_millions:.1f}M  "
          f"(remaining ${config.budget_millions - squad.budget_used:.1f}M)")
    print(f"Total horizon points: {squad.objective:.2f}")

    # Solve lineup for the first round in the horizon.
    squad_in_round = _select_round(predictions, first_round, squad.player_ids)
    lineup = solve_lineup(squad_in_round)

    starters = squad_in_round[squad_in_round["player_id"].isin(lineup.starter_ids)]
    bench = squad_in_round.set_index("player_id").loc[lineup.bench_ids].reset_index()
    captain = squad_in_round[squad_in_round["player_id"] == lineup.captain_id].iloc[0]
    vice = squad_in_round[squad_in_round["player_id"] == lineup.vice_captain_id].iloc[0]

    print()
    print(f"=== Starting XI ({lineup.formation}) for round {first_round} "
          f"— expected {lineup.objective:.2f} pts ===")
    print(starters.sort_values(["position", "predicted_points"], ascending=[True, False])[
        ["full_name", "country_abbr", "position", "price_millions", "is_home",
         "opponent_abbr", "predicted_points"]
    ].round(2).to_string(index=False))
    print()
    print("=== Bench (auto-sub priority order) ===")
    print(bench[
        ["full_name", "country_abbr", "position", "price_millions", "predicted_points"]
    ].round(2).to_string(index=False))
    print()
    print(f"Captain:      {captain.full_name} ({captain.country_abbr}, "
          f"E={captain.predicted_points:.2f} → {2*captain.predicted_points:.2f} doubled)")
    print(f"Vice-captain: {vice.full_name} ({vice.country_abbr}, E={vice.predicted_points:.2f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = args.out_dir / f"recommendation_{stage.value}_{date}.json"
    payload = {
        "stage": stage.value,
        "horizon_rounds": list(horizon),
        "budget_used": squad.budget_used,
        "budget_total": config.budget_millions,
        "total_horizon_points": squad.objective,
        "squad_player_ids": squad.player_ids,
        "lineup": {
            "round_id": first_round,
            "formation": lineup.formation,
            "starter_ids": lineup.starter_ids,
            "bench_ids_priority_order": lineup.bench_ids,
            "captain_id": lineup.captain_id,
            "vice_captain_id": lineup.vice_captain_id,
            "expected_points": lineup.objective,
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nrecommendation written → {out_path}")


if __name__ == "__main__":
    main()
