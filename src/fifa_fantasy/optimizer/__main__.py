"""CLI entry point.

    python -m fifa_fantasy.optimizer                                # MD1 (fresh)
    python -m fifa_fantasy.optimizer --stage GROUP_MD2 \\
        --from results/<host>_recommendation_GROUP_MD1_<ts>.json    # transfer

Outputs per run, both into --out-dir (default `results/`):

  <host>_recommendation_<STAGE>_<UTC-timestamp>.json
  <host>_recommendation_<STAGE>_<UTC-timestamp>.md

The JSON carries the full payload (squad, lineup, captain, transfer
block if any) for a UI to consume. The markdown is the squad table only.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fifa_fantasy.collector.schemas import Stage

from .pipeline import aggregate_to_player, apply_scouting_bonus
from .report import render_markdown
from .solvers import (
    TransferSolution,
    solve_lineup,
    solve_squad,
    solve_transfer,
)
from .stage_config import DEFAULT_ROUND_HORIZON, STAGE_CONFIGS

DEFAULT_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("results")


def _hostname() -> str:
    raw = socket.gethostname() or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


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
    parser.add_argument("--from", dest="from_json", type=Path, default=None,
                        help="previous recommendation JSON to transfer from")
    parser.add_argument("--rolled-over", type=int, default=0,
                        help="rolled-over free transfers from the prior round")
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    stage = Stage(args.stage)
    config = STAGE_CONFIGS[stage]
    horizon = DEFAULT_ROUND_HORIZON[stage]
    first_round = horizon[0]

    predictions = pd.read_parquet(_latest(args.predictions_dir, "predictions"))
    predictions = apply_scouting_bonus(predictions)
    player_table = aggregate_to_player(predictions, horizon)

    transfer: TransferSolution | None = None
    if args.from_json is not None:
        previous = json.loads(args.from_json.read_text())
        current_squad_ids = previous["squad_player_ids"]
        transfer = solve_transfer(
            player_table,
            current_squad_ids=current_squad_ids,
            config=config,
            rolled_over_transfers=args.rolled_over,
        )
        chosen_ids = transfer.player_ids
        gross_objective = transfer.gross_objective
        net_objective = transfer.objective
        budget_used = transfer.budget_used
    else:
        squad = solve_squad(player_table, config)
        chosen_ids = squad.player_ids
        gross_objective = squad.objective
        net_objective = squad.objective
        budget_used = squad.budget_used

    squad_in_round = _select_round(predictions, first_round, chosen_ids)
    lineup = solve_lineup(squad_in_round)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    prefix = f"{_hostname()}_recommendation_{stage.value}_{ts}"
    json_path = args.out_dir / f"{prefix}.json"
    md_path = args.out_dir / f"{prefix}.md"

    payload: dict = {
        "stage": stage.value,
        "host": _hostname(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_rounds": list(horizon),
        "budget_used": budget_used,
        "budget_total": config.budget_millions,
        "total_horizon_points": gross_objective,
        "net_horizon_points": net_objective,
        "squad_player_ids": chosen_ids,
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
    if transfer is not None:
        payload["transfer"] = {
            "from": str(args.from_json),
            "rolled_over_free_transfers": args.rolled_over,
            "free_transfers_total": (
                None if config.free_transfers is None
                else (config.free_transfers + args.rolled_over)
            ),
            "transfers_in": transfer.transfers_in,
            "transfers_out": transfer.transfers_out,
            "n_transfers": transfer.n_transfers,
            "n_extra_transfers": transfer.n_extra_transfers,
            "transfer_cost_points": transfer.transfer_cost_points,
        }
    json_path.write_text(json.dumps(payload, indent=2))

    md_player_cols = ["player_id", "full_name", "position", "country",
                      "country_abbr", "price_millions", "ownership_fraction"]
    if "one_to_watch" in predictions.columns:
        md_player_cols.append("one_to_watch")
    players_for_report = predictions[md_player_cols].drop_duplicates("player_id")

    md_path.write_text(render_markdown(
        squad_player_ids=chosen_ids,
        starter_ids=lineup.starter_ids,
        bench_ids_priority_order=lineup.bench_ids,
        captain_id=lineup.captain_id,
        vice_captain_id=lineup.vice_captain_id,
        players=players_for_report,
        round_predictions=squad_in_round,
        target_round=first_round,
    ))

    print(f"json  {json_path}")
    print(f"md    {md_path}")


if __name__ == "__main__":
    main()
