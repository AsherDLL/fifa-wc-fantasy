"""CLI entry point.

    python -m fifa_fantasy.optimizer                                # MD1 (fresh)
    python -m fifa_fantasy.optimizer --stage GROUP_MD2 \\
        --from results/<host>_recommendation_<backend>_GROUP_MD1_<ts>.json  # transfer

Outputs per run, both into --out-dir (default `results/`):

  <host>_recommendation_<backend>_<STAGE>_<UTC-timestamp>.json
  <host>_recommendation_<backend>_<STAGE>_<UTC-timestamp>.md

The JSON carries the full payload (squad, lineup, captain, transfer
block if any) for a UI to consume. The markdown is the squad table only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fifa_fantasy.collector.schemas import Stage

from .captain import select_captain_vice
from .pipeline import (
    aggregate_to_player, apply_availability_discount, apply_scouting_bonus,
)
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
    cols = ["player_id", "full_name", "position", "country", "country_abbr",
            "price_millions", "predicted_points", "is_home", "opponent_abbr"]
    # Carry the discounted score so solve_lineup optimizes what solve_squad
    # optimized. Dropping it here silently reverted the XI to raw points.
    if "effective_points" in rows.columns:
        cols.append("effective_points")
    return rows[cols]


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
    parser.add_argument(
        "--standings-pos-pct", type=float, default=0.9,
        help=("user's league position as a percentile: 0.0 = leader, "
              "1.0 = last. Drives the captain ceiling weighting; the "
              "default 0.9 reflects a team near the bottom that needs "
              "upside. Set lower when protecting a lead."))
    args = parser.parse_args()

    stage = Stage(args.stage)
    config = STAGE_CONFIGS[stage]
    horizon = DEFAULT_ROUND_HORIZON[stage]
    first_round = horizon[0]

    predictions = pd.read_parquet(_latest(args.predictions_dir, "predictions"))
    backend = (
        str(predictions["model_backend"].iloc[0])
        if "model_backend" in predictions.columns and len(predictions) > 0
        else "unknown"
    )
    model_version = (
        str(predictions["model_version"].iloc[0])
        if "model_version" in predictions.columns and len(predictions) > 0
        else ""
    )
    predictions = apply_scouting_bonus(predictions)
    predictions = apply_availability_discount(predictions)
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

    # Override the lineup solver's naive mean-argmax captain with the
    # ceiling-aware composite selector. On the starting XI it weights the
    # q90 ceiling by how hard the user is chasing, which captured nearly
    # twice the captain points of mean-argmax on the leak-free backtest
    # (docs 11g). Falls back cleanly to the mean when quantiles are absent
    # (non-GBM backends): _row_to_dict defaults p90/p10 to predicted_points.
    xi_ctx = predictions[
        (predictions["round_id"] == first_round)
        & (predictions["player_id"].isin(lineup.starter_ids))
    ].copy()
    if "ownership_fraction" in xi_ctx.columns:
        xi_ctx["ownership_pct"] = xi_ctx["ownership_fraction"].astype(float) * 100.0
    # The composite selector reads predicted_p10/p90; the GBM emits q10/q90.
    for q, p in (("predicted_q10", "predicted_p10"), ("predicted_q90", "predicted_p90")):
        if q in xi_ctx.columns and p not in xi_ctx.columns:
            xi_ctx[p] = xi_ctx[q]
    captain_id, vice_captain_id = lineup.captain_id, lineup.vice_captain_id
    try:
        decision = select_captain_vice(
            xi_ctx, standings_pos_pct=args.standings_pos_pct)
        captain_id, vice_captain_id = decision.captain_id, decision.vice_id
    except (ValueError, KeyError, IndexError):
        pass  # keep the solver's picks if the selector cannot run

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    # Filename includes the model backend so heuristic vs gbm outputs do
    # not get mixed up. Pattern:
    #   <host>_recommendation_<backend>_<stage>_<UTC-timestamp>.{json,md}
    prefix = f"{_hostname()}_recommendation_{backend}_{stage.value}_{ts}"
    json_path = args.out_dir / f"{prefix}.json"
    md_path = args.out_dir / f"{prefix}.md"

    # Build the rich "squad" array consumed by the web UI: one entry per
    # squad member with name, country, position, price, ownership, role,
    # opponent context, predicted_points, and (when available) GBM
    # quantile bands. The flat squad_player_ids / starter_ids fields are
    # kept for backward-compatible consumers.
    squad_in_round_idx = squad_in_round.set_index("player_id")
    full_players = predictions[
        [c for c in predictions.columns
         if c in ("player_id", "full_name", "first_name", "last_name",
                  "known_name", "position", "country", "country_abbr",
                  "squad_id", "price_millions", "ownership_fraction",
                  "status", "is_eliminated", "one_to_watch",
                  "one_to_watch_text", "total_points", "last_round_points",
                  "form")]
    ].drop_duplicates("player_id").set_index("player_id")

    starter_set = set(lineup.starter_ids)
    bench_pos = {pid: i + 1 for i, pid in enumerate(lineup.bench_ids)}
    quantile_cols = [c for c in ("predicted_q10", "predicted_q50", "predicted_q90")
                     if c in predictions.columns]

    def _role(pid: int) -> str:
        if pid == captain_id:
            return "Captain"
        if pid == vice_captain_id:
            return "Vice"
        if pid in starter_set:
            return "Start"
        return f"Bench {bench_pos[pid]}"

    squad_array = []
    for pid in chosen_ids:
        meta = full_players.loc[pid]
        ctx = squad_in_round_idx.loc[pid]
        entry = {
            "player_id": pid,
            "full_name": meta["full_name"],
            "first_name": meta.get("first_name"),
            "last_name": meta.get("last_name"),
            "known_name": meta.get("known_name"),
            "position": meta["position"],
            "country": meta["country"],
            "country_abbr": meta["country_abbr"],
            "squad_id": int(meta["squad_id"]),
            "price_millions": float(meta["price_millions"]),
            "ownership_fraction": float(meta["ownership_fraction"]),
            "status": meta.get("status"),
            "is_eliminated": bool(meta.get("is_eliminated", False)),
            "one_to_watch": bool(meta.get("one_to_watch", False)),
            "one_to_watch_text": meta.get("one_to_watch_text"),
            "form": float(meta["form"]) if "form" in meta and meta["form"] is not None else None,
            "role": _role(pid),
            "in_starting_xi": pid in starter_set,
            "bench_priority": bench_pos.get(pid),
            "opponent_abbr": ctx["opponent_abbr"],
            "is_home": bool(ctx["is_home"]),
            "predicted_points": float(ctx["predicted_points"]),
        }
        # The score the solvers actually optimized (scouting bonus +
        # availability discount). Without it the payload's per-player
        # numbers cannot be reconciled with total_horizon_points.
        if "effective_points" in ctx:
            entry["effective_points"] = float(ctx["effective_points"])
        # Optional quantile bands when the GBM backend was used.
        for q in quantile_cols:
            preds_q_row = predictions[
                (predictions["round_id"] == first_round)
                & (predictions["player_id"] == pid)
            ]
            if not preds_q_row.empty and q in preds_q_row.columns:
                entry[q] = float(preds_q_row[q].iloc[0])
        # NaN is not valid JSON (json.dumps emits a bare NaN token that
        # JSON.parse rejects). Nulls instead.
        for key, val in list(entry.items()):
            if isinstance(val, float) and math.isnan(val):
                entry[key] = None
        squad_array.append(entry)

    payload: dict = {
        "stage": stage.value,
        "model_backend": backend,
        "model_version": model_version,
        "host": _hostname(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_rounds": list(horizon),
        "budget_used": budget_used,
        "budget_total": config.budget_millions,
        "total_horizon_points": gross_objective,
        "net_horizon_points": net_objective,
        "squad_player_ids": chosen_ids,
        "squad": squad_array,
        "lineup": {
            "round_id": first_round,
            "formation": lineup.formation,
            "starter_ids": lineup.starter_ids,
            "bench_ids_priority_order": lineup.bench_ids,
            "captain_id": captain_id,
            "vice_captain_id": vice_captain_id,
            "expected_points": lineup.objective,
        },
    }
    if transfer is not None:
        # Resolve player details for both sides so the JSON is self-contained.
        # IN players are in the new squad and live in `full_players`; OUT
        # players were in the previous squad and we read their details from
        # the previous JSON if available (it has the `squad` array since
        # the rich-JSON change), otherwise fall back to bare ids.
        prev_squad_lookup: dict[int, dict] = {}
        if args.from_json is not None:
            try:
                prev = json.loads(args.from_json.read_text())
                for ent in prev.get("squad", []):
                    prev_squad_lookup[int(ent["player_id"])] = ent
            except Exception:
                pass

        def _detail_in(pid: int) -> dict:
            meta = full_players.loc[pid]
            return {
                "player_id": int(pid),
                "full_name": str(meta["full_name"]),
                "country": str(meta["country"]),
                "country_abbr": str(meta["country_abbr"]),
                "position": str(meta["position"]),
                "price_millions": float(meta["price_millions"]),
            }

        def _detail_out(pid: int) -> dict:
            ent = prev_squad_lookup.get(int(pid))
            if ent is not None:
                return {
                    "player_id": int(pid),
                    "full_name": ent.get("full_name"),
                    "country": ent.get("country"),
                    "country_abbr": ent.get("country_abbr"),
                    "position": ent.get("position"),
                    "price_millions": ent.get("price_millions"),
                }
            return {"player_id": int(pid)}

        payload["transfer"] = {
            "from": str(args.from_json),
            "rolled_over_free_transfers": args.rolled_over,
            "free_transfers_total": (
                None if config.free_transfers is None
                else (config.free_transfers + args.rolled_over)
            ),
            "n_transfers": transfer.n_transfers,
            "n_extra_transfers": transfer.n_extra_transfers,
            "transfer_cost_points": transfer.transfer_cost_points,
            "transfers_in": transfer.transfers_in,
            "transfers_out": transfer.transfers_out,
            "transfers_in_detail": [_detail_in(pid) for pid in transfer.transfers_in],
            "transfers_out_detail": [_detail_out(pid) for pid in transfer.transfers_out],
        }
    json_path.write_text(json.dumps(payload, indent=2))

    md_player_cols = ["player_id", "full_name", "position", "country",
                      "country_abbr", "price_millions", "ownership_fraction"]
    if "one_to_watch" in predictions.columns:
        md_player_cols.append("one_to_watch")
    players_for_report = predictions[md_player_cols].drop_duplicates("player_id")

    md_path.write_text(render_markdown(
        stage=stage.value,
        backend=backend,
        generated_at_utc=payload["generated_at_utc"],
        squad_player_ids=chosen_ids,
        starter_ids=lineup.starter_ids,
        bench_ids_priority_order=lineup.bench_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        players=players_for_report,
        round_predictions=squad_in_round,
        target_round=first_round,
        transfer=transfer,
        transfers_out_detail=payload.get("transfer", {}).get("transfers_out_detail"),
    ))

    print(f"json  {json_path}")
    print(f"md    {md_path}")


if __name__ == "__main__":
    main()
