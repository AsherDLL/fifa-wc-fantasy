"""Render a recommendation as a minimal markdown report.

Each file starts with a fact-only title line listing the backend, stage,
and generation timestamp, followed by the squad table. If the run was
transfer-mode (`--from`), an IN/OUT section is appended below the table.

No prose, no analysis, no narrative. Just data the optimizer produced.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .solvers import TransferSolution

POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _format_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def render_markdown(
    *,
    stage: str,
    backend: str,
    generated_at_utc: str,
    squad_player_ids: list[int],
    starter_ids: list[int],
    bench_ids_priority_order: list[int],
    captain_id: int,
    vice_captain_id: int,
    players: pd.DataFrame,
    round_predictions: pd.DataFrame,
    target_round: int,
    transfer: TransferSolution | None = None,
    transfers_out_detail: list[dict] | None = None,
) -> str:
    """Squad table plus, optionally, an IN/OUT section for transfer mode."""
    if "player_id" in players.columns and players.index.name != "player_id":
        players = players.set_index("player_id")
    round_idx = round_predictions.set_index("player_id")

    starter_set = set(starter_ids)
    bench_pos = {pid: i + 1 for i, pid in enumerate(bench_ids_priority_order)}

    def role(pid: int) -> str:
        if pid == captain_id:
            return "Captain (C)"
        if pid == vice_captain_id:
            return "Vice (V)"
        if pid in starter_set:
            return "Start"
        return f"Bench {bench_pos[pid]}"

    has_otw = "one_to_watch" in players.columns
    rows = []
    for pid in squad_player_ids:
        p = players.loc[pid]
        r = round_idx.loc[pid]
        name = p["full_name"]
        if has_otw and bool(p.get("one_to_watch", False)):
            name = name + " [OTW]"
        rows.append({
            "_pos_sort": POSITION_ORDER[p["position"]],
            "_md_pred": float(r["predicted_points"]),
            "Role": role(pid),
            "Player": name,
            "Cty": p["country_abbr"],
            "Pos": p["position"],
            "Price": f"${float(p['price_millions']):.1f}M",
            "Own%": f"{float(p['ownership_fraction']) * 100:.1f}%",
            "Opp": ("vs " if bool(r["is_home"]) else "@ ") + str(r["opponent_abbr"]),
            f"R{target_round} E[pts]": f"{float(r['predicted_points']):.2f}",
        })
    rows.sort(key=lambda x: (x["_pos_sort"], -x["_md_pred"]))
    table = _format_table(
        rows,
        ["Role", "Player", "Cty", "Pos", "Price", "Own%", "Opp",
         f"R{target_round} E[pts]"],
    )

    title = (
        f"# Recommendation\n\n"
        f"- stage: {stage}\n"
        f"- backend: {backend}\n"
        f"- generated_at_utc: {generated_at_utc}\n"
    )

    transfer_section = ""
    if transfer is not None and (transfer.transfers_in or transfer.transfers_out):
        out_lookup = {
            int(d["player_id"]): d
            for d in (transfers_out_detail or [])
            if "player_id" in d
        }

        def _line_in(pid: int) -> str:
            p = players.loc[pid]
            return (f"- {p['full_name']} ({p['country_abbr']}, "
                    f"{p['position']}, ${float(p['price_millions']):.1f}M)")

        def _line_out(pid: int) -> str:
            d = out_lookup.get(int(pid))
            if d is not None and d.get("full_name"):
                price = d.get("price_millions")
                price_s = f", ${float(price):.1f}M" if price is not None else ""
                return (f"- {d['full_name']} ({d.get('country_abbr', '?')}, "
                        f"{d.get('position', '?')}{price_s})")
            return f"- player_id {pid}"

        in_lines = [_line_in(pid) for pid in transfer.transfers_in] or ["- (none)"]
        out_lines = [_line_out(pid) for pid in transfer.transfers_out] or ["- (none)"]
        transfer_section = (
            "\n\n## Transfers from previous squad\n\n"
            f"- transfers_made: {transfer.n_transfers}\n"
            f"- extra_above_free_quota: {transfer.n_extra_transfers}\n"
            f"- hit_points: -{transfer.transfer_cost_points}\n"
            "\n### IN\n\n"
            + "\n".join(in_lines)
            + "\n\n### OUT\n\n"
            + "\n".join(out_lines)
        )

    return title + "\n" + table + transfer_section + "\n"
