"""Render a recommendation as a markdown squad table.

The output is intentionally minimal: a single ASCII markdown table
containing only the 15-player squad. Captain/vice/bench-position info is
in the Role column. Anything richer (diffs, alternatives, captain
narrative) belongs in the JSON next to the markdown so a UI can decide
what to render.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

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
    squad_player_ids: list[int],
    starter_ids: list[int],
    bench_ids_priority_order: list[int],
    captain_id: int,
    vice_captain_id: int,
    players: pd.DataFrame,
    round_predictions: pd.DataFrame,
    target_round: int,
) -> str:
    """Return the squad table as a markdown string."""
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
    return _format_table(
        rows,
        ["Role", "Player", "Cty", "Pos", "Price", "Own%", "Opp",
         f"R{target_round} E[pts]"],
    ) + "\n"
